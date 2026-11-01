"""Unit tests for flax_lora.injector and flax_lora.optimizer."""

import pytest
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

from flax_lora.layers import LoRALinear
from flax_lora.params import LoRAParam
from flax_lora.injector import (
    inject_lora,
    eject_lora,
    merge_lora,
    unmerge_lora,
    get_lora_summary,
)
from flax_lora.optimizer import create_lora_optimizer, lora_value_and_grad


class SimpleBlock(nnx.Module):
    def __init__(self, d_in: int, d_out: int, *, rngs: nnx.Rngs):
        self.attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=d_in,
            qkv_features=d_in,
            out_features=d_out,
            decode=False,
            rngs=rngs,
        )
        self.mlp = nnx.Linear(d_out, d_out, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.attn(x)
        return self.mlp(h)


class SimpleTransformer(nnx.Module):
    def __init__(self, num_blocks: int = 2, dim: int = 16, *, rngs: nnx.Rngs):
        self.blocks = nnx.List([SimpleBlock(dim, dim, rngs=rngs) for _ in range(num_blocks)])
        self.head = nnx.Linear(dim, 8, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def test_inject_lora_targets():
    """Verify selective injection into targeted layers."""
    rngs = nnx.Rngs(42)
    model = SimpleTransformer(num_blocks=2, dim=16, rngs=rngs)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 16))
    base_out = model(x)

    injected = inject_lora(model, target_layers=("query", "value"), rank=4, alpha=8.0, rngs=rngs)

    # 2 blocks * 2 targets (query, value) = 4 injected layers
    assert len(injected) == 4
    for layer in injected:
        assert isinstance(layer, LoRALinear)

    # Output at init must match base output exactly
    lora_out = model(x)
    assert jnp.allclose(base_out, lora_out, atol=1e-6)

    # Verify summary
    summary = get_lora_summary(model)
    assert summary["trainable_lora_parameters"] > 0
    assert summary["trainable_percentage"] < 50.0


def test_eject_lora():
    """Verify eject_lora cleanly restores original module architecture."""
    rngs = nnx.Rngs(42)
    model = SimpleTransformer(num_blocks=2, dim=16, rngs=rngs)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 16))
    orig_out = model(x)

    inject_lora(model, target_layers=("query", "value"), rank=4, rngs=rngs)
    ejected_count = eject_lora(model)
    assert ejected_count == 4

    # Check that no LoRALinear or LoRAParam remain
    summary = get_lora_summary(model)
    assert summary["trainable_lora_parameters"] == 0
    restored_out = model(x)
    assert jnp.allclose(orig_out, restored_out, atol=1e-6)


def test_merge_and_unmerge_tree():
    """Verify tree-wide merge_lora and unmerge_lora."""
    rngs = nnx.Rngs(42)
    model = SimpleTransformer(num_blocks=2, dim=16, rngs=rngs)
    injected = inject_lora(model, target_layers=("query", "value"), rank=4, alpha=8.0, rngs=rngs)

    # Set non-zero weights on adapters
    for i, layer in enumerate(injected):
        layer.lora_b[...] = jax.random.normal(jax.random.PRNGKey(10 + i), layer.lora_b.shape)

    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 16))
    out_unmerged = model(x)

    # Merge all
    merged_count = merge_lora(model)
    assert merged_count == 4
    out_merged = model(x)
    assert jnp.allclose(out_unmerged, out_merged, atol=1e-5)

    # Unmerge all
    unmerged_count = unmerge_lora(model)
    assert unmerged_count == 4
    out_unmerged_again = model(x)
    assert jnp.allclose(out_unmerged, out_unmerged_again, atol=1e-5)


def test_optimizer_freezes_base_weights():
    """Verify that create_lora_optimizer only updates LoRAParam, leaving base weights frozen."""
    rngs = nnx.Rngs(42)
    model = SimpleTransformer(num_blocks=1, dim=8, rngs=rngs)
    inject_lora(model, target_layers=("mlp",), rank=2, alpha=4.0, rngs=rngs)

    # Snapshot base weights
    base_kernel_init = jnp.array(model.blocks[0].mlp.base_linear.kernel[...])

    optimizer = create_lora_optimizer(model, tx=optax.sgd(learning_rate=0.1))

    @nnx.jit
    def train_step(m, opt, batch):
        def loss_fn(model, x_in):
            out = model(x_in)
            return jnp.mean(out ** 2)

        grad_fn = lora_value_and_grad(loss_fn)
        loss, grads = grad_fn(m, batch)
        opt.update(m, grads)
        return loss

    x = jnp.ones((2, 4, 8))
    # Run multiple steps
    for _ in range(5):
        loss = train_step(model, optimizer, x)

    # Base weights must remain EXACTLY identical
    assert jnp.array_equal(model.blocks[0].mlp.base_linear.kernel[...], base_kernel_init)

    # Adapter weights must have changed from 0
    lora_layer = model.blocks[0].mlp
    assert not jnp.allclose(lora_layer.lora_b[...], jnp.zeros_like(lora_layer.lora_b[...]))
