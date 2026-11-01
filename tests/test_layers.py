"""Unit tests for flax_lora.layers.LoRALinear."""

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from flax_lora.layers import LoRALinear
from flax_lora.params import LoRAParam, is_lora_param


def test_zero_initialization():
    """Verify that at initialization (B=0), LoRALinear output matches base linear output."""
    rngs = nnx.Rngs(42)
    base = nnx.Linear(8, 16, rngs=rngs)
    lora = LoRALinear(base, rank=4, alpha=16.0, rngs=rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 8))
    base_out = base(x)
    lora_out = lora(x)

    assert jnp.allclose(base_out, lora_out, atol=1e-7)
    assert is_lora_param(lora.lora_a)
    assert is_lora_param(lora.lora_b)


def test_forward_computation():
    """Verify forward computation when B is non-zero."""
    rngs = nnx.Rngs(42)
    base = nnx.Linear(4, 4, use_bias=False, rngs=rngs)
    lora = LoRALinear(base, rank=2, alpha=4.0, rngs=rngs)

    # Set known weights
    lora.lora_b[...] = jnp.ones_like(lora.lora_b[...])
    x = jnp.ones((1, 4))

    base_out = base(x)
    lora_out = lora(x)

    # Manual delta calculation: x @ A @ B * (alpha / rank)
    expected_delta = (x @ lora.lora_a[...] @ lora.lora_b[...]) * (4.0 / 2.0)
    assert jnp.allclose(lora_out, base_out + expected_delta, atol=1e-6)


def test_merge_and_unmerge_linear():
    """Verify that merge() matches unmerged output and unmerge() restores base weights."""
    rngs = nnx.Rngs(42)
    base = nnx.Linear(16, 32, rngs=rngs)
    lora = LoRALinear(base, rank=4, alpha=8.0, rngs=rngs)

    # Inject random weights into B so delta is non-trivial
    lora.lora_b[...] = jax.random.normal(jax.random.PRNGKey(1), lora.lora_b.shape)

    original_kernel = jnp.array(base.kernel[...])
    x = jax.random.normal(jax.random.PRNGKey(2), (3, 16))

    out_unmerged = lora(x)

    # Merge
    lora.merge()
    assert lora.is_merged
    out_merged = lora(x)
    assert jnp.allclose(out_unmerged, out_merged, atol=1e-6)

    # Unmerge
    lora.unmerge()
    assert not lora.is_merged
    assert jnp.allclose(base.kernel[...], original_kernel, atol=1e-6)
    out_unmerged_again = lora(x)
    assert jnp.allclose(out_unmerged, out_unmerged_again, atol=1e-6)


def test_linear_general_mha():
    """Verify LoRALinear works seamlessly with nnx.LinearGeneral inside MultiHeadAttention."""
    rngs = nnx.Rngs(42)
    mha = nnx.MultiHeadAttention(
        num_heads=4,
        in_features=16,
        qkv_features=16,
        out_features=16,
        decode=False,
        rngs=rngs,
    )

    lora_q = LoRALinear(mha.query, rank=4, alpha=8.0, rngs=rngs)
    lora_out = LoRALinear(mha.out, rank=4, alpha=8.0, rngs=rngs)

    lora_q.lora_b[...] = jax.random.normal(jax.random.PRNGKey(3), lora_q.lora_b.shape)
    lora_out.lora_b[...] = jax.random.normal(jax.random.PRNGKey(4), lora_out.lora_b.shape)

    x = jax.random.normal(jax.random.PRNGKey(5), (2, 4, 16))

    # Test Query adapter merge/unmerge
    q_out_unmerged = lora_q(x)
    lora_q.merge()
    assert jnp.allclose(q_out_unmerged, lora_q(x), atol=1e-6)
    lora_q.unmerge()
    assert jnp.allclose(q_out_unmerged, lora_q(x), atol=1e-6)

    # Test Out adapter merge/unmerge
    x_heads = jax.random.normal(jax.random.PRNGKey(6), (2, 4, 4, 4))
    out_unmerged = lora_out(x_heads)
    lora_out.merge()
    assert jnp.allclose(out_unmerged, lora_out(x_heads), atol=1e-6)
    lora_out.unmerge()
    assert jnp.allclose(out_unmerged, lora_out(x_heads), atol=1e-6)


def test_invalid_parameters():
    """Verify validation on rank and alpha."""
    rngs = nnx.Rngs(42)
    base = nnx.Linear(8, 8, rngs=rngs)

    with pytest.raises(ValueError, match="rank must be a positive integer"):
        LoRALinear(base, rank=0, rngs=rngs)

    with pytest.raises(ValueError, match="alpha must be positive"):
        LoRALinear(base, rank=4, alpha=-1.0, rngs=rngs)
