"""Unit tests for flax_lora.checkpoint."""

import tempfile
import pathlib
import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from flax_lora.injector import inject_lora
from flax_lora.checkpoint import save_adapter, load_adapter


class SmallModel(nnx.Module):
    def __init__(self, d: int = 16, *, rngs: nnx.Rngs):
        self.lin1 = nnx.Linear(d, d, rngs=rngs)
        self.lin2 = nnx.Linear(d, 4, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.lin2(nnx.relu(self.lin1(x)))


def test_save_and_load_adapter_roundtrip():
    """Verify that saved adapter weights restore a fresh model to identical outputs."""
    rngs = nnx.Rngs(42)
    model = SmallModel(d=16, rngs=rngs)
    inject_lora(model, target_layers=("lin1", "lin2"), rank=4, alpha=8.0, rngs=rngs)

    # Set non-zero weights on adapters
    model.lin1.lora_b[...] = jax.random.normal(jax.random.PRNGKey(1), model.lin1.lora_b.shape)
    model.lin2.lora_b[...] = jax.random.normal(jax.random.PRNGKey(2), model.lin2.lora_b.shape)

    x = jax.random.normal(jax.random.PRNGKey(3), (2, 16))
    expected_output = model(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = pathlib.Path(tmpdir) / "test_lora_adapter"
        metadata = {"task": "test", "step": 100}
        saved_path = save_adapter(save_dir, model, metadata=metadata)
        assert saved_path.exists()
        assert (saved_path / "adapter_config.json").exists()

        # Instantiate fresh model with identical base weights (using fresh Rngs(42))
        fresh_model = SmallModel(d=16, rngs=nnx.Rngs(42))
        inject_lora(fresh_model, target_layers=("lin1", "lin2"), rank=4, alpha=8.0, rngs=nnx.Rngs(99))

        # Before loading, outputs must differ
        fresh_output_before = fresh_model(x)
        assert not jnp.allclose(fresh_output_before, expected_output, atol=1e-5)

        # Load adapter
        loaded_config = load_adapter(saved_path, fresh_model)
        assert loaded_config["rank"] == 4
        assert loaded_config["custom_metadata"]["task"] == "test"

        # After loading, outputs must match exactly
        fresh_output_after = fresh_model(x)
        assert jnp.allclose(fresh_output_after, expected_output, atol=1e-6)


def test_adapter_checkpoint_size():
    """Verify that adapter checkpoint is lightweight (< 1 MB)."""
    rngs = nnx.Rngs(42)
    model = SmallModel(d=32, rngs=rngs)
    inject_lora(model, target_layers=("lin1", "lin2"), rank=4, rngs=rngs)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = pathlib.Path(tmpdir) / "lightweight_adapter"
        save_adapter(save_dir, model)

        total_bytes = sum(f.stat().st_size for f in save_dir.rglob("*") if f.is_file())
        total_kb = total_bytes / 1024.0
        # Should be well under 1000 KB (1 MB)
        assert total_kb < 1000.0, f"Adapter checkpoint size was {total_kb:.2f} KB"
