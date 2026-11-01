"""Lightweight adapter persistence for Flax NNX LoRA using Orbax."""

from typing import Optional, Dict, Any, Union
import json
import pathlib
import jax
import orbax.checkpoint as ocp
import flax.nnx as nnx

from flax_lora.params import LoRAParam
from flax_lora.layers import LoRALinear


def save_adapter(
    save_path: Union[str, pathlib.Path],
    model: nnx.Module,
    metadata: Optional[Dict[str, Any]] = None,
) -> pathlib.Path:
    """Serializes only LoRAParam adapter weights and configuration metadata to disk.

    Creates an Orbax PyTree checkpoint containing only low-rank adapter weights
    (< 1 MB for typical models), alongside an `adapter_config.json` metadata file.
    Base model weights are excluded.

    Args:
        save_path: Directory path where the adapter checkpoint should be saved.
        model: NNX model containing trained LoRAParam weights.
        metadata: Optional dictionary with user-defined metadata (e.g. dataset, loss, epochs).

    Returns:
        Pathlib Path to the saved adapter directory.
    """
    save_path = pathlib.Path(save_path).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Extract only LoRAParam PyTree state
    lora_state = nnx.state(model, LoRAParam)

    # 2. Extract adapter configuration from first LoRALinear found
    adapter_rank = None
    adapter_alpha = None
    dropout_rate = 0.0
    for _, mod in nnx.iter_modules(model):
        if isinstance(mod, LoRALinear):
            adapter_rank = mod.rank
            adapter_alpha = mod.alpha
            dropout_rate = mod.dropout_rate
            break

    config = {
        "rank": adapter_rank,
        "alpha": adapter_alpha,
        "dropout_rate": dropout_rate,
        "custom_metadata": metadata or {},
    }

    config_path = save_path / "adapter_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 3. Save PyTree via Orbax
    weights_path = save_path / "weights"
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(weights_path, lora_state, force=True)

    return save_path


def load_adapter(
    load_path: Union[str, pathlib.Path],
    model: nnx.Module,
    *,
    sharding: Optional[jax.sharding.Sharding] = None,
) -> Dict[str, Any]:
    """Restores adapter weights from an Orbax checkpoint and hydrates the model in-place.

    Args:
        load_path: Directory path where the adapter checkpoint was saved.
        model: NNX model with matching LoRALinear layers already injected.
        sharding: Target JAX sharding (defaults to SingleDeviceSharding on default device).

    Returns:
        The configuration and metadata dictionary loaded from `adapter_config.json`.
    """
    load_path = pathlib.Path(load_path).resolve()
    if not load_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint path does not exist: {load_path}")

    # 1. Read metadata if available
    config = {}
    config_path = load_path / "adapter_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    # 2. Prepare target restore PyTree template
    target_lora_state = nnx.state(model, LoRAParam)

    weights_path = load_path / "weights"
    checkpointer = ocp.PyTreeCheckpointer()

    if sharding is None:
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    restore_args = jax.tree_util.tree_map(
        lambda leaf: ocp.ArrayRestoreArgs(sharding=sharding),
        target_lora_state,
    )

    restored_state = checkpointer.restore(
        weights_path,
        item=target_lora_state,
        restore_args=restore_args,
    )

    # 3. Hydrate model in-place
    nnx.update(model, restored_state)
    return config
