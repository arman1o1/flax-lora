"""Recursive model tree injection and management for Flax NNX LoRA."""

from typing import Callable, List, Optional, Sequence, Union, Dict, Any
import re
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from flax_lora.layers import LoRALinear
from flax_lora.params import LoRAParam, is_lora_param


def _matches_target(name: str, target_layers: Sequence[Union[str, Callable[[str, Any], bool]]], module: Any) -> bool:
    """Check if an attribute name or module matches target criteria."""
    for target in target_layers:
        if callable(target):
            if target(name, module):
                return True
        elif isinstance(target, str):
            if target.lower() in name.lower():
                return True
            try:
                if re.search(target, name, re.IGNORECASE):
                    return True
            except re.error:
                pass
    return False


def inject_lora(
    model: nnx.Module,
    target_layers: Sequence[Union[str, Callable[[str, Any], bool]]] = ("query", "value", "q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout_rate: float = 0.0,
    *,
    rngs: Optional[nnx.Rngs] = None,
) -> List[LoRALinear]:
    """Recursively injects LoRALinear adapters into matching layers of an NNX model in-place.

    Args:
        model: Root Flax NNX module to adapt.
        target_layers: Sequence of layer names/patterns (or filter callables) to target.
        rank: Rank r of the low-rank decomposition.
        alpha: Scaling factor alpha (scale = alpha / rank).
        dropout_rate: Dropout probability on adapter activations.
        rngs: Optional NNX Rngs instance for parameter initialization.

    Returns:
        List of all newly created and injected LoRALinear instances.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)

    injected_layers: List[LoRALinear] = []

    def _walk_and_inject(parent: Any, attr_name: Optional[str] = None):
        target_obj = getattr(parent, attr_name) if attr_name is not None else parent

        # If it's already a LoRALinear, don't re-wrap
        if isinstance(target_obj, LoRALinear):
            return

        # Check if target_obj is a linear layer with a kernel
        if hasattr(target_obj, "kernel") and isinstance(target_obj, nnx.Module):
            if attr_name is not None and _matches_target(attr_name, target_layers, target_obj):
                lora_layer = LoRALinear(
                    target_obj,
                    rank=rank,
                    alpha=alpha,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                )
                setattr(parent, attr_name, lora_layer)
                injected_layers.append(lora_layer)
                return

        # Handle container lists (e.g. nnx.List or python list)
        if isinstance(target_obj, (list, tuple)) or (hasattr(target_obj, "__iter__") and not isinstance(target_obj, (str, bytes, nnx.Module))):
            try:
                for idx, item in enumerate(target_obj):
                    if isinstance(item, nnx.Module):
                        if hasattr(item, "kernel") and not isinstance(item, LoRALinear) and _matches_target(f"layer_{idx}", target_layers, item):
                            lora_layer = LoRALinear(
                                item,
                                rank=rank,
                                alpha=alpha,
                                dropout_rate=dropout_rate,
                                rngs=rngs,
                            )
                            target_obj[idx] = lora_layer
                            injected_layers.append(lora_layer)
                        else:
                            _walk_and_inject(item)
            except (TypeError, IndexError):
                pass
            return

        # Handle child attributes of an NNX module
        if isinstance(target_obj, nnx.Module):
            for child_name in dir(target_obj):
                if child_name.startswith("_") or child_name in ("rngs", "scope"):
                    continue
                try:
                    child_val = getattr(target_obj, child_name)
                    if isinstance(child_val, (nnx.Module, list, tuple)):
                        _walk_and_inject(target_obj, child_name)
                except Exception:
                    continue

    _walk_and_inject(model)
    return injected_layers


def eject_lora(model: nnx.Module) -> int:
    """Removes all injected LoRALinear wrappers, restoring original base layers in-place.

    Returns:
        Number of restored layers.
    """
    count = 0

    def _walk_and_eject(target_obj: Any):
        nonlocal count
        if isinstance(target_obj, (list, tuple)) or (hasattr(target_obj, "__iter__") and not isinstance(target_obj, (str, bytes, nnx.Module))):
            try:
                for idx, item in enumerate(target_obj):
                    if isinstance(item, LoRALinear):
                        target_obj[idx] = item.base_linear
                        count += 1
                        _walk_and_eject(item.base_linear)
                    elif isinstance(item, nnx.Module):
                        _walk_and_eject(item)
            except (TypeError, IndexError):
                pass
            return

        if isinstance(target_obj, nnx.Module):
            for child_name in dir(target_obj):
                if child_name.startswith("_") or child_name in ("rngs", "scope"):
                    continue
                try:
                    child_val = getattr(target_obj, child_name)
                    if isinstance(child_val, LoRALinear):
                        setattr(target_obj, child_name, child_val.base_linear)
                        count += 1
                        _walk_and_eject(child_val.base_linear)
                    elif isinstance(child_val, (nnx.Module, list, tuple)):
                        _walk_and_eject(child_val)
                except Exception:
                    continue

    _walk_and_eject(model)
    return count


def merge_lora(model: nnx.Module) -> int:
    """Recursively merges adapter weights into base linear kernels across the model."""
    count = 0
    for _, module in nnx.iter_modules(model):
        if isinstance(module, LoRALinear):
            module.merge()
            count += 1
    return count


def unmerge_lora(model: nnx.Module) -> int:
    """Recursively unmerges adapter weights from base linear kernels across the model."""
    count = 0
    for _, module in nnx.iter_modules(model):
        if isinstance(module, LoRALinear):
            module.unmerge()
            count += 1
    return count


def get_lora_summary(model: nnx.Module) -> Dict[str, Any]:
    """Returns parameter statistics comparing base model and adapter weights."""
    all_param_state = nnx.state(model, nnx.Param)
    total_params = sum(
        leaf.size for leaf in jax.tree_util.tree_leaves(all_param_state) if hasattr(leaf, "size")
    )

    lora_param_state = nnx.state(model, LoRAParam)
    lora_params = sum(
        leaf.size for leaf in jax.tree_util.tree_leaves(lora_param_state) if hasattr(leaf, "size")
    )

    base_params = total_params - lora_params
    trainable_pct = (lora_params / total_params * 100.0) if total_params > 0 else 0.0

    return {
        "total_parameters": total_params,
        "base_parameters": base_params,
        "trainable_lora_parameters": lora_params,
        "trainable_percentage": round(trainable_pct, 4),
    }
