"""Flax LoRA: Lightweight, Native Low-Rank Adaptation for Flax NNX."""

from flax_lora.params import LoRAParam, is_lora_param
from flax_lora.layers import LoRALinear
from flax_lora.injector import (
    inject_lora,
    eject_lora,
    merge_lora,
    unmerge_lora,
    get_lora_summary,
)
from flax_lora.optimizer import create_lora_optimizer, lora_value_and_grad
from flax_lora.checkpoint import save_adapter, load_adapter

# GPT-2 utilities require optional dependencies (transformers, torch).
# Import explicitly: from flax_lora.gpt2 import load_pretrained_gpt2, generate_text


def __getattr__(name):
    """Lazy import for optional GPT-2 module."""
    _gpt2_names = {"GPT2Model", "load_pretrained_gpt2", "generate_text"}
    if name in _gpt2_names:
        try:
            from flax_lora import gpt2 as _gpt2
        except ImportError as e:
            raise ImportError(
                f"'{name}' requires optional dependencies. "
                "Install them with: pip install flax-lora[examples]"
            ) from e
        return getattr(_gpt2, name)
    raise AttributeError(f"module 'flax_lora' has no attribute {name!r}")


__version__ = "0.1.0"
__all__ = [
    "LoRAParam",
    "is_lora_param",
    "LoRALinear",
    "inject_lora",
    "eject_lora",
    "merge_lora",
    "unmerge_lora",
    "get_lora_summary",
    "create_lora_optimizer",
    "lora_value_and_grad",
    "save_adapter",
    "load_adapter",
    "GPT2Model",
    "load_pretrained_gpt2",
    "generate_text",
]
