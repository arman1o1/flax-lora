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
from flax_lora.gpt2 import GPT2Model, load_pretrained_gpt2, generate_text

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
