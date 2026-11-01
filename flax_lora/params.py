"""Parameter definitions and filters for Flax NNX LoRA."""

import flax.nnx as nnx


class LoRAParam(nnx.Param):
    """Specialized parameter variable type marking low-rank adapter weights.

    Inherits from nnx.Param so standard Flax NNX operations recognize it as a
    valid model parameter, while allowing precise type-based state filtering
    (e.g., via `nnx.state(model, LoRAParam)` or `nnx.split(model, LoRAParam, ...)`).
    """
    pass


def is_lora_param(val) -> bool:
    """Return True if the object is an instance of LoRAParam or a LoRAParam variable."""
    return isinstance(val, LoRAParam)
