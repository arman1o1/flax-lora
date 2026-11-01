"""Optimizer helpers for Flax NNX LoRA training."""

from typing import Callable, Any, Optional
import optax
import flax.nnx as nnx

from flax_lora.params import LoRAParam


def create_lora_optimizer(
    model: nnx.Module,
    tx: Optional[optax.GradientTransformation] = None,
) -> nnx.Optimizer:
    """Creates a Flax NNX Optimizer configured to update ONLY LoRAParam variables.

    Base model parameters are automatically frozen. Gradients computed with respect
    to LoRAParam are passed directly to `tx` (defaults to AdamW with learning rate 1e-4).

    Args:
        model: The NNX model containing injected LoRALinear layers.
        tx: Optax gradient transformation (e.g. optax.adamw, optax.sgd).

    Returns:
        An nnx.Optimizer instance bounded to `wrt=LoRAParam`.
    """
    if tx is None:
        tx = optax.adamw(learning_rate=1e-4, weight_decay=0.01)

    return nnx.Optimizer(model, tx, wrt=LoRAParam)


def lora_value_and_grad(
    loss_fn: Callable[..., Any],
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Wraps nnx.value_and_grad to differentiate strictly with respect to LoRAParam variables.

    Args:
        loss_fn: Loss function with signature `loss_fn(model, *args, **kwargs)`.
        has_aux: Whether loss_fn returns auxiliary data alongside loss (e.g. (loss, metrics)).

    Returns:
        A callable returning (loss, grads) or ((loss, aux), grads) where grads
        contains only gradients for LoRAParam leaves.
    """
    return nnx.value_and_grad(
        loss_fn,
        argnums=nnx.DiffState(0, LoRAParam),
        has_aux=has_aux,
    )
