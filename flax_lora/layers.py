"""Low-Rank Adaptation (LoRA) layer implementation for Flax NNX."""

import math
from typing import Optional, Union, Tuple, Sequence
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from flax_lora.params import LoRAParam


class LoRALinear(nnx.Module):
    """Wraps a linear module with low-rank adapter matrices (A and B).

    Adapts the weight matrix W of a linear transformation using:
        W_eff = W_base + (alpha / r) * (B @ A)
    where:
        - A in R^(in_features x rank), initialized with He uniform random values
        - B in R^(rank x out_features), initialized to exact zeros
    Guaranteeing that at initialization, Delta W = 0 and model output is unchanged.

    Supports both standard `nnx.Linear` (2D kernel) and `nnx.LinearGeneral`
    (multi-dimensional query/key/value/out attention kernels).
    """

    def __init__(
        self,
        base_linear: nnx.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {rank}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        self.base_linear = base_linear
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)
        self.dropout_rate = float(dropout_rate)
        self.is_merged = False

        if not hasattr(base_linear, "kernel"):
            raise AttributeError("base_linear module must have a 'kernel' attribute.")

        # Inspect kernel and feature dimensions
        k_shape = base_linear.kernel.shape

        in_f = (
            base_linear.in_features
            if hasattr(base_linear, "in_features")
            else k_shape[0]
        )
        out_f = (
            base_linear.out_features
            if hasattr(base_linear, "out_features")
            else k_shape[1]
        )

        self.in_tuple: Tuple[int, ...] = in_f if isinstance(in_f, tuple) else (in_f,)
        self.out_tuple: Tuple[int, ...] = out_f if isinstance(out_f, tuple) else (out_f,)

        self.in_dim = math.prod(self.in_tuple)
        self.out_dim = math.prod(self.out_tuple)

        # Initialize A with He normal / uniform scaled by 1/sqrt(in_dim)
        std = 1.0 / math.sqrt(self.in_dim)
        a_init = jax.random.normal(rngs.params(), (self.in_dim, self.rank)) * std
        # Initialize B to exact zeros
        b_init = jnp.zeros((self.rank, self.out_dim), dtype=base_linear.kernel.dtype)

        self.lora_a = LoRAParam(a_init.astype(base_linear.kernel.dtype))
        self.lora_b = LoRAParam(b_init)

    def _compute_delta_w(self) -> jax.Array:
        """Compute the low-rank delta weight matrix reshaped to base kernel shape."""
        delta = (self.lora_a[...] @ self.lora_b[...]) * self.scale
        return delta.reshape(self.base_linear.kernel.shape).astype(
            self.base_linear.kernel.dtype
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jax.Array:
        base_out = self.base_linear(x)

        if self.is_merged:
            return base_out

        in_axes_count = len(self.in_tuple)
        batch_prefix = x.shape[:-in_axes_count]
        x_flat = x.reshape(*batch_prefix, self.in_dim)

        lora_h = x_flat @ self.lora_a[...]

        if self.dropout_rate > 0.0 and not deterministic:
            if rngs is None:
                raise ValueError("rngs must be provided for non-deterministic dropout.")
            keep_prob = 1.0 - self.dropout_rate
            mask = jax.random.bernoulli(rngs.dropout(), keep_prob, lora_h.shape)
            lora_h = jnp.where(mask, lora_h / keep_prob, 0.0)

        lora_flat = lora_h @ self.lora_b[...]
        lora_out = lora_flat.reshape(*batch_prefix, *self.out_tuple)

        return base_out + (lora_out * self.scale)

    def merge(self) -> None:
        """Folds adapter delta weights directly into base linear kernel in-place."""
        if self.is_merged:
            return
        delta = self._compute_delta_w()
        self.base_linear.kernel[...] = self.base_linear.kernel[...] + delta
        self.is_merged = True

    def unmerge(self) -> None:
        """Subtracts adapter delta weights from base linear kernel in-place."""
        if not self.is_merged:
            return
        delta = self._compute_delta_w()
        self.base_linear.kernel[...] = self.base_linear.kernel[...] - delta
        self.is_merged = False
