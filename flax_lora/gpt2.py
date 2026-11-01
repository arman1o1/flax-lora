"""Flax NNX GPT-2 model with pretrained weight loader from HuggingFace."""

from typing import Optional, List
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2Block(nnx.Module):
    """Transformer decoder block matching HuggingFace GPT-2 architecture."""

    def __init__(self, d: int = 768, num_heads: int = 12, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = d // num_heads

        self.ln_1 = nnx.LayerNorm(d, epsilon=1e-5, rngs=rngs)
        self.q_proj = nnx.Linear(d, d, rngs=rngs)
        self.k_proj = nnx.Linear(d, d, rngs=rngs)
        self.v_proj = nnx.Linear(d, d, rngs=rngs)
        self.out_proj = nnx.Linear(d, d, rngs=rngs)

        self.ln_2 = nnx.LayerNorm(d, epsilon=1e-5, rngs=rngs)
        self.mlp_fc = nnx.Linear(d, d * 4, rngs=rngs)
        self.mlp_proj = nnx.Linear(d * 4, d, rngs=rngs)

    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        # Pre-LN Self-Attention
        h = self.ln_1(x)
        b, l, d = h.shape

        q = self.q_proj(h).reshape(b, l, self.num_heads, self.head_dim).swapaxes(1, 2)
        k = self.k_proj(h).reshape(b, l, self.num_heads, self.head_dim).swapaxes(1, 2)
        v = self.v_proj(h).reshape(b, l, self.num_heads, self.head_dim).swapaxes(1, 2)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = (q @ k.swapaxes(-1, -2)) * scale
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)

        attn = jax.nn.softmax(scores, axis=-1)
        out = (attn @ v).swapaxes(1, 2).reshape(b, l, d)
        x = x + self.out_proj(out)

        # Pre-LN MLP
        h2 = self.ln_2(x)
        mlp = self.mlp_proj(jax.nn.gelu(self.mlp_fc(h2), approximate=True))
        x = x + mlp
        return x


class GPT2Model(nnx.Module):
    """Flax NNX 124M GPT-2 Causal Language Model."""

    def __init__(
        self,
        vocab_size: int = 50257,
        maxlen: int = 1024,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_blocks: int = 12,
        *,
        rngs: nnx.Rngs,
    ):
        self.maxlen = maxlen
        self.wte = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.wpe = nnx.Embed(maxlen, embed_dim, rngs=rngs)
        self.blocks = nnx.List([
            GPT2Block(d=embed_dim, num_heads=num_heads, rngs=rngs)
            for _ in range(num_blocks)
        ])
        self.ln_f = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)

    def __call__(self, token_ids: jax.Array) -> jax.Array:
        b, l = token_ids.shape
        positions = jnp.arange(l)[None, :]
        x = self.wte(token_ids) + self.wpe(positions)
        mask = jnp.tril(jnp.ones((1, 1, l, l), dtype=bool))

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        # Weight tying with token embeddings
        logits = x @ self.wte.embedding[...].T
        return logits


def load_pretrained_gpt2(
    model_name: str = "gpt2",
    *,
    rngs: Optional[nnx.Rngs] = None,
) -> GPT2Model:
    """Instantiates GPT2Model in Flax NNX and populates it with official pretrained weights.

    Args:
        model_name: HuggingFace model identifier (e.g. 'gpt2').
        rngs: Optional NNX Rngs instance.

    Returns:
        Hydrated GPT2Model ready for inference or LoRA fine-tuning.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)

    model = GPT2Model(rngs=rngs)
    pt_model = GPT2LMHeadModel.from_pretrained(model_name)
    sd = pt_model.state_dict()

    # Load Embeddings
    model.wte.embedding[...] = jnp.array(sd["transformer.wte.weight"].numpy())
    model.wpe.embedding[...] = jnp.array(sd["transformer.wpe.weight"].numpy())
    model.ln_f.scale[...] = jnp.array(sd["transformer.ln_f.weight"].numpy())
    model.ln_f.bias[...] = jnp.array(sd["transformer.ln_f.bias"].numpy())

    # Load Blocks
    for i, b in enumerate(model.blocks):
        prefix = f"transformer.h.{i}."
        b.ln_1.scale[...] = jnp.array(sd[prefix + "ln_1.weight"].numpy())
        b.ln_1.bias[...] = jnp.array(sd[prefix + "ln_1.bias"].numpy())

        c_attn_w = sd[prefix + "attn.c_attn.weight"].numpy()
        c_attn_b = sd[prefix + "attn.c_attn.bias"].numpy()

        b.q_proj.kernel[...] = jnp.array(c_attn_w[:, :768])
        b.q_proj.bias[...] = jnp.array(c_attn_b[:768])
        b.k_proj.kernel[...] = jnp.array(c_attn_w[:, 768:1536])
        b.k_proj.bias[...] = jnp.array(c_attn_b[768:1536])
        b.v_proj.kernel[...] = jnp.array(c_attn_w[:, 1536:])
        b.v_proj.bias[...] = jnp.array(c_attn_b[1536:])

        b.out_proj.kernel[...] = jnp.array(sd[prefix + "attn.c_proj.weight"].numpy())
        b.out_proj.bias[...] = jnp.array(sd[prefix + "attn.c_proj.bias"].numpy())

        b.ln_2.scale[...] = jnp.array(sd[prefix + "ln_2.weight"].numpy())
        b.ln_2.bias[...] = jnp.array(sd[prefix + "ln_2.bias"].numpy())

        b.mlp_fc.kernel[...] = jnp.array(sd[prefix + "mlp.c_fc.weight"].numpy())
        b.mlp_fc.bias[...] = jnp.array(sd[prefix + "mlp.c_fc.bias"].numpy())
        b.mlp_proj.kernel[...] = jnp.array(sd[prefix + "mlp.c_proj.weight"].numpy())
        b.mlp_proj.bias[...] = jnp.array(sd[prefix + "mlp.c_proj.bias"].numpy())

    return model


@nnx.jit
def _predict_next_logits(model: GPT2Model, input_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    return logits[0, -1, :]


def generate_text(
    model: GPT2Model,
    prompt: str,
    tokenizer: Optional[GPT2Tokenizer] = None,
    max_new_tokens: int = 20,
    temperature: float = 0.7,
    top_k: int = 40,
) -> str:
    """Generates autoregressive text using JIT-compiled next-token prediction."""
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    tokens: List[int] = tokenizer.encode(prompt)
    key = jax.random.PRNGKey(42)

    for _ in range(max_new_tokens):
        ctx = tokens[-model.maxlen :]
        input_ids = jnp.array([ctx], dtype=jnp.int32)
        next_logits = _predict_next_logits(model, input_ids) / max(temperature, 1e-4)

        if top_k > 0:
            top_vals, top_indices = jax.lax.top_k(next_logits, top_k)
            key, subkey = jax.random.split(key)
            choice_idx = int(jax.random.categorical(subkey, top_vals))
            next_token = int(top_indices[choice_idx])
        else:
            key, subkey = jax.random.split(key)
            next_token = int(jax.random.categorical(subkey, next_logits))

        tokens.append(next_token)

    return tokenizer.decode(tokens)
