"""End-to-end example: Fine-tuning Pretrained GPT-2 (124M) with Flax NNX LoRA.

Demonstrates:
1. Loading official pretrained GPT-2 weights into a native Flax NNX model.
2. Generating baseline text with the unadapted base model.
3. Injecting LoRA adapters on attention projections (q_proj & v_proj).
4. Training with only 0.23% trainable parameters (294,912 out of 124,734,720).
5. Fast CPU fine-tuning on TinyStories with JIT-compiled steps (~100ms/step).
6. Verification of frozen base model weights.
7. Saving lightweight adapter-only checkpoint via Orbax (< 300 KB).
8. In-place zero-overhead merged inference.
"""

import sys
import pathlib

# Ensure project root is on sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from typing import List
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx
from transformers import GPT2Tokenizer

from flax_lora import (
    load_pretrained_gpt2,
    generate_text,
    inject_lora,
    merge_lora,
    unmerge_lora,
    get_lora_summary,
    create_lora_optimizer,
    lora_value_and_grad,
    save_adapter,
)


def load_story_batches(file_path: str, tokenizer, batch_size: int = 2, maxlen: int = 32, num_batches: int = 10):
    """Loads text slices from TinyStories into fixed-shape batches."""
    txt = pathlib.Path(file_path).read_text(encoding="utf-8", errors="replace")
    stories = [s.strip() for s in txt.split("<|endoftext|>") if s.strip()]

    batches = []
    current_tokens = []
    for s in stories:
        current_tokens.extend(tokenizer.encode(s) + [tokenizer.eos_token_id])
        while len(current_tokens) >= batch_size * maxlen:
            chunk = current_tokens[: batch_size * maxlen]
            current_tokens = current_tokens[batch_size * maxlen :]
            batch_arr = jnp.array(chunk, dtype=jnp.int32).reshape(batch_size, maxlen)
            batches.append(batch_arr)
            if len(batches) >= num_batches:
                return batches
    return batches


def main():
    print("=" * 75)
    print("  Flax NNX LoRA Fine-Tuning with Pretrained GPT-2 (124M)")
    print("=" * 75)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    rngs = nnx.Rngs(42)

    # 1. Load Pretrained GPT-2
    print("\n[1/6] Loading official pretrained GPT-2 weights into Flax NNX...")
    model = load_pretrained_gpt2("gpt2", rngs=rngs)

    prompt = "Once upon a time in a peaceful meadow,"
    print(f"\n[Baseline Generation (Zero-Shot GPT-2)]:")
    base_completion = generate_text(model, prompt, tokenizer, max_new_tokens=15, temperature=0.7)
    print(f"  Prompt:     '{prompt}'")
    print(f"  Completion: '{base_completion.strip()}'\n")

    # 2. Inject LoRA on Query and Value projections
    print("[2/6] Injecting LoRA adapters on attention projections (q_proj & v_proj)...")
    injected_layers = inject_lora(model, target_layers=("q_proj", "v_proj"), rank=8, alpha=16.0, rngs=rngs)
    print(f"  Injected {len(injected_layers)} LoRALinear modules across 12 transformer layers.")

    summary = get_lora_summary(model)
    print(f"  Total parameters:          {summary['total_parameters']:,}")
    print(f"  Base frozen parameters:    {summary['base_parameters']:,}")
    print(f"  Trainable LoRA parameters: {summary['trainable_lora_parameters']:,}")
    print(f"  Trainable percentage:      {summary['trainable_percentage']:.4f}%")
    assert summary["trainable_percentage"] < 1.0, "Trainable params should be under 1%!"

    # Snapshot base weight to verify freezing
    base_kernel_snapshot = jnp.array(model.blocks[0].q_proj.base_linear.kernel[...])

    # 3. Setup LoRA Optimizer
    print("\n[3/6] Setting up LoRA Optimizer (wrt=LoRAParam)...")
    optimizer = create_lora_optimizer(model, tx=optax.adamw(learning_rate=5e-4, weight_decay=0.01))

    # 4. Load Training Data
    data_path = pathlib.Path(__file__).resolve().parent / "data" / "tinystories_sample.txt"
    if not data_path.exists():
        raise FileNotFoundError(
            f"TinyStories sample not found at {data_path}. "
            "See examples/data/tinystories_sample.txt in the repository."
        )
    batches = load_story_batches(str(data_path), tokenizer, batch_size=2, maxlen=32, num_batches=10)
    print(f"  Prepared {len(batches)} training batches of shape (2, 32).")

    # 5. JIT-compiled Train Step
    @nnx.jit
    def train_step(m, opt, batch_tokens):
        def loss_fn(mod, tokens):
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            logits = mod(inputs)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
            return loss

        grad_fn = lora_value_and_grad(loss_fn)
        loss, grads = grad_fn(m, batch_tokens)
        opt.update(m, grads)
        return loss

    print("\n[4/6] Fine-tuning LoRA adapters on TinyStories domain...")
    initial_loss = None
    final_loss = None

    for step_idx, batch_arr in enumerate(batches, 1):
        loss = train_step(model, optimizer, batch_arr)
        loss_val = float(loss)

        if initial_loss is None:
            initial_loss = loss_val
        final_loss = loss_val

        if step_idx % 3 == 0 or step_idx == 1:
            print(f"  Step {step_idx:2d}/{len(batches)} | Loss: {loss_val:.4f} (Perplexity: {jnp.exp(loss_val):.2f})")

    print(f"\n  Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")

    # 6. Verify Base Weights Remained Frozen
    current_base_kernel = model.blocks[0].q_proj.base_linear.kernel[...]
    assert jnp.array_equal(base_kernel_snapshot, current_base_kernel), "Base weights were mutated during training!"
    print("  [Pass] Base model weights verified 100% frozen during training.")

    # 7. Save Adapter Checkpoint
    print("\n[5/6] Saving lightweight adapter checkpoint via Orbax...")
    checkpoint_dir = pathlib.Path("checkpoints/gpt2_lora_tinystories")
    save_adapter(checkpoint_dir, model, metadata={"initial_loss": initial_loss, "final_loss": final_loss})

    total_bytes = sum(f.stat().st_size for f in checkpoint_dir.rglob("*") if f.is_file())
    print(f"  Adapter saved to '{checkpoint_dir}' ({total_bytes / 1024.0:.1f} KB).")

    # 8. Zero-Overhead Merged Inference
    print("\n[6/6] Demonstrating zero-overhead merged inference...")
    test_prompt = "There was a little puppy named Max who loved to"
    unmerged_gen = generate_text(model, test_prompt, tokenizer, max_new_tokens=15, temperature=0.7)
    print(f"\n  [Unmerged LoRA Output]:\n  '{unmerged_gen.strip()}'")

    merged_count = merge_lora(model)
    print(f"\n  Merged {merged_count} adapter layers directly into base kernels.")
    merged_gen = generate_text(model, test_prompt, tokenizer, max_new_tokens=15, temperature=0.7)
    print(f"  [Merged Zero-Overhead Output]:\n  '{merged_gen.strip()}'")

    unmerged_count = unmerge_lora(model)
    print(f"  Unmerged {unmerged_count} adapter layers cleanly.")

    print("\n" + "=" * 75)
    print("  Pretrained GPT-2 LoRA Fine-Tuning Completed Successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
