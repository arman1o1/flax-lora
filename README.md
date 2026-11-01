# Flax LoRA (`flax_lora`)

> **Native, modular Low-Rank Adaptation (LoRA) for Flax NNX.**

`flax_lora` provides lightweight Parameter-Efficient Fine-Tuning (PEFT) built specifically for the stateful Flax NNX (`flax.nnx`) API. It enables adapting transformer and neural network layers with $<0.25\%$ trainable parameters, provides in-place weight merging for zero runtime deployment overhead, and saves compact (<1.1 MB) adapter checkpoints via Orbax.

---

## Key Features

- **Native Flax NNX Integration**: Leverages custom `LoRAParam(nnx.Param)` types, enabling type-based parameter isolation via `nnx.state(model, LoRAParam)` and strict derivative filtering with `nnx.DiffState(0, LoRAParam)`.
- **Seamless Layer Injection & Ejection**: Adapt any existing NNX model tree in-place with `inject_lora(...)` targeting attention projections (`query`, `value`, `q_proj`, `v_proj`, etc.) without rewriting model code. Restore base models cleanly with `eject_lora(...)`.
- **Linear & LinearGeneral Support**: Fully compatible with both standard 2D `nnx.Linear` layers and multi-dimensional `nnx.LinearGeneral` layers (as found in `nnx.MultiHeadAttention`).
- **Guaranteed Zero-Init Equivalence**: Low-rank matrix $B$ is initialized to exact zeros, guaranteeing that at step 0 the model outputs are bit-for-bit identical to the unadapted base model.
- **Zero-Overhead Merged Inference**: In-place `.merge()` and `.unmerge()` fold $\Delta W = \frac{\alpha}{r} B A$ directly into the base weights, eliminating runtime dispatch overhead during production serving.
- **Lightweight Orbax Persistence**: `save_adapter(...)` and `load_adapter(...)` serialize only adapter tensors and configuration metadata (<1.1 MB for GPT-2, <25 KB for toy architectures), keeping checkpoints portable and fast to transfer.
- **Tested on Pretrained GPT-2 (124M)**: Includes native HuggingFace weight porting into NNX and verified domain adaptation on TinyStories.

---

## Architecture

$$\Delta W = \frac{\alpha}{r} B A, \quad W_{\text{eff}} = W_0 + \Delta W$$

```
                      Input x
                         │
           ┌─────────────┴─────────────┐
           │                           │
           ▼                           ▼
    Base Linear W_0            LoRA Branch:
(Frozen: nnx.Param)         x @ A @ B * (alpha / r)
           │              (Trainable: LoRAParam)
           │                           │
           └─────────────┬─────────────┘
                         │ (+)
                         ▼
                   Output y
```

During training, base weights $W_0$ remain 100% frozen. During deployment, calling `merge_lora(model)` adds $\Delta W$ directly to $W_0$ in-place so inference runs with zero adapter latency and zero memory overhead.

---

## Installation

Install directly in editable mode:
```bash
git clone https://github.com/arman1o1/flax-lora.git
cd flax-lora
pip install -e .
```

To run the examples and full test suite:
```bash
pip install -e ".[examples,dev]"
```

---

## Quickstart

```python
import flax.nnx as nnx
import optax
from flax_lora import (
    inject_lora,
    merge_lora,
    create_lora_optimizer,
    lora_value_and_grad,
    save_adapter,
    load_adapter,
    get_lora_summary,
)

# 1. Instantiate any Flax NNX model
rngs = nnx.Rngs(0)
model = MyTransformer(rngs=rngs)

# 2. Inject LoRA adapters into attention projections
inject_lora(model, target_layers=("query", "value"), rank=4, alpha=8.0, rngs=rngs)

# 3. View parameter efficiency
summary = get_lora_summary(model)
print(f"Trainable: {summary['trainable_lora_parameters']:,} / {summary['total_parameters']:,} ({summary['trainable_percentage']}%)")

# 4. Create optimizer (freezes base model, trains only LoRAParam)
optimizer = create_lora_optimizer(model, tx=optax.adamw(learning_rate=1e-3))

# 5. JIT-compiled train step
@nnx.jit
def train_step(m, opt, batch, targets):
    def loss_fn(model, x, y):
        logits = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    grad_fn = lora_value_and_grad(loss_fn)
    loss, grads = grad_fn(m, batch, targets)
    opt.update(m, grads)
    return loss

# 6. Save adapter-only checkpoint (<1.1 MB)
save_adapter("checkpoints/my_adapter", model)

# 7. Merge weights for zero-overhead inference
merge_lora(model)
```

---

## Benchmarks & Examples

### 1. Pretrained GPT-2 (124M) Fine-Tuning (`examples/finetune_gpt2.py`)

Adapts an official pretrained GPT-2 model (12-layer, 768-dim, 12-head) on children's stories (TinyStories):

```bash
python examples/finetune_gpt2.py
```

#### Results & Parameter Efficiency

| Metric | Full Fine-Tuning | Flax NNX LoRA (`r=8, alpha=16`) | Savings |
| :--- | :--- | :--- | :--- |
| **Total Parameters** | 124,734,720 | 124,734,720 | — |
| **Trainable Parameters** | 124,734,720 (100%) | **294,912** (**0.236%**) | **99.76% reduction** |
| **Base Model Frozen** | 0% | **100% verified** | Bit-for-bit unchanged |
| **Training Loss** | — | **3.73 $\to$ 3.10** | Continuous descent |
| **Checkpoint Size** | ~498 MB | **1.1 MB** (Orbax adapter) | **99.78% smaller** |
| **Inference Overhead** | 0 ms | **0 ms** (via `merge_lora`) | Zero latency penalty |

#### Generation Showcase

**Baseline GPT-2 Prompt:**
> *Prompt:* `"Once upon a time in a peaceful meadow,"`  
> *Baseline GPT-2:* `"...there was a village where there lived a family who had lived there for hundreds..."`

**LoRA-Adapted Prompt (TinyStories Domain):**
> *Prompt:* `"There was a little puppy named Max who loved to"`  
> *LoRA-Adapted GPT-2:* `"...play the game, and he had a good time with it. Max would..."`

---

### 2. Lightweight MiniGPT Example (`examples/finetune_stories.py`)

A fully self-contained demonstration combining Flax NNX, Google Grain data pipelining, and LoRA injection on custom architectures:

```bash
python examples/finetune_stories.py
```

- **0.03% trainable parameters** (2,048 out of 6,504,064)
- **24.1 KB adapter checkpoint**
- **Exact output equivalence verification between unmerged and merged inference**

---

## API Reference

### `inject_lora`
```python
inject_lora(
    model: nnx.Module,
    target_layers: Sequence[str] = ("query", "value", "q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout_rate: float = 0.0,
    rngs: Optional[nnx.Rngs] = None,
) -> List[LoRALinear]
```
Recursively traverses `model` and replaces matching linear layers with `LoRALinear` wrappers in-place.

### `eject_lora`
```python
eject_lora(model: nnx.Module) -> int
```
Removes all `LoRALinear` layers from `model`, restoring the original base linear layers in-place.

### `merge_lora` / `unmerge_lora`
```python
merge_lora(model: nnx.Module) -> int
unmerge_lora(model: nnx.Module) -> int
```
Recursively folds or subtracts low-rank adapter deltas directly into base kernels across the entire model tree.

### `create_lora_optimizer`
```python
create_lora_optimizer(model: nnx.Module, tx: Optional[optax.GradientTransformation] = None) -> nnx.Optimizer
```
Instantiates an `nnx.Optimizer` bounded with `wrt=LoRAParam`, automatically freezing all base parameters.

### `lora_value_and_grad`
```python
lora_value_and_grad(loss_fn: Callable, has_aux: bool = False) -> Callable
```
Differentiates `loss_fn` strictly with respect to `LoRAParam` variables via `nnx.DiffState(0, LoRAParam)`.

### `save_adapter` / `load_adapter`
```python
save_adapter(save_path: str | Path, model: nnx.Module, metadata: Optional[dict] = None) -> Path
load_adapter(load_path: str | Path, model: nnx.Module, sharding: Optional[jax.sharding.Sharding] = None) -> dict
```
Serializes and restores adapter PyTrees using Orbax, omitting base model weights.

### `get_lora_summary`
```python
get_lora_summary(model: nnx.Module) -> dict
```
Returns a parameter census: total parameters, base parameters, trainable LoRA parameters, and percentage.

---

## Test Suite

All tests can be executed with `pytest`:

```bash
pytest tests/ -v
```

```
tests/test_checkpoint.py::test_save_and_load_adapter_roundtrip PASSED
tests/test_checkpoint.py::test_adapter_checkpoint_size PASSED
tests/test_injector.py::test_inject_lora_targets PASSED
tests/test_injector.py::test_eject_lora PASSED
tests/test_injector.py::test_merge_and_unmerge_tree PASSED
tests/test_injector.py::test_optimizer_freezes_base_weights PASSED
tests/test_layers.py::test_zero_initialization PASSED
tests/test_layers.py::test_forward_computation PASSED
tests/test_layers.py::test_merge_and_unmerge_linear PASSED
tests/test_layers.py::test_linear_general_mha PASSED
tests/test_layers.py::test_invalid_parameters PASSED
============================== 11 passed ==============================
```

---

## License

This project is licensed under the [MIT License](LICENSE).
