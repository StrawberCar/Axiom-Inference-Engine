# SFT config reference

This table describes every field used by `axim sft`. Defaults are the hardcoded
fallback values in `axim/sft/train.py` when the field is not set by CLI or
config.

## I/O

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_model` | str | `model.axim` | Path to the base `.axim` or a directory with safetensors/config/tokenizer. |
| `train` | str / null | `null` | Training JSONL path. Required unless resuming. |
| `val` | str / null | `null` | Validation JSONL path. If null, a slice of `train` is held out. |
| `val_frac` | float | `0.02` | Fraction of training data to hold out when `val` is not given. |
| `output` | str | `finetuned.axim` | Final merged `.axim` output path. |
| `out_dir` | str / null | `null` | Checkpoint directory. Defaults to a `sft_checkpoints` folder next to `output`. |
| `resume_from` | str / null | `null` | Resume from a `.axim` or checkpoint directory. |

## Device and precision

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device` | str | `""` | Device: `cuda`, `cpu`, `mps`. Empty string auto-detects. |
| `dtype` | str | `""` | Compute dtype: `bfloat16`, `float16`, `float32`. Empty string auto-detects via nanoChat. |
| `compile` | int / bool | `0` | Whether to `torch.compile` the model. Off by default for portability. |
| `cast_dtype` | int / bool | `0` | Cast params/buffers to compute dtype after loading. Useful for GPUs that cannot run the stored dtype, e.g. bf16 weights on a T4. |

## Data

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_seq_len` | int / null | `null` | Training context length. Defaults to the base model's `max_position_embeddings` and is clamped to it. |
| `train_on` | str | `"assistant"` | Tokens to supervise: `"assistant"` or `"all"`. |
| `pack` | str | `"bestfit"` | Packing mode: `"bestfit"` or `"single"`. |
| `system_prompt` | str / null | `null` | System prompt prepended to records that lack one. |
| `seed` | int | `42` | RNG seed for shuffling and splits. |

## Training

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | int | `4` | Micro-batch rows per forward pass. |
| `grad_accum` | int | `8` | Gradient accumulation steps. |
| `epochs` | float | `3` | Training epochs. `-1` disables and uses `steps`. |
| `steps` | int | `-1` | Max optimizer steps. `-1` means derive from `epochs`. |
| `max_grad_norm` | float | `1.0` | Gradient clipping max norm. `0` disables. |

## Optimizer

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `optimizer` | str | `"muon"` | Optimizer: `"muon"` or `"adamw"`. |
| `lr` | float / null | `null` | Base LR for AdamW. Muon ignores this and uses the group LRs below. |
| `matrix_lr` | float | `0.02` | Muon LR for transformer matrices. |
| `embedding_lr` | float | `0.3` | Muon LR for token embeddings. |
| `unembedding_lr` | float | `0.004` | Muon LR for the output lm_head. |
| `scalar_lr` | float | `0.5` | Muon LR for per-layer scalars. |
| `weight_decay` | float | `0.0` | Weight decay. Applied to matrices for Muon; all params for AdamW. |
| `emb_lr_mult` | float | `1.0` | AdamW embedding LR multiplier relative to `--lr`. |

## LR schedule

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `init_lr_frac` | float | `0.5` | LR at step 0, as a fraction of the base LR. |
| `warmup_ratio` | float | `0.03` | Fraction of training devoted to linear warmup. |
| `warmdown_ratio` | float | `0.2` | Fraction of training devoted to linear warmdown. |
| `final_lr_frac` | float | `0.0` | Final LR as a fraction of base LR; the warmdown floor. |

## Logging and checkpointing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `eval_every` | int | `200` | Run validation every N steps. `-1` disables. |
| `eval_steps` | int | `20` | Number of validation batches per evaluation. |
| `sample_every` | int | `500` | Generate sample completions every N steps. `-1` disables. |
| `sample_prompts` | int | `3` | Number of validation prompts to sample at each `sample_every`. |
| `sample_tokens` | int | `64` | Max tokens per sample generation. |
| `sample_static_file` | str / null | `null` | JSONL of fixed prompts sampled every `sample_every`. |
| `sample_random` | int / null | `null` | Number of random validation prompts to sample each time. |
| `save_every` | int | `-1` | Write a numbered `.axim` snapshot every N steps. `-1` disables. |
| `checkpoint_every` | int | `-1` | Write resumable `checkpoint.axim` + `checkpoint.pt` every N steps. `-1` disables. |
| `save_optim` | int / bool | `0` | Persist optimizer state with the final model. |
| `wandb` | str | `"dummy"` | WandB run name. `"dummy"` or empty disables logging. |
| `log_every` | int | `1` | Console log interval in steps. |
