# Training with `axim sft`

Run the fine-tuner:

```bash
axim sft --config configs/sft_example.json \
    --base-model base.axim --train train.jsonl --val val.jsonl \
    --output tuned.axim --device cuda --dtype bf16
```

Every knob can also be passed directly as a CLI flag; see `axim sft --help`.

## Config resolution

For each option the trainer uses the first available value in this order:

1. **CLI flag** — e.g. `--batch-size 8`.
2. **Config file** — e.g. `"batch_size": 8` in the JSON config.
3. **Hardcoded default** — defined in `axim/sft/train.py`.

## Base model loading

The trainer loads `base_model` with `_load_base`, which supports:

- `.axim` files — via `axim.core.load_axim`.
- A directory with `model.safetensors` + `config.json` + `tokenizer.pkl`.

`max_seq_len` defaults to the base model's `max_position_embeddings` and is
clamped to that value if a larger value is requested.

## Data packing

The rendered conversations are packed into fixed-length rows by `PackedBatcher`:

- `"bestfit"` (default) — greedily bin-packs conversations into each row and pads
  the remainder with BOS tokens. Mirrors nanoChat's SFT packing and wastes no tokens.
- `"single"` — one conversation per row, truncated and padded. Simpler but less
  efficient.

Targets are int64 tensors with `-1` at padded/unsupervised positions so they are
ignored by cross-entropy. Inputs are targets shifted right with a BOS prefix.

## Assistant-only masking

When `train_on="assistant"` (default), only assistant content tokens are
supervised. Special scaffold tokens (`<|user_start|>`, `<|assistant_start|>`,
etc.) and user content are masked to `-1`. Use `train_on="all"` to supervise
every content token.

## Optimizer

### MuonAdamW (default)

```bash
axim sft --optimizer muon \
    --matrix-lr 0.02 --embedding-lr 0.3 --unembedding-lr 0.004 --scalar-lr 0.5
```

Muon uses four parameter groups:

- `matrix` — transformer block matrices.
- `embedding` — token embeddings.
- `unembedding` — `lm_head`.
- `scalar` — per-layer scalars.

During the first 300 steps Muon momentum ramps from `0.85` to `0.95`.

### AdamW

```bash
axim sft --optimizer adamw --lr 3e-4 --emb-lr-mult 1.0
```

AdamW uses two parameter groups: embeddings at `lr * emb_lr_mult` and everything
else at `lr`. Weight decay applies to all trainable parameters.

If Muon fails to initialize (e.g. on CPU/MPS), the trainer automatically falls
back to AdamW.

## Learning-rate schedule (WSD)

The default schedule is warmup-stable-decay:

- `init_lr_frac` (default `0.5`) — LR at step 0 as a fraction of the base LR.
- `warmup_ratio` (default `0.03`) — fraction of training spent linearly warming up
  from `init_lr_frac` to `1.0`.
- `warmdown_ratio` (default `0.2`) — fraction of training spent linearly decaying
  to `final_lr_frac`.
- `final_lr_frac` (default `0.0`) — LR floor at the end of training.

`--lr` is used by AdamW; Muon always uses the four group LRs (`--matrix-lr`, `--embedding-lr`, `--unembedding-lr`, `--scalar-lr`).

## Batch sizing

```text
total_batch = batch_size * max_seq_len * grad_accum * world_size
```

- `batch_size` — micro-batch rows per forward pass.
- `grad_accum` — gradient accumulation steps.
- Multi-GPU training is supported via `torchrun`, which scales the total batch by
  `world_size`.

## In-training sampling

The trainer can generate sample completions during training to watch quality
improve:

- `sample_every` (default `500`) — generate samples every N steps.
- `sample_prompts` (default `3`) — number of validation prompts to sample.
- `sample_tokens` (default `64`) — max tokens per sample.
- `sample_static_file` — JSONL of fixed prompts; the same prompts are used every
  sample so you can compare outputs across checkpoints.
- `sample_random` — number of random validation prompts to sample each time
  (re-seeded by step for variety).

## Validation

If `--val` is not provided, a deterministic slice of the training data is held
out using `val_frac` (default `0.02`). Evaluation runs every `eval_every` steps
(default `200`) over `eval_steps` batches (default `20`). A baseline validation
loss is computed before training starts.

## Checkpoints and saving

- `save_every` — write a numbered full `.axim` snapshot (`step{N}.axim`) every N
  steps. Heavy because each is a full model copy.
- `checkpoint_every` — write a resumable `checkpoint.axim` + `checkpoint.pt`
  (optimizer state) every N steps. Required for `--resume-from` to restore
  optimizer state.
- `save_optim` — persist optimizer state alongside the final model.
- `resume_from` — resume weights (and optimizer state if available) from a `.axim`
  or a checkpoint directory.

The default output directory is next to `--output`, e.g. `out/sft_checkpoints`.

## Final merged `.axim`

At the end of training the trainer writes the full merged model to `--output`.
The output file contains:

- updated weights,
- the original config plus `training_step`, `val_loss`, and `finetuned=True`,
- the tokenizer,
- metadata including SFT provenance and the resolved training config.

The output is a single, ready-to-serve `.axim` file.
