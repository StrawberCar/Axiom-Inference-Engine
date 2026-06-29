# axim sft

Fine-tune an `.axim` model on supervised chat data.

`axim sft` does **not** use the shared subparser. It short-circuits to the SFT trainer's own `argparse`, so the full flag set and defaults are shown with:

```bash
axim sft --help
```

## Usage

```bash
axim sft [--config <path>] [flags...]
```

## Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `None` | JSON or YAML config file; CLI flags override it |
| `--base-model` | `model.axim` | Base `.axim` to fine-tune, or a directory with `model.safetensors` + `config.json` |
| `--train` | `None` | Training JSONL path |
| `--val` | `None` | Validation JSONL path (optional; otherwise a slice of `--train` is held out) |
| `--output` | `finetuned.axim` | Output `.axim` path |
| `--device` | autodetect | `cuda`, `cpu`, `mps`, or empty for auto |
| `--dtype` | autodetect | `bfloat16`, `float16`, `float32`, or empty for auto |
| `--epochs` | `3` | Number of passes over the training data |
| `--batch-size` | `4` | Micro-batch size |
| `--grad-accum` | `8` | Gradient accumulation steps |
| `--optimizer` | `muon` | `muon` (MuonAdamW) or `adamw` |
| `--lr` | `None` | Base learning rate |
| `--max-grad-norm` | `1.0` | Gradient clipping max norm; `0` disables |

!!! note "Full flag list"
    The trainer exposes many more knobs for learning-rate schedules, sampling, checkpointing, logging, and distributed training. See `axim sft --help` for the complete list.

## Quick 3-step recipe

1. Download a base model:

    ```bash
    axim download --repo my-org/my-model --output model.axim
    ```

2. Prepare SFT data:

    ```bash
    axim prepare-data \
        --repo my-org/chat-dataset \
        --output-train data/train.jsonl \
        --output-val data/val.jsonl
    ```

3. Fine-tune:

    ```bash
    axim sft --base-model model.axim \
        --train data/train.jsonl \
        --val data/val.jsonl \
        --output finetuned.axim \
        --device cuda --epochs 3
    ```

!!! tip "Resume training"
    Pass `--resume-from <checkpoint.axim>` or `--resume-from <out_dir>` to continue from a saved checkpoint. Resuming from a directory restores optimizer state when `--checkpoint-every` was used.
