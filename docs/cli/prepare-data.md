# axim prepare-data

Download a dataset from the Hugging Face Hub and prepare train/validation JSONL files for SFT.

## Usage

```bash
axim prepare-data --repo <repo> --output-train <path> --output-val <path> [flags]
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--repo` | required | Hugging Face dataset repository id |
| `--config` | `None` | Dataset config name to load |
| `--split` | `train` | Dataset split to use |
| `--train-examples` | `2000` | Number of examples to write to the training split |
| `--val-examples` | `200` | Number of examples to write to the validation split |
| `--output-train` | required | Output path for training JSONL |
| `--output-val` | required | Output path for validation JSONL |
| `--seed` | `42` | Random seed used for example selection |
| `--system-prompt` | `None` | Default system prompt to prepend to records without one |

## Example

```bash
axim prepare-data \
    --repo my-org/chat-dataset \
    --config default \
    --split train \
    --train-examples 2000 \
    --val-examples 200 \
    --output-train data/train.jsonl \
    --output-val data/val.jsonl \
    --system-prompt "You are a helpful coding assistant."
```

!!! tip "Dataset format"
    The downstream SFT trainer accepts JSONL records with `messages`, `prompt`/`completion`, `instruction`/`input`/`output`, or plain `text`. The `prepare-data` command renders records into one of those chat formats depending on the source dataset schema.
