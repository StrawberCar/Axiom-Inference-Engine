# axim export

Pack a model directory into a single `.axim` file.

The source directory must contain `model.safetensors` and `config.json`. The tokenizer can be supplied explicitly or discovered automatically.

## Usage

```bash
axim export --model-dir <dir> --output <path> [--tokenizer-pkl <path> | --repo-id <id>]
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir` | required | Directory containing `model.safetensors` and `config.json` |
| `--tokenizer-pkl` | `None` | Path to a `tokenizer.pkl` file |
| `--repo-id` | `None` | Hugging Face repo id to download `tokenizer.pkl` from |
| `--output` | required | Destination `.axim` path |

## Tokenizer resolution

The command looks for a tokenizer in this order:

1. `--tokenizer-pkl` if given
2. `--repo-id` if given (downloads `tokenizer.pkl` from the Hub)
3. `model_dir/tokenizer.pkl`
4. `model_dir/../tokenizer.pkl`

If none are found, export fails.

## Metadata auto-detection

If `model_dir/nanochat_metadata.json` exists, it is read automatically and embedded in the `.axim` package as model metadata.

## Example

```bash
axim export \
    --model-dir ./my_model \
    --output my_model.axim
```

!!! note "Explicit tokenizer from the Hub"
    ```bash
    axim export --model-dir ./my_model --repo-id my-org/my-model --output model.axim
    ```
