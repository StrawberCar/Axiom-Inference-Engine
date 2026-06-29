# axim download

Download a model from the Hugging Face Hub and produce a single `.axim` file.

## Usage

```bash
axim download --repo <repo> --output <path> [--filename <name>] [--model-dir <dir>] [--keep-stage]
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--repo` | required | Hugging Face repository id |
| `--output` | required | Destination `.axim` path |
| `--filename` | `None` | Specific `.axim` file to download when the repo already contains one |
| `--model-dir` | `None` | Staging directory; when omitted a temporary directory is used |
| `--keep-stage` | `False` | Keep the staging directory after packing (only applies when `--model-dir` is omitted) |

## Two download paths

### Direct `.axim` download

If the repository already contains `.axim` files, the requested (or first) `.axim` is downloaded directly and copied to `--output`.

```bash
axim download --repo my-org/my-model --output model.axim
```

### Safetensors staging, merge, and pack

If no `.axim` is present, the command:

1. Stages all repo files into a local directory
2. Merges sharded safetensors when `model.safetensors.index.json` exists
3. Loads weights, config, tokenizer, and optional `nanochat_metadata.json`
4. Packs everything into the output `.axim`
5. Removes the staging directory unless `--keep-stage` is set

## Example

```bash
axim download --repo my-org/my-model --output model.axim --keep-stage
```

!!! warning "Disk space"
    Staged repos contain safetensors shards, config, and tokenizer. Ensure you have enough free disk space, or provide `--model-dir` to control the staging location.
