# axim CLI Reference

The `axim` package exposes a single unified command-line interface. Every subcommand is invoked as `axim <command>` and is also available as `python -m axim <command>`.

## Running commands

```bash
axim <command> [args...]
python -m axim <command> [args...]
```

For help on any command, append `--help`:

```bash
axim serve --help
axim infer --help
```

!!! tip "Discover commands"
    Run `axim --help` to list every subcommand with a one-line description.

## Subcommands

| Command | Purpose |
|---------|---------|
| [`axim serve`](serve.md) | Serve an `.axim` model behind an OpenAI-compatible API |
| [`axim inspect`](inspect.md) | Print the internal layout of an `.axim` file |
| [`axim export`](export.md) | Pack a model directory (safetensors + config + tokenizer) into `.axim` |
| [`axim download`](download.md) | Download a model from the Hugging Face Hub and produce `.axim` |
| [`axim infer`](infer.md) | Run raw or chat-mode text generation |
| [`axim prepare-data`](prepare-data.md) | Turn a Hub dataset into SFT-ready JSONL train/val files |
| [`axim sft`](sft.md) | Fine-tune an `.axim` model on supervised chat data |

## Quick start

```bash
# download a base model
axim download --repo my-org/my-model --output model.axim

# ask it a question
axim infer --axim model.axim --chat --prompt "What is 2+2?"

# serve it as a local API
axim serve --axim model.axim --port 8000
```
