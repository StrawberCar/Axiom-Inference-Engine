# axim infer

Run text generation on an `.axim` model or a raw model directory.

## Usage

```bash
axim infer (--axim <path> | --model-dir <dir>) [flags]
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--axim` | `None` | Path to the `.axim` package to load |
| `--model-dir` | `None` | Path to a directory with `model.safetensors` and `config.json` |
| `--prompt` | `Once upon a time` | Input prompt text |
| `--max-tokens` | `50` | Maximum number of new tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-k` | `50` | Top-k sampling cutoff |
| `--repetition-penalty` | `1.0` | Repetition penalty; `1.0` disables it |
| `--device` | `cpu` | Device to run on (`cpu`, `cuda`, `mps`, etc.) |
| `--chat` | `False` | Use chat mode with special tokens |
| `--system-prompt` | `None` | System prompt prepended in chat mode |
| `--stream` | `True` | Stream generated tokens to the terminal as they arrive |
| `--no-stream` | — | Disable streaming; collect the full completion before printing |

## Streaming output

By default `axim infer` streams each generated token to the terminal as soon as it is produced. The prompt is printed first, then tokens are appended in real time, followed by a final status line showing token count, elapsed time, and tokens per second.

To fall back to the old collect-then-print behavior, pass `--no-stream`.

## Chat mode

When `--chat` is set, the prompt is rendered as a conversation and the model is primed with `<|assistant_start|>`. Generation stops when the model emits `<|assistant_end|>`.

## Example

```bash
axim infer --axim model.axim \
    --chat \
    --system-prompt "You are a helpful assistant." \
    --prompt "What is the capital of France?" \
    --max-tokens 100 \
    --temperature 0.8
```

!!! note "Repetition penalty"
    Values above `1.0` reduce the likelihood of repeating tokens. The default of `1.0` leaves logits unchanged.
