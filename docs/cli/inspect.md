# axim inspect

Inspect an `.axim` package and print a human-readable summary of its contents.

## Usage

```bash
axim inspect <path>
```

## Flags

| Argument | Default | Description |
|----------|---------|-------------|
| `path` | required | Path to the `.axim` file |

## Output

The command prints:

* Header information (magic, version, offset table)
* Section list (weights, config, tokenizer, metadata)
* A config summary (model size, vocabulary, sequence length, attention details)

## Example

```bash
axim inspect model.axim
```

!!! tip "Use before serving"
    Run `axim inspect` on a freshly exported or downloaded `.axim` to confirm it contains weights, config, and tokenizer before serving or fine-tuning.
