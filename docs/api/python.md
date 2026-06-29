# Python API reference

Import the public API from the `axim` package:

```python
from axim import load_axim, load_model, load_from_dir, Tokenizer, sample
```

The Python API is a thin layer over the `.axim` container format and the vendored
nanoChat model code. It is the same code path used by the `axim` CLI.

## `load_axim(path)`

Read an `.axim` file and return its raw sections.

```python
data = load_axim("model.axim")

cfg      = data["config"]      # dict with architecture hyperparameters
weights  = data["weights"]     # dict[str, torch.Tensor] (safetensors-backed)
enc      = data["tokenizer"]   # tiktoken Encoding
meta     = data["metadata"]    # dict (may be empty)
```

Use this when you only need the contents of the file, not a runnable model.

## `load_model(axim_path, device, dtype=None)`

Build a nanoChat `GPT` from an `.axim` file and move it to `device`.

```python
import torch

model, enc, cfg = load_model("model.axim", device=torch.device("cuda"), dtype=torch.bfloat16)
```

Returns:

- `model` — `axim._nanochat.gpt.GPT` in `eval()` mode.
- `enc` — raw tiktoken `Encoding`.
- `cfg` — model config dict (vocab size, hidden size, layers, sequence length, etc.).

The model is created with freshly initialized weights and then its state dict is
overwritten from the `.axim` weights. Passing `dtype` casts the parameters after
loading.

## `load_from_dir(model_dir, device, dtype=None)`

Same as `load_model`, but loads from a directory containing:

- `model.safetensors`
- `config.json`
- `tokenizer.pkl` (looked for in `model_dir` or its parent)

```python
model, enc, cfg = load_from_dir("./model-safetensors", device="cuda")
```

## `Tokenizer`

Wraps the tiktoken encoding stored in the `.axim` file.

```python
tok = Tokenizer(enc)
```

### Methods and properties

| Member | Description |
|--------|-------------|
| `tok.bos` | Integer id of `<|bos|>`. |
| `tok.encode(text)` | Encode ordinary text to a list of token ids. |
| `tok.decode(ids)` | Decode a list of token ids to a string. |
| `tok.encode_special("<|assistant_start|>")` | Encode a single registered special token. |
| `tok.render_chat(messages, max_len=2048)` | Render a chat conversation to token ids. |

`render_chat` produces the nanoChat chat scaffold:

```python
ids = tok.render_chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
], max_len=cfg["max_position_embeddings"])

# Prime generation by appending the assistant-start token.
ids.append(tok.encode_special("<|assistant_start|>"))
```

A leading system message is merged into the first user turn (nanoChat convention).
Conversations must alternate `user` / `assistant` starting with `user`.

## `sample`

Autoregressive sampling loop shared by the API server and `axim infer`.

```python
status = {}
for token_id in sample(
    model, tok, prompt_ids,
    max_tokens=256,
    temperature=0.7,
    top_k=50,
    device=torch.device("cuda"),
    repetition_penalty=1.0,
    stop_strings=["\n\nuser:"],
    status=status,
):
    print(tok.decode([token_id]), end="", flush=True)

print("\nfinish:", status)
```

`status` is populated when generation ends. Common reasons:

- `stop` with cause `<|assistant_end|> generated`
- `stop` with cause matching a stop string
- `length` when `max_tokens` is reached

`temperature <= 0` runs greedy decoding. `top_k` filters the sampling distribution.
There is no KV cache; every token performs a full forward pass.
