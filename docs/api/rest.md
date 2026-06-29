# OpenAI-compatible REST API

`axim serve` starts a FastAPI server with a subset of the OpenAI completion
endpoints. The server prints a timestamped line for every request showing the
endpoint, sampling parameters, prompt length, and completion statistics.

## Start the server

```bash
axim serve --axim model.axim --device cuda --port 8000
```

Omit `--port` to bind to a random free port in the 8000-9000 range. The examples below assume `--port 8000`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List the loaded model. |
| POST | `/v1/completions` | Raw prompt to completion. |
| POST | `/v1/chat/completions` | Chat-formatted completion. |
| POST | `/v1/debug/template` | Render the chat template for messages; no forward pass. |
| GET | `/health` | Health check. |

## Completion parameters

Both `/v1/completions` and `/v1/chat/completions` accept:

- `max_tokens` (default `256`)
- `temperature` (default `0.7`; `0` = greedy)
- `top_k` (default `50`; `None` = disabled when greedy)
- `repetition_penalty` (default `1.0`)
- `stop` — string or list of strings
- `stream` — `true` for SSE token chunks

For `/v1/chat/completions`, send OpenAI-style `messages` and the server renders
them with `Tokenizer.render_chat`, then appends `<|assistant_start|>` before
generating.

## Important: no KV cache

Every generated token runs a full forward pass over the full context. This is fine
for small models on a GPU but slow on CPU or at long contexts.

## Examples

### Chat completion (non-streaming)

=== "Linux / macOS"

    ```bash
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "axim",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 50,
        "temperature": 0.7
      }'
    ```

=== "Windows PowerShell"

    ```powershell
    Invoke-RestMethod -Uri "http://localhost:8000/v1/chat/completions" `
      -Method POST `
      -ContentType "application/json" `
      -Body '{
        "model": "axim",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 50,
        "temperature": 0.7
      }'
    ```

Response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "axim",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hello! How can I help?"},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 6,
    "total_tokens": 12
  }
}
```

### Chat completion (streaming)

Add `"stream": true` to receive Server-Sent Events. The Linux/macOS example uses
curl; PowerShell does not stream SSE conveniently, so use the non-streaming
example above for Windows.

=== "Linux / macOS"

    ```bash
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "axim",
        "messages": [{"role": "user", "content": "Count to five"}],
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": true
      }'
    ```

=== "Windows PowerShell"

    ```powershell
    Invoke-RestMethod -Uri "http://localhost:8000/v1/chat/completions" `
      -Method POST `
      -ContentType "application/json" `
      -Body '{
        "model": "axim",
        "messages": [{"role": "user", "content": "Count to five"}],
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": true
      }'
    ```

The server yields chunks like:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"1"}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":", 2"}}]}
...
data: [DONE]
```

### Raw completion

=== "Linux / macOS"

    ```bash
    curl http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "axim",
        "prompt": "Once upon a time",
        "max_tokens": 50,
        "temperature": 0.7,
        "stop": ["\n"]
      }'
    ```

=== "Windows PowerShell"

    ```powershell
    Invoke-RestMethod -Uri "http://localhost:8000/v1/completions" `
      -Method POST `
      -ContentType "application/json" `
      -Body '{
        "model": "axim",
        "prompt": "Once upon a time",
        "max_tokens": 50,
        "temperature": 0.7,
        "stop": ["`n"]
      }'
    ```

### Debug template

```bash
curl http://localhost:8000/v1/debug/template \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a bot."},
      {"role": "user", "content": "hi"}
    ]
  }'
```

Response:

```json
{
  "token_ids": [3, 4, ...],
  "decoded": "<|bos|> <|user_start|> You are a bot.\n\nhi <|user_end|> ...",
  "count": 12
}
```

### List models

```bash
curl http://localhost:8000/v1/models
```

### Health check

```bash
curl http://localhost:8000/health
```
