# axim serve

Serve an `.axim` model behind an OpenAI-compatible HTTP API.

## Usage

```bash
axim serve --axim <path> [--host <host>] [--port <port>] [--device <device>]
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--axim` | required | Path to the `.axim` package to load |
| `--host` | `0.0.0.0` | Network interface to bind |
| `--port` | `None` | TCP port; when omitted, the server picks a random free port |
| `--device` | `cpu` | Device to load the model on (`cpu`, `cuda`, `mps`, etc.) |

## Example

```bash
axim serve --axim model.axim --host 127.0.0.1 --port 8080 --device cuda
```

!!! example "Request logging"
    The server prints a startup line with the bound host/port and a timestamped block for every request:

    ```
    server: http://0.0.0.0:8080
    [14:02:31] #   1 → POST /v1/chat/completions  prompt=128tok  msgs=4  stream=on  max_tokens=256
    [14:02:33] #   1 ← done  87tok in 2.04s (42.6 tok/s)  finish=stop (<|assistant_end|> generated)
    ```

!!! note "OpenAI-compatible endpoints"
    Use the displayed host:port in clients that expect an OpenAI base URL, for example `http://127.0.0.1:8080/v1`.
