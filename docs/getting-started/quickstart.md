# Quickstart

This guide walks you through installing the Axiom Inference Engine, downloading a base model, running local inference, and starting the OpenAI-compatible API server.

## 1. Install the package

```bash
pip install -e .
```

For the SFT pipeline extras, use `pip install -e ".[sft]"`. See [Installation](installation.md) for details.

## 2. Download a base model

The fastest way to get a model is to pull one from the Hugging Face Hub:

```bash
axim download --repo Strawbercar/Axiom-V1-Base --output base.axim
```

This writes `base.axim` to the current directory, packaging weights, config, tokenizer, and metadata into a single file.

!!! tip "Already an `.axim` on the Hub?"
    If the repository already contains an `.axim` file, `axim download` copies it directly. Otherwise it stages safetensors shards, `config.json`, and `tokenizer.pkl`, then packs them locally.

## 3. Run local inference

Run a raw completion or a chat-formatted prompt:

```bash
# Raw completion
axim infer --axim base.axim --prompt "Once upon a time" --device cuda

# Chat mode
axim infer --axim base.axim --prompt "Explain quantum computing in one sentence." --chat --device cuda
```

In chat mode the CLI renders the chat scaffold, appends `<|assistant_start|>`, and stops when `<|assistant_end|>` is generated. `--repetition-penalty`, `--temperature`, `--top-k`, and `--max-tokens` are applied automatically.

!!! info "Default device"
    The default device is `cpu`. Pass `--device cuda` to use a GPU.

## 4. Start the API server

```bash
axim serve --axim base.axim --device cuda --port 8000
```

The server binds to `0.0.0.0` and picks a random free port if `--port` is omitted. The examples below assume `--port 8000`; use the logged port if you omit it.

## 5. Send a request

=== "Linux / macOS"

    ```bash
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "axim",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": true
      }'
    ```

=== "Windows (PowerShell)"

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

!!! tip "Streaming"
    Add `"stream": true` to receive token chunks via SSE. Non-streaming requests return the full completion in one response.

## Next steps

- Read the [format spec](../format/spec.md) to understand how `.axim` files are laid out.
- Open `webui/index.html` in a browser and point it at `http://localhost:8000`.
- Try the [SFT pipeline](../sft/index.md) to fine-tune a chat model.
