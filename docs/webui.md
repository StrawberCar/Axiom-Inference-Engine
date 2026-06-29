# Web UI

`webui/index.html` is a self-contained dev console. It has no build step and no
backend of its own; it talks to the `axim serve` REST API. Open the file directly
in a browser.

## Quick start

1. Start the API server:

   ```bash
   axim serve --axim model.axim --device cuda
   ```

2. Open `webui/index.html` in a browser.
3. Switch to **Advanced** mode (top bar) and set the endpoint, e.g.
   `http://localhost:8000`.
4. Click **test connection**. The status dot turns green and the model id appears
   next to it when the UI can reach `/health` and `/v1/models`.

## Simple mode

Simple mode is a single chat thread.

- Conversations are stored in `localStorage` under the keys `axim.convos` and
  `axim.activeId`.
- The sidebar lists all saved conversations, sorted by most recent.
- Click a conversation to reopen it.
- Hover a conversation to rename or delete it.
- Click the top bar title to rename the current conversation.
- **Enter** sends a message. **Shift+Enter** inserts a newline.
- The **New chat** button starts a fresh thread.

## Advanced mode

Toggle **Advanced** in the top bar to open the right-hand panel.

### Connection

- **Endpoint** — URL of the `axim serve` API.
- **test connection** — hits `/health` and `/v1/models`.

### Generation controls

- **temperature** — sampling temperature; `0` is greedy.
- **top-k** — top-k filtering (blank defaults to `50`).
- **max tokens** — maximum tokens to generate.
- **repetition penalty** — HF-style repetition penalty.
- **system prompt** — prepended as a `system` message in chat mode.
- **stop sequences** — comma-separated strings that stop generation.
- **stream tokens** — when on, tokens stream in via SSE.

### Generation mode

- **Chat** — sends `messages` to `/v1/chat/completions`.
- **Raw completion** — sends the composer text verbatim to `/v1/completions`
  with a BOS prepended by the server.

### Preview prompt template

The **preview prompt template** button hits `/v1/debug/template` and shows:

- the rendered chat text,
- the token id list,
- the token count.

This is useful for verifying exactly how the tokenizer turns messages into ids.

### Streaming status and usage

While generating, a status line under the composer shows:

```
generating · N tok · X tok/s
```

When finished:

```
done · N tok in Y.Ys · X tok/s
```

In Advanced mode, each assistant message displays usage stats:

```
P↑ C↓ (T)
```

where `P` is prompt tokens, `C` is completion tokens, and `T` is total tokens.

### Message actions

- **regenerate** (assistant messages) — delete the assistant reply and generate it
  again from the last user message.
- **edit & resend** (user messages) — load the message back into the composer,
  truncate the thread to before that message, and resend after editing.
- **copy** — copy the message text to the clipboard.

## Mobile layout

On narrow screens the sidebar and advanced panel become floating overlays that
slide in/out instead of fixed columns.
