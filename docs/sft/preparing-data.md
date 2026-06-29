# Preparing SFT data

SFT training consumes one JSON object per line. The canonical format is
nanoChat-style chat JSONL:

```jsonl
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 is 4."}]}
{"messages": [{"role": "user", "content": "Say hello"}, {"role": "assistant", "content": "Hello!"}]}
```

## `axim prepare-data`

Download and normalize a dataset from the Hugging Face Hub:

```bash
axim prepare-data --repo HuggingFaceTB/smol-smoltalk \
    --train-examples 2000 --val-examples 200 \
    --output-train train.jsonl --output-val val.jsonl \
    --system-prompt "You are a helpful coding assistant."
```

`prepare-data` recognizes several common dataset shapes and normalizes them to
`{"messages": [...]}`:

- `messages` — passed through.
- `conversations` — ShareGPT style with `from`/`value` mapped to `role`/`content`.
- `instruction` + `output` (+ optional `input`) — Alpaca style.
- `problem` + `solution` (+ optional `reasoning`).
- `question` + `answer`.
- `prompt` + `completion`.
- `text` — treated as a single assistant turn with an empty user turn.

Records are filtered to keep only rows that contain at least one user message
and one assistant message.

## Conversation format

Every training record must end up as:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

A leading `system` message is allowed. Conversations must alternate `user` /
`assistant` starting with `user`.

Assistant content may also be a list of typed parts:

```json
{"role": "assistant", "content": [
  {"type": "text", "text": "..."},
  {"type": "python", "text": "print(1)"},
  {"type": "python_output", "text": "1"}
]}
```

## `normalize_record` behavior

During training, each JSONL object is passed through `normalize_record` in
`axim/sft/data.py`. It accepts these shapes:

- `{"messages": [...]}`
- `{"prompt": "...", "completion": "..."}`
- `{"instruction": "...", "output": "...", "input"?: "..."}`
- `{"text": "..."}`

For all shapes, a `default_system` prompt can be injected when the record lacks
a leading system message.

## System prompt injection

Use the `--system-prompt` CLI flag (or `system_prompt` in the config file) to
prepend a system message to every record that does not already have one:

```bash
axim sft --system-prompt "You are a helpful assistant." ...
```

The system message is merged into the first user turn when rendering (nanoChat
convention), so it does not appear as a separate scaffold token sequence during
inference.

## `train_on` — assistant vs all

The `train_on` config field controls which positions contribute to the loss:

- `"assistant"` (default) — only assistant content tokens are supervised. User
  turns, special scaffold tokens, and padding are masked to `-1` (ignored).
- `"all"` — every content token is supervised, including user turns. Useful for
  continued pre-training on chat-formatted data.

This is implemented by `render_conversation` in `axim/sft/data.py`, which emits
`(ids, mask)` pairs where `mask == 1` marks supervised positions.
