# Axiom-Inference-Engine

Package and inference stuff for **.axim** model files. One binary, everything inside.

This repo is basically made for my [Axiom lineup of models](https://huggingface.co/Strawbercar). As of right now that's just Axiom-V1-Base and maybe a finetuned model.

---

## What is .axim?

.axim is a binary format that stuffs everything you need into one file:

- **weights** (safetensors format, fast, memory-mappable)
- **config** (json with architecture details)
- **tokenizer** (pickled tiktoken encoding)
- **metadata** (training info, whatever else)

### Why not just use safetensors?

Safetensors is great for weights but it's *just* weights. You still need the config file, the tokenizer file, the metadata file... before you know it you've got 5 files scattered across your downloads folder.. oh and also i got sick of manuelly typing out massive commands and stuff

### Why not GGUF?

GGUF is cool but it's built for llama.cpp architectures. My models use nanoChat (Karpathy's thing) which has custom stuff like value embeddings, smear, backout, relu² activations which none of llama.cpp understands. So GGUF was a non-starter.

### Why not just a zip file?

Zips are fine but this is cooler.

---

## Format Spec (v1)

```
[Header]          13 bytes
  magic: "AXIM\0"   5 bytes
  version: uint32   4 bytes
  flags: uint32     4 bytes

[Section Table]   variable
  num_sections: uint32
  sections[]:
    type: uint32     (1=weights, 2=config, 3=tokenizer, 4=metadata)
    offset: uint64
    size: uint64
    name_len: uint32
    name: UTF-8

[Section Data]    aligned to 8-byte boundaries
  weights: safetensors blob
  config: JSON text
  tokenizer: pickle bytes
  metadata: JSON text
```

That's it. Simple. The weights section can be memory-mapped directly since it's just raw safetensors data.

---

## Installation

```bash
git clone https://github.com/StrawberCar/Axiom-Inference-Engine.git
cd Axiom-Inference-Engine
pip install -e .
```

nanochat is vendored under `axim/_nanochat/` (MIT, © Andrej Karpathy) — we maintain our own fork, so no separate clone is needed.

---

## Usage

Everything goes through the unified `axim` CLI once the package is installed (`pip install -e .` adds the `axim` console-script). You can also call it as `python -m axim ...`.

### CLI reference

| Command | What it does | Example |
|---|---|---|
| `axim serve` | Serve an .axim model over an OpenAI-compatible API | `axim serve --axim model.axim --device cuda` |
| `axim inspect` | Inspect an .axim file | `axim inspect model.axim` |
| `axim export` | Export a model directory to .axim | `axim export --model-dir ./model-safetensors --tokenizer-pkl ./tokenizer.pkl --output model.axim` |
| `axim download` | Download a base model from the HF Hub to .axim | `axim download --repo Strawbercar/Axiom-V1-Base --output base.axim` |
| `axim infer` | Run inference on a model | `axim infer --axim model.axim --prompt "hello world" --chat --device cuda` |
| `axim prepare-data` | Prepare an SFT dataset JSONL from the HF Hub | `axim prepare-data --repo HuggingFaceTB/smol-smoltalk --output-train train.jsonl --output-val val.jsonl` |
| `axim sft` | Fine-tune an .axim model (SFT) | `axim sft --config configs/sft_example.json --base-model base.axim --train train.jsonl --val val.jsonl --output tuned.axim --device cuda --dtype bf16` |

`axim export` takes either `--tokenizer-pkl PKL` (a local pickle) or `--repo-id REPO` (pull the tokenizer straight from the HF Hub). `axim infer` takes either `--axim PATH` or `--model-dir DIR` as its model source. Run `axim <command> --help` for the full flag set on any subcommand; `axim sft --help` shows every training knob.

### Python API

```python
from axim import load_axim, load_model, Tokenizer, sample

data = load_axim("model.axim")           # weights/config/tokenizer/metadata
model, enc, cfg = load_model("model.axim", device="cuda")
tok = Tokenizer(enc)
ids = tok.render_chat([{"role": "user", "content": "hi"}])
```

`load_axim` returns the raw section dict (weights as `dict[str, torch.Tensor]`, config as a dict, tokenizer as a tiktoken `Encoding`, metadata as a dict). `load_model` builds an actual nanochat GPT ready for inference. `Tokenizer` wraps the encoding with `encode` / `decode` / `render_chat`, and `sample` is the autoregressive sampling loop. See `axim/__init__.py` for the full re-export list.

### Run the API server

```bash
axim serve --axim model.axim --device cuda
```

Omit `--port` and it picks a random free one. The server logs every request to its
console — one timestamped block per request showing the endpoint, sampling params,
prompt token count, then completion length / elapsed time / tokens-per-second / stop
reason when it finishes:

```
[14:02:31] #   1 → POST /v1/chat/completions  prompt=128tok  msgs=4  stream=on  max_tokens=256  temp=0.7  top_k=50  rep_pen=1.0
[14:02:33] #   1 ← done  87tok in 2.04s (42.6 tok/s)  finish=stop (<|assistant_end|> generated)
```

Endpoints:

| Method | Path | What it does |
|--------|------|--------------|
| POST | `/v1/chat/completions` | OpenAI-style chat. `stream: true` returns SSE token chunks. |
| POST | `/v1/completions` | Raw prompt → completion. `stream: true` returns SSE token chunks. |
| POST | `/v1/debug/template` | Render the chat template for a set of messages (no forward pass). returns token ids + decoded text + count. |
| GET | `/v1/models` | List the loaded model. |
| GET | `/health` | Health check. |

Both completion endpoints accept `temperature`, `max_tokens`, `top_k`,
`repetition_penalty`, `stop` (string or list), and `stream`. There's no KV cache, so
each token is a full forward pass. fine for a 1.38B model on a GPU, slow on CPU.

Then hit it with standard OpenAI-style requests:

**Linux / macOS:**
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

**Windows (PowerShell):**
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

Supports streaming too, just add `"stream": true`.

### Web UI

Open `webui/index.html` in a browser — it's a self-contained dev console (no build
step, no server of its own; it just talks to the API). Start the API with
`axim serve --axim model.axim --device cuda` first, then point the UI at it.

- **Simple mode** — single chat thread. Conversations save to `localStorage`; rename,
  reopen, close, or delete them from the sidebar. Enter sends, Shift+Enter for a
  newline.
- **Advanced mode** (toggle in the top bar) — adds a right-hand panel with sampling
  controls (temperature, top-k, max tokens, repetition penalty, system prompt, stop
  sequences), a **Raw completion** mode that posts verbatim to `/v1/completions`,
  a connection tester, a **preview prompt template** button (hits
  `/v1/debug/template`), and per-message usage stats + regenerate / edit-and-resend.
- **Token streaming** — toggle the "stream tokens" switch (on by default). A live
  status line under the composer shows `generating · N tok · X tok/s` while tokens
  arrive, then `done · N tok in Y.Ys · X tok/s` when the request finishes.

Point it at your API endpoint (default `http://localhost:8000`) in the Advanced
panel and click **test connection**.

### Fine-tune a model (SFT)

The repo includes a small SFT pipeline for turning a base `.axim` into a chat-tuned
`.axim`. The quick path is the Colab notebook (one command per cell, runs on a single
A100):

```bash
# 1. download a base model from the HF Hub
axim download --repo Strawbercar/Axiom-V1-Base --output base.axim

# 2. build train/val JSONL from an HF dataset (ShareGPT-style {role, content} turns)
axim prepare-data --repo HuggingFaceTB/smol-smoltalk \
    --train-examples 50000 --val-examples 500 \
    --output-train train.jsonl --output-val val.jsonl

# 3. fine-tune — writes a merged .axim you can drop straight into the API server
axim sft --config configs/sft_example.json \
    --base-model base.axim --train train.jsonl --val val.jsonl \
    --output tuned.axim --device cuda --dtype bf16
```

`axim sft` supports LoRA / partial / full fine-tuning, best-fit packing with
assistant-only loss masking, Muon or AdamW, warmup + warmdown scheduling, in-training
static + random sampling so you can watch the model learn, and periodic checkpoints.
See `configs/sft_example.json` for every knob, or open `notebooks/axim_sft_colab.ipynb`
for the end-to-end Colab recipe.

---

## File Structure

```
Axiom-Inference-Engine/
├── axim/
│   ├── __init__.py          # public API: load_axim, Tokenizer, load_model, sample, ...
│   ├── core.py              # .axim format: read / write / inspect
│   ├── model.py             # load_model() / load_from_dir() — the one model-loading dance
│   ├── tokenizer.py         # Tokenizer: encode / decode / render_chat
│   ├── generate.py          # sample() — autoregressive sampling loop
│   ├── serve.py             # OpenAI-compatible API server (FastAPI + SSE)
│   ├── cli.py               # unified `axim <command>` dispatcher
│   ├── __main__.py           # `python -m axim`
│   ├── sft/
│   │   ├── __init__.py
│   │   ├── data.py          # SFT dataset loader + packing
│   │   └── train.py         # SFT fine-tuner (LoRA / partial / full)
│   └── _nanochat/           # vendored nanochat (MIT, (c) Andrej Karpathy) — our own fork
│       ├── __init__.py
│       ├── gpt.py
│       ├── common.py
│       ├── optim.py
│       └── flash_attention.py
├── configs/
│   └── sft_example.json     # example SFT training config
├── data/
│   └── example_sft.jsonl    # tiny example SFT dataset
├── webui/
│   └── index.html           # dev console web UI (self-contained, talks to the API)
├── notebooks/
│   └── axim_sft_colab.ipynb # one-command-per-cell Colab SFT pipeline
├── pyproject.toml           # deps + `axim` console-script
└── README.md                # you're reading it
```

---

## Performance

Loading an .axim file is basically as fast as loading safetensors because the weights section *is* safetensors. The extra overhead is:

- Reading a ~200 byte header + section table
- Parsing a few KB of JSON
- Unpickling the tokenizer

So microseconds. The weights load via `safetensors.load()` which is already optimized.

---

## Limitations

- Right now this is built specifically for nanoChat models. If you're using Llama or GPT-2 or whatever, the format works but the inference scripts won't.
- No KV cache optimization yet in the API server. It does full forward passes. Fine for small models, not great for big ones.
- No quantization support. Weights are stored as full precision (fp32/bf16). If you need 4-bit or 8-bit, that's a future problem.
- Only supports the RustBPETokenizer from nanoChat. If your model uses a different tokenizer, you'll need to adapt the scripts.

---

## Roadmap (maybe)

- [ ] KV cache for faster inference
- [ ] Quantization support (GGUF-style Q4_K_M etc)
- [x] Streaming tokenizer (decode tokens as they arrive)
- [ ] Multi-model support in API server
- [x] Better web UI with conversation history
- [ ] Model zoo / registry
- [ ] Digital signatures for .axim files

No promises though.

---

## About

Made with around $100 by Strawbercar, released under the Axiom Research Project.

**Special thanks to:**
- **Andrej Karpathy** — for nanoChat and being an inspiration to every indie ML dev
- **HuggingFace** — for transformers, safetensors, and the hub
- **The llama.cpp team** — for proving that small, fast inference is possible
- **The tiktoken/rustbpe folks** — for tokenizers that actually work
- **Anyone who's ever released open source ML tools** — you know who you are

nanochat is vendored under `axim/_nanochat/` (MIT, © Andrej Karpathy) — we maintain our own fork.