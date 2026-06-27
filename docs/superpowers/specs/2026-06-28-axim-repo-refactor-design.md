# axim repo refactor — design

**Date:** 2026-06-28
**Status:** approved (pending user review of this spec)
**Topic:** Restructure the Axiom-Inference-Engine repository into a self-contained, installable Python package with a unified CLI and a vendored copy of nanochat.

## Goal

Make the repo cleaner, easier, and more consistent to use. Concretely: kill the duplicated model-loading / tokenizer / sampling code spread across scripts, remove the `sys.path` hacks in every file, give every capability one entry point, and make the whole thing `pip install -e .`-able with no external nanochat clone required.

This is a **restructure**, not a feature add. No model behavior, endpoints, sampling features, or format changes.

## Decisions (settled with the user)

- **Breakage tolerance:** full rework. Old `scripts/*.py` invocations go away; the Colab notebook is rewritten to use the new CLI. Muscle memory breaks; that's accepted.
- **nanochat dependency:** vendor the needed nanochat modules directly into the `axim` package as a private `_nanochat/` subpackage. We own the fork; we do not track upstream. One clone + `pip install -e .`, no separate nanochat clone, no `sys.path` hacks.
- **Scope:** whole repo in one pass — package, vendored model, inference, SFT pipeline, unified CLI, webui (docs only), notebook, README, requirements/gitignore.
- **Layout:** flat modules in `axim/` with one `sft/` subpackage and a private `_nanochat/` vendor dir.

## Out of scope

- **The `.axim` binary format is unchanged.** No version bump, no new section types. `load_axim`, `write_axim`, `print_axim_info` keep their existing signatures so every `.axim` file already on disk still loads.
- **Quantization / `.qaxim`** (the subject of the earlier feasibility discussion) is a separate future feature, not part of this refactor.
- No new model behavior, no new HTTP endpoints, no new sampling controls. Same capabilities, one consistent way to reach each.

## Target layout

```
axim/
  __init__.py          # public API: load_axim, write_axim, print_axim_info, Tokenizer, load_model, sample
  core.py               # .axim format: load/write/inspect  (UNCHANGED — stable API)
  model.py              # load_model() — single home for the init_weights + load_state_dict dance
  tokenizer.py          # Tokenizer: encode/encode_special/decode/bos/render_chat  (deduped)
  generate.py           # sample() — single autoregressive sampling loop, shared by serve + infer
  serve.py              # build_app() FastAPI + verbose per-request logging  (was api_server.py)
  cli.py                # unified argparse subcommand dispatcher
  __main__.py           # `python -m axim` -> axim.cli:main
  sft/
    __init__.py
    data.py             # was sft_data.py — dataset loader + packing
    train.py            # was sft_train.py — the trainer
  _nanochat/            # vendored, private — we own it
    __init__.py         # attribution: "Vendored from karpathy/nanochat (MIT)"
    gpt.py
    common.py
    optim.py
    flash_attention.py
pyproject.toml         # deps + `axim` console-script -> axim.cli:main
configs/sft_example.json
data/example_sft.jsonl
webui/index.html        # unchanged (self-contained, talks to HTTP API)
notebooks/axim_sft_colab.ipynb   # moved from repo root; rewritten to use the CLI
README.md               # rewritten
```

`scripts/` is **removed** — the package + CLI replace it.

## Components

### axim.core (unchanged)
Stays exactly as-is: `load_axim`, `write_axim`, `print_axim_info`, `SectionType`, `AXIMHeader`. The format is the product; we don't touch it. `print_axim_info` is reused by `axim inspect`.

### axim.model — single source for model loading
`load_model(axim_path, device, dtype=None) -> (model, tokenizer, cfg)`.

Owns the one copy of the dance currently copy-pasted 5× across `api_server.py`, `test_inference.py` (×2), and `sft_train.py` (×2):
1. `data = load_axim(axim_path)`
2. `sys.path`-free `from axim._nanochat.gpt import GPT, GPTConfig`
3. Build `GPTConfig` from `data["config"]`
4. `model = GPT(config); model.init_weights()`
5. `model.load_state_dict(data["weights"], strict=True, assign=True)`
6. `model.to(device); model.eval()`
7. Return `(model, data["tokenizer"], data["config"])`

`serve`, `infer`, and `sft.train` all call this. No script touches `GPTConfig` / `init_weights` / `load_state_dict` directly anymore.

### axim.tokenizer — single source for tokenization
`class Tokenizer` (the former `_TokenizerWrap`): wraps the pickled tiktoken `Encoding` with `encode`, `encode_special`, `decode`, `bos` (property), `render_chat(messages, max_len)`.

Replaces `_TokenizerWrap` in `api_server.py` and the inline tokenizer handling in `test_inference.py`. `sft.data` consumes the primitives (`encode`, `encode_special`) for building training sequences with assistant-only masking — it does **not** go through `render_chat` (that's an inference-time chat scaffold).

### axim.generate — single source for sampling
`sample(model, tokenizer, prompt_ids, max_tokens, temperature, top_k, device, repetition_penalty=1.0, stop_strings=None, status=None) -> generator[int]`.

A generator yielding token ids, with the `status` dict finish-reason tracking we just added to `api_server._generate` (`status["reason"]` = "stop" | "length", `status["cause"]` = detail). Stops on `<|assistant_end|>` / `<|bos|>` and on user stop strings (substring match, stop string not emitted). HF-style repetition penalty, top-k, greedy vs sampling.

This replaces both `api_server._generate` and `test_inference.generate` — they had diverged. `serve` and `infer` both call it.

### axim.serve — the API server
`build_app(model, tokenizer, cfg, device) -> FastAPI` plus a `serve(axim_path, host, port, device)` entry the CLI calls. Endpoints unchanged (`/v1/chat/completions`, `/v1/completions`, `/v1/debug/template`, `/v1/models`, `/health`), same OpenAI-compatible shapes, same SSE streaming, same verbose per-request console logging we added. Uses `axim.generate.sample` and `axim.tokenizer.Tokenizer`.

### axim.cli — unified entry point
`main()` = argparse with subcommands. Each subcommand is a thin parser that calls the matching module function. `python -m axim` and the `axim` console script both route here.

```
axim serve        --axim PATH [--host HOST] [--port PORT] [--device DEVICE]
axim inspect      PATH
axim export       --model-dir DIR (--tokenizer-pkl PKL | --repo-id REPO) --output OUT
axim download     --repo REPO --output OUT [--filename F] [--model-dir DIR] [--keep-stage]
axim infer        (--axim PATH | --model-dir DIR) --prompt TEXT [--max-tokens N]
                  [--temperature F] [--top-k K] [--repetition-penalty F] [--chat]
                  [--system-prompt TEXT] [--device DEVICE]
axim prepare-data --repo REPO --output-train TR --output-val VA [--train-examples N]
                  [--val-examples N] [--split S] [--config C] [--seed N] [--system-prompt TEXT]
axim sft          --config CFG [--base-model PATH] [--train TR] [--val VA] [--output OUT]
                  [--device D] [--dtype DT] ...   (full sft_train flag set)
```

Help text convention: imperative mood, capitalized first word, one-line summary (e.g. "Serve an .axim model over an OpenAI-compatible API"). `inspect` finally gets real argparse (today it crashes on `--help`).

### axim.sft — training pipeline
- `axim/sft/train.py` ← `scripts/sft_train.py` (moved, imports rewritten to `axim._nanochat` / `axim.model` / `axim.tokenizer`). Behavior preserved.
- `axim/sft/data.py` ← `scripts/sft_data.py` (moved).
- `prepare_sft_data.py` logic → the `axim prepare-data` subcommand (small enough to live as a function, likely in `axim/sft/data.py`, called from `cli.py`).

### axim._nanochat — vendored nanochat (we own it)
- Copy `nanochat/nanochat/gpt.py`, `common.py`, `optim.py`, `flash_attention.py` into `axim/_nanochat/`.
- Rewrite internal imports `from nanochat.X import ...` → relative `from .X import ...` so the subpackage is self-contained. `axim.model` imports `from axim._nanochat.gpt import GPT, GPTConfig`.
- **Vendor faithfully — do not trim behavior.** DDP paths, `compute_init`/`compute_cleanup`, `DistMuonAdamW`, `is_ddp_initialized`, the `COMPUTE_DTYPE` global all stay so `sft.train` runs identically. Only import paths change.
- `_nanochat/__init__.py` carries attribution: "Vendored from karpathy/nanochat (MIT). See LICENSE in this directory." Copy nanochat's LICENSE into `_nanochat/`.

### pyproject.toml — single source of truth
- Build: setuptools (or hatchling) backend.
- Deps: `torch>=2.0`, `safetensors>=0.4`, `tiktoken`, `fastapi`, `uvicorn`, `numpy`, `huggingface_hub`, `datasets`. (`requests` is not imported directly anywhere — it arrives transitively via `huggingface_hub`.)
- Optional extras: `sft` (`accelerate`, `tqdm`), `flash` (`flash-attn`).
- Entry point: `axim = axim.cli:main`.
- `requirements.txt` is **removed** — pyproject is the only dep manifest, no drift.

### webui/index.html — unchanged
Self-contained HTML that talks to the HTTP API. The API keeps the same `/v1/*` routes, so the webui needs zero code changes. Only the README section about it is updated (endpoint default, CLI to start the server).

### notebooks/axim_sft_colab.ipynb — rewritten
Moved from repo root to `notebooks/`. Cells rewritten: clone this repo → `pip install -e .` → `axim download ...` → `axim prepare-data ...` → `axim sft ...`. No `python scripts/...`, no nanochat clone cell, no `sys.path` cells. This is updated **last**, after the CLI is verified working, so it's never in a broken state.

### README.md — rewritten
- Install: `git clone` + `pip install -e .` (one clone, one install — no separate nanochat clone).
- CLI reference table (the 7 subcommands).
- Package API section (`from axim import load_axim, load_model, Tokenizer, sample`).
- File structure (the target layout above).
- SFT recipe (uses the CLI).
- Web UI section (unchanged behavior, new start command `axim serve`).
- Vendored-nanochat credit (Karpathy, MIT) — keep the existing About credit, add a `_nanochat/` note.

### .gitignore — fixed
- Remove the blanket `*.ipynb` ignore — the Colab notebook is now a first-class tracked artifact under `notebooks/`. (If scratch notebooks need ignoring later, scope a rule to a scratch dir rather than `*.ipynb` globally.)
- Drop the stray `.README.md` line.
- Drop `nanochat/` (no longer an external clone).
- Add `*.egg-info/`, `dist/`, `build/` (packaging artifacts).
- Keep `*.axim`, `out/`, `__pycache__/`, `.claude`, `.remember`.

## Data flow

Inference: `axim serve` / `axim infer` → `axim.model.load_model()` → `axim.tokenizer.Tokenizer` builds prompt ids → `axim.generate.sample()` yields tokens → `serve` streams SSE chunks / `infer` prints decoded text.

Training: `axim sft` → `axim.sft.train` → `axim.model.load_model()` for the base → `axim.sft.data` builds packed training sequences (using `Tokenizer` primitives) → trainer steps the model → `axim.core.write_axim()` saves the merged result.

Both flows load through `axim.model.load_model` and tokenize through `axim.tokenizer.Tokenizer`. No duplication of either anywhere.

## Verification plan

After each component, before moving on:
- `python -m py_compile` on every moved/created `.py`.
- `axim <subcommand> --help` works for all 7 subcommands (catches argparse + import wiring).
- `axim inspect <existing .axim>` produces the same output as the old `scripts/inspect_axim.py`.
- `axim infer --axim <model> --prompt "hello" --max-tokens 8` runs end to end (only on a GPU per the no-CPU-model-runs constraint; on CPU, limit to `--help` + `inspect` + `py_compile` smoke tests).
- `axim serve` boots and `/health` + `/v1/models` respond.
- `axim sft --help` shows the full flag set; a tiny 2-step dry run if a GPU is available.
- Notebook: run through the cells once the CLI is verified.

The risky bits are (a) the nanochat vendoring + import rewrite and (b) `sft_train.py` running identically after the move — both get extra attention and a smoke test.

## Migration order (high level — detailed plan comes next)

1. pyproject + package skeleton + `_nanochat` vendoring + import rewrite. Verify imports resolve.
2. `axim.model`, `axim.tokenizer`, `axim.generate` — extract and dedup.
3. `axim.serve` — port `api_server.py` onto the shared modules.
4. `axim.cli` — wire all subcommands; port `inspect`/`export`/`download`/`infer`/`prepare-data` as thin functions.
5. `axim.sft` — move `sft_train.py` / `sft_data.py` / `prepare_sft_data.py`, rewrite imports.
6. Delete `scripts/`; remove `requirements.txt`.
7. Move + rewrite notebook; rewrite README; fix `.gitignore`.
8. Full smoke-test pass.

Each step is independently verifiable and committable. Nothing is deleted until its replacement is verified.