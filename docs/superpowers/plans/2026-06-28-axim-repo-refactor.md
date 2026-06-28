# axim repo refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure Axiom-Inference-Engine into a self-contained, `pip install -e .`-able Python package with a unified `axim` CLI, vendoring nanochat in and deduping the model/tokenizer/sampling code that's copy-pasted across scripts.

**Architecture:** Flat `axim/` package (core format + model + tokenizer + generate + serve + cli) with one `axim/sft/` subpackage and a private `axim/_nanochat/` vendored subpackage. Every capability has one entry point (`axim <subcommand>`). The `.axim` binary format and `axim.core` API are unchanged; this is a pure restructure.

**Tech Stack:** Python 3.10+, PyTorch, safetensors, tiktoken, FastAPI/uvicorn, huggingface_hub, datasets, filelock, setuptools build backend.

## Global Constraints

- **Do not change the `.axim` binary format.** `axim.core.load_axim`/`write_axim`/`print_axim_info` keep their exact signatures and behavior. Existing `.axim` files must still load.
- **Do not change model/inference behavior.** Same sampling, same endpoints, same chat scaffold, same SFT schedule. The refactor moves code and dedups; it does not alter semantics.
- **No `sys.path.insert` anywhere in the final code.** The package is importable as `axim` after `pip install -e .`.
- **Vendor nanochat faithfully.** Copy `gpt.py`, `common.py`, `optim.py`, `flash_attention.py` + LICENSE into `axim/_nanochat/`; rewrite only import paths (`from nanochat.X` → `from axim._nanochat.X` or relative). Do not trim DDP/`compute_init`/`DistMuonAdamW`/`COMPUTE_DTYPE` — `sft` must run identically.
- **New transitive dep: `filelock`** (nanochat/common.py imports it). Add to pyproject.
- **No CPU model runs.** Per user constraint, never run forward passes / generation / training on CPU. Smoke tests on CPU are limited to `py_compile`, `--help`, `axim inspect`, and import wiring. Any test that loads a model or generates requires `--device cuda` and is only run if a GPU is available; otherwise mark the step "skipped (no GPU)" and note it.
- **Frequent commits.** Commit after each task. Do not push unless asked.
- **Working directory:** `C:\Users\Bleh\Documents\Utilities\GGUF-Conv\axie`. Shell is PowerShell for `Bash`-tool-incompatible steps, but prefer the Bash tool for git/file ops. Use forward slashes.
- **Nothing is deleted until its replacement is verified.** `scripts/` stays until Task 11, after the CLI is confirmed working.

---

## File Structure (target)

```
axim/
  __init__.py          # public API
  core.py               # UNCHANGED — .axim format
  model.py              # NEW — load_model / load_from_dir
  tokenizer.py          # NEW — Tokenizer (primitives + render_chat)
  generate.py           # NEW — sample()
  serve.py              # from scripts/api_server.py, onto shared modules
  cli.py                # NEW — unified subcommand dispatcher
  __main__.py           # NEW — python -m axim
  sft/
    __init__.py         # NEW
    data.py             # from scripts/sft_data.py (AximTokenizer subclasses Tokenizer)
    train.py            # from scripts/sft_train.py (imports rewritten)
  _nanochat/            # NEW — vendored
    __init__.py
    gpt.py, common.py, optim.py, flash_attention.py
    LICENSE
pyproject.toml         # NEW
configs/sft_example.json
data/example_sft.jsonl
webui/index.html        # unchanged
notebooks/axim_sft_colab.ipynb   # moved from root, rewritten
README.md               # rewritten
```

Removed at the end: `scripts/` (all of it), `requirements.txt`.

---

### Task 1: pyproject.toml + package skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `axim/__init__.py` (overwrite — currently a 9-line file, expand the public API)
- Create: `axim/sft/__init__.py`
- Create: `axim/_nanochat/__init__.py` (placeholder, filled in Task 2)
- Create: `notebooks/` directory (empty for now)

**Interfaces:**
- Produces: an installable package skeleton so `pip install -e .` works and `import axim` succeeds (core only at this point).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "axim"
version = "1.0.0"
description = "Package and inference engine for .axim model files"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "safetensors>=0.4",
    "tiktoken",
    "fastapi",
    "uvicorn",
    "numpy",
    "huggingface_hub",
    "datasets",
    "filelock",
]

[project.optional-dependencies]
sft = ["accelerate", "tqdm"]
flash = ["flash-attn"]

[project.scripts]
axim = "axim.cli:main"

[tool.setuptools.packages.find]
include = ["axim*"]
```

- [ ] **Step 2: Overwrite `axim/__init__.py`**

```python
"""
axim — Axiom Model Package Format + inference engine.

Public API:
    load_axim, write_axim, print_axim_info  — .axim format I/O
    Tokenizer                                — tiktoken wrapper + chat rendering
    load_model, load_from_dir                — build a nanochat GPT from .axim / safetensors
    sample                                   — autoregressive sampling loop
"""

from .core import load_axim, write_axim, print_axim_info, SectionType, AXIMHeader
from .tokenizer import Tokenizer
from .model import load_model, load_from_dir
from .generate import sample

__all__ = [
    "load_axim", "write_axim", "print_axim_info", "SectionType", "AXIMHeader",
    "Tokenizer", "load_model", "load_from_dir", "sample",
]
```

Note: this imports `tokenizer`, `model`, `generate` which don't exist yet. Step 4 creates them as stubs so the import succeeds; they're filled in Tasks 3-5. Alternatively, do this task's import check AFTER Tasks 3-5. To keep tasks independently verifiable, create stub modules now.

- [ ] **Step 3: Create stub modules so `import axim` works**

Create `axim/tokenizer.py`, `axim/model.py`, `axim/generate.py` each containing only a docstring + `pass`-level placeholder. These are filled in for real in Tasks 3, 4, 5. Example for `axim/tokenizer.py`:

```python
"""Tokenizer wrapper + chat rendering. Filled in Task 3."""
```

(`model.py` and `generate.py` similarly one-line placeholders.)

- [ ] **Step 4: Create `axim/sft/__init__.py`**

```python
"""axim.sft — supervised fine-tuning pipeline for .axim models."""
```

- [ ] **Step 5: Create `axim/_nanochat/__init__.py` (placeholder)**

```python
"""Vendored nanochat (from karpathy/nanochat, MIT). Filled in Task 2.

We own this fork and do not track upstream. See LICENSE in this directory.
"""
```

- [ ] **Step 6: Create `notebooks/` directory**

Run: `mkdir notebooks` (Bash) or `New-Item -ItemType Directory notebooks` (PowerShell). It will hold the moved notebook in Task 12.

- [ ] **Step 7: Install the package editable and verify import**

Run: `pip install -e .`
Expected: installs successfully, creates `axim.egg-info`.
Run: `python -c "import axim; print(axim.load_axim, axim.Tokenizer, axim.load_model, axim.sample)"`
Expected: prints four repr lines, no ImportError.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml axim/__init__.py axim/tokenizer.py axim/model.py axim/generate.py axim/sft/__init__.py axim/_nanochat/__init__.py
git commit -m "feat: add pyproject + package skeleton (installable axim)"
```

---

### Task 2: Vendor nanochat into axim/_nanochat

**Files:**
- Create: `axim/_nanochat/gpt.py`, `common.py`, `optim.py`, `flash_attention.py` (copied from `nanochat/nanochat/`)
- Create: `axim/_nanochat/LICENSE` (copied from `nanochat/LICENSE`)

**Interfaces:**
- Consumes: the existing vendored `nanochat/nanochat/{gpt,common,optim,flash_attention}.py`.
- Produces: `axim._nanochat` as a self-contained subpackage importable as `from axim._nanochat.gpt import GPT, GPTConfig`.

- [ ] **Step 1: Copy the four modules + LICENSE**

Run (Bash):
```bash
cp nanochat/nanochat/gpt.py axim/_nanochat/gpt.py
cp nanochat/nanochat/common.py axim/_nanochat/common.py
cp nanochat/nanochat/optim.py axim/_nanochat/optim.py
cp nanochat/nanochat/flash_attention.py axim/_nanochat/flash_attention.py
cp nanochat/LICENSE axim/_nanochat/LICENSE
```

- [ ] **Step 2: Rewrite `axim/_nanochat/gpt.py` imports**

In `axim/_nanochat/gpt.py`, replace:
```python
from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn
```
with:
```python
from axim._nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from axim._nanochat.optim import MuonAdamW, DistMuonAdamW
from axim._nanochat.flash_attention import flash_attn
```

- [ ] **Step 3: Rewrite `axim/_nanochat/optim.py` imports**

Replace:
```python
from nanochat.common import COMPUTE_DTYPE
```
with:
```python
from axim._nanochat.common import COMPUTE_DTYPE
```

- [ ] **Step 4: Rewrite `axim/_nanochat/flash_attention.py` imports**

Replace every `from nanochat.common import` (there is a lazy one around line 57) with `from axim._nanochat.common import`. Also replace the self-referential `from nanochat.flash_attention import flash_attn` (around line 8, inside a try/except) with `from axim._nanochat.flash_attention import flash_attn`. Use grep to find all occurrences first:

Run: `grep -n "nanochat" axim/_nanochat/flash_attention.py` to confirm all sites, then edit each to `axim._nanochat`.

- [ ] **Step 5: Confirm no `nanochat.` references remain in the vendored files**

Run: `grep -rn "nanochat" axim/_nanochat/`
Expected: only matches inside docstrings/comments or the `__init__.py` attribution text — no live `from nanochat` / `import nanochat` statements.

- [ ] **Step 6: Verify the vendored package imports**

Run: `python -c "from axim._nanochat.gpt import GPT, GPTConfig; from axim._nanochat.optim import MuonAdamW; from axim._nanochat.common import COMPUTE_DTYPE; print('ok')"`
Expected: `ok`. (This imports torch; if torch import is slow that's fine. No GPU needed — no model is constructed.)

- [ ] **Step 7: py_compile all vendored files**

Run: `python -m py_compile axim/_nanochat/gpt.py axim/_nanochat/common.py axim/_nanochat/optim.py axim/_nanochat/flash_attention.py && echo OK`
Expected: `OK`.

- [ ] **Step 8: Commit**

```bash
git add axim/_nanochat/
git commit -m "feat: vendor nanochat (gpt/common/optim/flash_attention) into axim._nanochat"
```

---

### Task 3: axim.tokenizer — single tokenizer wrapper

**Files:**
- Modify: `axim/tokenizer.py` (replace placeholder with real module)

**Interfaces:**
- Consumes: a tiktoken `Encoding` object (the `data["tokenizer"]` from `load_axim`).
- Produces: `class Tokenizer` with:
  - `Tokenizer(enc)` constructor
  - `.enc` attribute, `.bos` property → `int`
  - `encode_special(s: str) -> int`
  - `encode(text: str) -> list[int]`
  - `decode(ids: list[int]) -> str`
  - `render_chat(messages: list[dict], max_len: int = 2048) -> list[int]` — returns the chat scaffold ids (BOS + alternating `<|user_start|>...<|user_end|>` / `<|assistant_start|>...<|assistant_end|>` turns, system merged into first user). Does NOT append a trailing `<|assistant_start|>`; the caller does that to prime generation. This is the exact logic currently in `scripts/api_server.py` `_TokenizerWrap.render_chat` (lines 67-90 of the current file).

- [ ] **Step 1: Write `axim/tokenizer.py`**

```python
"""Tokenizer wrapper + chat rendering — the single source for tokenization.

Wraps the pickled tiktoken Encoding pulled from an .axim file and reimplements
the pieces nanochat's RustBPETokenizer provides that inference needs:
encode / decode / encode_special / bos / render_chat.
"""

import copy
from typing import List


class Tokenizer:
    """Minimal wrapper around a tiktoken Encoding.

    Used by `axim.serve`, `axim` infer, and subclassed by `axim.sft.data.AximTokenizer`
    (which adds the training-side render_conversation with assistant masking).
    """

    def __init__(self, enc):
        self.enc = enc
        self._bos = enc.encode_single_token("<|bos|>")

    @property
    def bos(self) -> int:
        return self._bos

    def encode_special(self, text: str) -> int:
        # encode_single_token raises KeyError if `text` is not a registered special token
        return self.enc.encode_single_token(text)

    def encode(self, text: str) -> List[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)

    def render_chat(self, messages, max_len: int = 2048) -> List[int]:
        """Render a chat conversation into the inference prompt scaffold.

        Produces: <|bos|> <|user_start|> ... <|user_end|> <|assistant_start|> ... <|assistant_end|> ...
        A leading system message is merged into the first user turn's content
        (nanochat convention). Does NOT append a trailing <|assistant_start|> —
        the caller appends it to prime generation.
        """
        ids: List[int] = []
        msgs = copy.deepcopy(messages)

        if msgs and msgs[0]["role"] == "system":
            msgs[1]["content"] = msgs[0]["content"] + "\n\n" + msgs[1]["content"]
            msgs = msgs[1:]

        us = self.encode_special("<|user_start|>")
        ue = self.encode_special("<|user_end|>")
        ast = self.encode_special("<|assistant_start|>")
        aen = self.encode_special("<|assistant_end|>")

        ids.append(self._bos)
        for i, msg in enumerate(msgs):
            want = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == want, f"expected {want!r} at index {i}, got {msg['role']!r}"
            if msg["role"] == "user":
                ids += [us] + self.encode(msg["content"]) + [ue]
            else:
                ids += [ast] + self.encode(msg["content"]) + [aen]

        return ids[:max_len]
```

- [ ] **Step 2: Verify import + a quick render_chat smoke test (no model, no GPU)**

Run:
```bash
python -c "
from axim.tokenizer import Tokenizer
import tiktoken
# can't easily build the real axim tokenizer without a model file, so just
# confirm the class imports and the method signatures exist.
t = Tokenizer.__new__(Tokenizer)
print('Tokenizer OK', hasattr(Tokenizer, 'render_chat'), hasattr(Tokenizer, 'encode_special'))
"
```
Expected: `Tokenizer OK True True`.

- [ ] **Step 3: py_compile**

Run: `python -m py_compile axim/tokenizer.py && echo OK`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add axim/tokenizer.py
git commit -m "feat: add axim.tokenizer.Tokenizer (single tokenizer wrapper)"
```

---

### Task 4: axim.model — single model-loading source

**Files:**
- Modify: `axim/model.py` (replace placeholder)

**Interfaces:**
- Consumes: `axim.core.load_axim`, `axim._nanochat.gpt.{GPT, GPTConfig}`, safetensors `load_file`.
- Produces:
  - `load_model(axim_path, device, dtype=None) -> (model, enc, cfg)` — `enc` is the raw tiktoken Encoding; `cfg` is the config dict. Model is `eval()` on `device`.
  - `load_from_dir(model_dir, device, dtype=None) -> (model, enc, cfg)` — same but from a directory of `model.safetensors` + `config.json` + `tokenizer.pkl` (the `test_inference --model-dir` path).
  - `build_gpt_config(cfg: dict) -> GPTConfig` — the shared `GPTConfig(...)` mapping currently duplicated 5×. Reads `vocab_size, num_hidden_layers, num_attention_heads, num_key_value_heads, hidden_size, max_position_embeddings, window_pattern` (window_pattern defaults to `[]` if absent, matching `test_inference`).

- [ ] **Step 1: Write `axim/model.py`**

```python
"""Model loading — the single home for the init_weights + load_state_dict dance.

`load_model` loads from an .axim file; `load_from_dir` loads from a directory of
raw safetensors + config + tokenizer.pkl. Both return (model, enc, cfg) with the
model already in eval() mode on the given device. `enc` is the raw tiktoken
Encoding — callers wrap it in axim.tokenizer.Tokenizer (or axim.sft.data.AximTokenizer
for training).
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .core import load_axim
from ._nanochat.gpt import GPT, GPTConfig


def build_gpt_config(cfg: Dict[str, Any]) -> GPTConfig:
    """Map an .axim / safetensors config dict to a nanochat GPTConfig."""
    return GPTConfig(
        vocab_size=cfg["vocab_size"],
        n_layer=cfg["num_hidden_layers"],
        n_head=cfg["num_attention_heads"],
        n_kv_head=cfg["num_key_value_heads"],
        n_embd=cfg["hidden_size"],
        sequence_len=cfg["max_position_embeddings"],
        window_pattern=cfg.get("window_pattern", []),
    )


def _instantiate(cfg: Dict[str, Any], weights: Dict[str, torch.Tensor],
                  device, dtype=None) -> torch.nn.Module:
    config = build_gpt_config(cfg)
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(weights, strict=True, assign=True)
    if dtype is not None:
        model = model.to(dtype)
    model = model.to(device)
    model.eval()
    return model


def load_model(axim_path, device, dtype=None) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """Load a model from an .axim file. Returns (model, enc, cfg)."""
    data = load_axim(axim_path)
    model = _instantiate(data["config"], data["weights"], device, dtype)
    return model, data["tokenizer"], data["config"]


def load_from_dir(model_dir, device, dtype=None) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """Load a model from a directory with model.safetensors + config.json + tokenizer.pkl.

    Looks for tokenizer.pkl in `model_dir` or its parent (matching the old
    test_inference behaviour).
    """
    from safetensors.torch import load_file

    model_dir = Path(model_dir)
    weights_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"

    tok_path = None
    for p in [model_dir / "tokenizer.pkl", model_dir.parent / "tokenizer.pkl"]:
        if p.exists():
            tok_path = p
            break

    if not weights_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    if not tok_path:
        raise FileNotFoundError(f"tokenizer.pkl not found near {model_dir}")

    weights = load_file(str(weights_path))
    with open(config_path, "r") as f:
        config = json.load(f)
    with open(tok_path, "rb") as f:
        enc = pickle.load(f)

    model = _instantiate(config, weights, device, dtype)
    return model, enc, config
```

- [ ] **Step 2: py_compile**

Run: `python -m py_compile axim/model.py && echo OK`
Expected: `OK`.

- [ ] **Step 3: Verify import wiring (no model load, no GPU)**

Run: `python -c "from axim.model import load_model, load_from_dir, build_gpt_config; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add axim/model.py
git commit -m "feat: add axim.model.load_model / load_from_dir (single load source)"
```

---

### Task 5: axim.generate — single sampling loop

**Files:**
- Modify: `axim/generate.py` (replace placeholder)

**Interfaces:**
- Consumes: a model with `.forward(ids)` (nanochat GPT), an `axim.tokenizer.Tokenizer` (needs `encode_special`, `decode`, `bos`).
- Produces: `sample(model, tok, prompt_ids, max_tokens, temperature, top_k, device, repetition_penalty=1.0, stop_strings=None, status=None) -> generator[int]` — yields token ids; populates `status["reason"]` ("stop"|"length") and `status["cause"]` (str). This is the exact logic currently in `scripts/api_server.py` `_generate` (the version with the status dict we added).

- [ ] **Step 1: Write `axim/generate.py`**

```python
"""Autoregressive sampling — the single sampling loop shared by serve + infer.

Stops on the model's own <|assistant_end|> / <|bos|> and additionally on any
user-supplied stop_strings (substring match on decoded-so-far; the stop string
itself is NOT emitted, OpenAI-style). HF-style repetition penalty, top-k, and
greedy vs sampling. If `status` (a dict) is passed, it is populated with the
finish reason after the loop exits.
"""

import torch
import torch.nn.functional as F


@torch.inference_mode()
def sample(model, tok, prompt_ids, max_tokens, temperature, top_k, device,
           repetition_penalty=1.0, stop_strings=None, status=None):
    """Yield generated token ids one at a time."""
    def _finish(reason, cause):
        if status is not None:
            status["reason"] = reason
            status["cause"] = cause

    stops = [s for s in (stop_strings or []) if s]
    aend = tok.encode_special("<|assistant_end|>")
    bos = tok.bos
    greedy = (temperature is None or temperature <= 0)

    rng = None
    if not greedy:
        rng = torch.Generator(device=device)
        rng.manual_seed(42)

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, T)
    generated = []
    stop_cause = None
    for _ in range(max_tokens):
        logits = model.forward(ids)[:, -1, :]  # (1, vocab)

        if repetition_penalty and repetition_penalty != 1.0 and generated:
            for t in set(generated):
                logits[0, t] = logits[0, t] / repetition_penalty if logits[0, t] > 0 \
                    else logits[0, t] * repetition_penalty

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        if greedy:
            next_id = int(torch.argmax(logits, dim=-1).item())
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1, generator=rng).item())

        if next_id == aend or next_id == bos:
            stop_cause = "eos"
            break
        if stops:
            decoded = tok.decode(generated + [next_id])
            hit = next((s for s in stops if s in decoded), None)
            if hit is not None:
                stop_cause = ("stop_string", hit)
                break

        generated.append(next_id)
        ids = torch.cat((ids, torch.tensor([[next_id]], dtype=torch.long, device=device)), dim=1)
        yield next_id

    if stop_cause is None:
        _finish("length", f"max_tokens={max_tokens} reached")
    elif stop_cause == "eos":
        _finish("stop", "<|assistant_end|> generated")
    else:
        _finish("stop", f"stop string {stop_cause[1]!r} matched")
```

- [ ] **Step 2: py_compile**

Run: `python -m py_compile axim/generate.py && echo OK`
Expected: `OK`.

- [ ] **Step 3: Verify the full public API now imports**

Run: `python -c "import axim; print(axim.load_model, axim.Tokenizer, axim.sample)"`
Expected: three repr lines, no ImportError.

- [ ] **Step 4: Commit**

```bash
git add axim/generate.py
git commit -m "feat: add axim.generate.sample (single sampling loop)"
```

---

### Task 6: axim.serve — port the API server onto shared modules

**Files:**
- Create: `axim/serve.py`
- (Reference: `scripts/api_server.py` for the endpoint/logging logic to port.)

**Interfaces:**
- Consumes: `axim.model.load_model`, `axim.tokenizer.Tokenizer`, `axim.generate.sample`, `axim.core`.
- Produces:
  - `build_app(model, tok, cfg, device) -> FastAPI` — the OpenAI-compatible app (routes `/v1/chat/completions`, `/v1/completions`, `/v1/debug/template`, `/v1/models`, `/health`) with the verbose per-request logging and SSE streaming currently in `scripts/api_server.py`.
  - `serve(axim_path, host, port, device)` — loads the model via `load_model`, wraps the tokenizer in `Tokenizer`, builds the app, runs uvicorn. This is what `axim serve` calls.

- [ ] **Step 1: Write `axim/serve.py`**

Port the current `scripts/api_server.py` with these changes:
1. Remove the `sys.path.insert` lines and the local `_load_model` / `_TokenizerWrap` / `_generate` / logging helpers — use `axim.model.load_model`, `axim.tokenizer.Tokenizer`, `axim.generate.sample` instead. Keep the `_log`, `_next_req_id`, `_fmt_params` logging helpers (they're serve-specific; move them into `axim/serve.py`).
2. `build_app(model, tok, cfg, device)` takes an already-constructed `Tokenizer` `tok` (was `_TokenizerWrap`). The endpoints call `axim.generate.sample(model, tok, prompt_ids, max_tok, t, k, device, repetition_penalty=rep_pen, stop_strings=stop, status=status)` instead of the local `_generate`.
3. Add `serve(axim_path, host, port, device)`:

```python
def serve(axim_path, host="0.0.0.0", port=None, device="cpu"):
    import uvicorn
    dev = torch.device(device)
    model, enc, cfg = load_model(axim_path, dev)
    tok = Tokenizer(enc)

    total = sum(p.numel() for p in model.parameters())
    print(f"loaded {total:,} params ({total/1e9:.2f}B)")
    print(f"device: {dev}")
    if port is None:
        port = _find_free_port()

    app = build_app(model, tok, cfg, dev)
    print(f"server: http://{host}:{port}")
    print("endpoints:")
    print("  POST /v1/chat/completions   (OpenAI-style chat, stream=true for SSE)")
    print("  POST /v1/completions        (raw prompt completion, stream=true for SSE)")
    print("  POST /v1/debug/template     (render chat template, no forward pass)")
    print("  GET  /v1/models  /health")
    print("verbose per-request logs below (one line per request ->/<-):")
    uvicorn.run(app, host=host, port=port)
```

Keep `_find_free_port` in `axim/serve.py` (copy from `api_server.py`). Keep the endpoint bodies and the verbose `_log`/`_fmt_params`/`_next_req_id` helpers verbatim from the current `api_server.py`, with the two substitutions above.

- [ ] **Step 2: py_compile**

Run: `python -m py_compile axim/serve.py && echo OK`
Expected: `OK`.

- [ ] **Step 3: Verify import + app construction (no model load, no GPU)**

Run:
```bash
python -c "
from axim.serve import build_app, serve
print('serve imports ok')
"
```
Expected: `serve imports ok`.

- [ ] **Step 4: Commit**

```bash
git add axim/serve.py
git commit -m "feat: add axim.serve (API server onto shared model/tokenizer/generate)"
```

---

### Task 7: axim.cli + __main__ — unified CLI

**Files:**
- Modify: `axim/cli.py` (create — currently the placeholder is in model/tokenizer/generate, not cli)
- Create: `axim/__main__.py`

**Interfaces:**
- Consumes: `axim.core.print_axim_info`/`write_axim`, `axim.serve.serve`, `axim.model.load_model`/`load_from_dir`, `axim.tokenizer.Tokenizer`, `axim.generate.sample`, and the export/download/prepare-data logic ported from `scripts/export_to_axim.py`, `scripts/download_model.py`, `scripts/prepare_sft_data.py` (these become functions in `axim/cli.py` or, for prepare-data, a function in `axim/sft/data.py` — see Task 10). For Task 7, port `export` and `download` as functions in `cli.py` directly (they're short and CLI-coupled); wire `prepare-data` and `sft` as thin dispatches that import from `axim.sft` (added in Tasks 9-10) — for now, have those two subcommands raise `SystemExit("not yet wired — see Task 9/10")` so the CLI parses but defers.
- Produces: `axim.cli.main()` — the single entry point for the `axim` console script and `python -m axim`. Subcommands: `serve`, `inspect`, `export`, `download`, `infer`, `prepare-data`, `sft`.

- [ ] **Step 1: Write `axim/cli.py`**

```python
"""Unified CLI for axim: `axim <command>` or `python -m axim <command>`."""

import argparse
import sys
from pathlib import Path


def _cmd_serve(args):
    from .serve import serve
    serve(args.axim, host=args.host, port=args.port, device=args.device)


def _cmd_inspect(args):
    from .core import print_axim_info
    print_axim_info(args.path)


def _cmd_export(args):
    import json
    from safetensors.torch import load_file
    from .core import write_axim

    model_dir = Path(args.model_dir)
    print(f"loading weights from {model_dir / 'model.safetensors'}")
    weights = load_file(str(model_dir / "model.safetensors"))
    print(f"loading config from {model_dir / 'config.json'}")
    with open(model_dir / "config.json", "r") as f:
        config = json.load(f)
    meta = None
    meta_path = model_dir / "nanochat_metadata.json"
    if meta_path.exists():
        print(f"loading metadata from {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)

    tok_path = None
    if args.tokenizer_pkl:
        tok_path = args.tokenizer_pkl
    elif args.repo_id:
        from huggingface_hub import hf_hub_download
        tok_path = hf_hub_download(repo_id=args.repo_id, filename="tokenizer.pkl")
    else:
        for p in [model_dir / "tokenizer.pkl", model_dir.parent / "tokenizer.pkl"]:
            if p.exists():
                tok_path = str(p)
                break
    if not tok_path:
        print("ERROR: no tokenizer.pkl found")
        sys.exit(1)
    print(f"loading tokenizer from {tok_path}")
    with open(tok_path, "rb") as f:
        tok_data = f.read()
    print(f"\nwriting {args.output}")
    write_axim(args.output, weights, config, tok_data, meta)
    print("done")


def _cmd_download(args):
    # Ported verbatim from scripts/download_model.py:main (HF Hub -> .axim).
    import os, json, shutil
    from huggingface_hub import list_repo_files, hf_hub_download, snapshot_download
    from safetensors.torch import load_file, save_file
    from .core import write_axim

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"listing files in {args.repo} ...")
    files = list_repo_files(args.repo)
    print("  " + "\n  ".join(files))
    axim_files = [f for f in files if f.endswith(".axim")]
    if axim_files:
        target = args.filename or axim_files[0]
        if target not in files:
            raise SystemExit(f"--filename {target!r} not found in repo; available .axim: {axim_files}")
        print(f"\nrepo has a ready .axim ({target}); downloading ...")
        dl = hf_hub_download(args.repo, target)
        real = os.path.realpath(dl)
        if not os.path.exists(real):
            raise SystemExit(f"downloaded file could not be resolved: {dl}")
        shutil.copy2(real, out)
        print(f"done -> {out} ({out.stat().st_size / 1e9:.2f} GB)")
        return
    print("\nno .axim in repo; staging safetensors/config/tokenizer ...")
    stage = Path(args.model_dir) if args.model_dir else out.parent / (out.stem + "_stage")
    stage.mkdir(parents=True, exist_ok=True)
    snapshot_download(args.repo, local_dir=str(stage))
    single = stage / "model.safetensors"
    idx = stage / "model.safetensors.index.json"
    if not single.exists() and idx.exists():
        print("  merging sharded safetensors -> model.safetensors ...")
        wmap = json.loads(idx.read_text())["weight_map"]
        merged, seen = {}, set()
        for shard in sorted(set(wmap.values())):
            if shard in seen:
                continue
            seen.add(shard)
            merged.update(load_file(str(stage / shard)))
        save_file(merged, str(single))
        del merged
    if not single.exists():
        raise SystemExit(f"no model.safetensors (and no shards) found in {stage}")
    print("  loading weights + config ...")
    weights = load_file(str(single))
    cfg = json.loads((stage / "config.json").read_text())
    meta = None
    mp = stage / "nanochat_metadata.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    tok_bytes = None
    local_tok = stage / "tokenizer.pkl"
    if local_tok.exists():
        tok_bytes = local_tok.read_bytes()
    else:
        print("  downloading tokenizer.pkl from repo ...")
        tok_path = hf_hub_download(args.repo, "tokenizer.pkl")
        tok_bytes = Path(tok_path).read_bytes()
    print(f"\npacking -> {out} ...")
    write_axim(str(out), weights, cfg, tok_bytes, meta)
    print("done.")
    if not args.keep_stage and args.model_dir is None and stage.exists():
        shutil.rmtree(stage, ignore_errors=True)


def _cmd_infer(args):
    import torch
    from .model import load_model, load_from_dir
    from .tokenizer import Tokenizer
    from .generate import sample

    device = torch.device(args.device)
    if args.axim:
        model, enc, cfg = load_model(args.axim, device)
    elif args.model_dir:
        model, enc, cfg = load_from_dir(args.model_dir, device)
    else:
        print("ERROR: specify --axim or --model-dir")
        sys.exit(1)

    tok = Tokenizer(enc)
    total = sum(p.numel() for p in model.parameters())
    print(f"loaded {total:,} params ({total/1e9:.2f}B) on {device}")

    if args.chat:
        msgs = []
        if args.system_prompt:
            msgs.append({"role": "system", "content": args.system_prompt})
        msgs.append({"role": "user", "content": args.prompt})
        prompt_ids = tok.render_chat(msgs, max_len=cfg["max_position_embeddings"])
        prompt_ids.append(tok.encode_special("<|assistant_start|>"))
        mode = "chat (stops on <|assistant_end|>)"
    else:
        prompt_ids = tok.encode(args.prompt)
        prompt_ids = [tok.bos] + prompt_ids if prompt_ids[0] != tok.bos else prompt_ids
        mode = "raw completion"

    print(f"prompt: {args.prompt!r}")
    print(f"mode: {mode}")
    print(f"generating up to {args.max_tokens} tokens (temp={args.temperature}, top_k={args.top_k}, rep_penalty={args.repetition_penalty})...\n")

    status = {}
    toks = list(sample(model, tok, prompt_ids, args.max_tokens, args.temperature,
                       args.top_k, device, repetition_penalty=args.repetition_penalty,
                       status=status))
    out = tok.decode(toks)
    print("=" * 50)
    print(args.prompt + out)
    print("=" * 50)


def _cmd_prepare_data(args):
    # Wired in Task 10.
    raise SystemExit("prepare-data: wired in Task 10 (axim.sft.data.prepare_dataset)")


def _cmd_sft(args):
    # Wired in Task 9.
    raise SystemExit("sft: wired in Task 9 (axim.sft.train)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="axim", description="axim — .axim model package + inference engine")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("serve", help="Serve an .axim model over an OpenAI-compatible API")
    sp.add_argument("--axim", required=True)
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=_cmd_serve)

    sp = sub.add_parser("inspect", help="Inspect an .axim file")
    sp.add_argument("path")
    sp.set_defaults(func=_cmd_inspect)

    sp = sub.add_parser("export", help="Export a model directory to .axim")
    sp.add_argument("--model-dir", required=True)
    sp.add_argument("--tokenizer-pkl", default=None)
    sp.add_argument("--repo-id", default=None)
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=_cmd_export)

    sp = sub.add_parser("download", help="Download a base model from the HF Hub to .axim")
    sp.add_argument("--repo", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--filename", default=None)
    sp.add_argument("--model-dir", default=None)
    sp.add_argument("--keep-stage", action="store_true")
    sp.set_defaults(func=_cmd_download)

    sp = sub.add_parser("infer", help="Run inference on a model")
    sp.add_argument("--axim", default=None)
    sp.add_argument("--model-dir", default=None)
    sp.add_argument("--prompt", default="Once upon a time")
    sp.add_argument("--max-tokens", type=int, default=50)
    sp.add_argument("--temperature", type=float, default=0.7)
    sp.add_argument("--top-k", type=int, default=50)
    sp.add_argument("--repetition-penalty", type=float, default=1.0)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--chat", action="store_true")
    sp.add_argument("--system-prompt", default=None)
    sp.set_defaults(func=_cmd_infer)

    sp = sub.add_parser("prepare-data", help="Prepare an SFT dataset JSONL from the HF Hub")
    sp.add_argument("--repo", required=True)
    sp.add_argument("--config", default=None)
    sp.add_argument("--split", default="train")
    sp.add_argument("--train-examples", type=int, default=2000)
    sp.add_argument("--val-examples", type=int, default=200)
    sp.add_argument("--output-train", required=True)
    sp.add_argument("--output-val", required=True)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--system-prompt", default=None)
    sp.set_defaults(func=_cmd_prepare_data)

    sp = sub.add_parser("sft", help="Fine-tune an .axim model (SFT)")
    sp.add_argument("--config", required=True)
    sp.set_defaults(func=_cmd_sft)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `axim/__main__.py`**

```python
from axim.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: py_compile + verify all subcommand help renders**

Run:
```bash
python -m py_compile axim/cli.py axim/__main__.py && \
axim --help && \
axim serve --help && \
axim inspect --help && \
axim export --help && \
axim download --help && \
axim infer --help && \
axim prepare-data --help && \
axim sft --help
```
Expected: each `--help` prints usage with no traceback. (`prepare-data` and `sft` parse fine; calling them without args raises the deferred `SystemExit` only when actually invoked, not on `--help`.)

- [ ] **Step 4: Verify `axim inspect` actually works (no GPU needed)**

Run: `axim inspect <path-to-any-existing-.axim>` — if a model file is unavailable, skip with note "skipped (no .axim handy)"; the `--help` check in Step 3 already exercises the argparse path. If a file exists, expected: prints the same section breakdown as the old `scripts/inspect_axim.py`.

- [ ] **Step 5: Commit**

```bash
git add axim/cli.py axim/__main__.py
git commit -m "feat: add unified axim CLI (serve/inspect/export/download/infer)"
```

---

### Task 8: axim.sft.data — move sft_data.py, dedup tokenizer primitives

**Files:**
- Create: `axim/sft/data.py` (from `scripts/sft_data.py`)
- (Reference: `scripts/sft_data.py` for `render_conversation`, `normalize_record`, `load_jsonl`, `prepare_examples`, `PackedBatcher`, `prepare_dataset`.)

**Interfaces:**
- Consumes: `axim.tokenizer.Tokenizer` (AximTokenizer subclasses it for the shared primitives).
- Produces:
  - `class AximTokenizer(Tokenizer)` — adds `render_conversation` (training scaffold with assistant mask), `render_for_inference`, `get_vocab_size`, `get_bos_token_id` (back-compat aliases). The primitive `encode`/`encode_special`/`decode`/`bos` come from `Tokenizer`.
  - `prepare_dataset(repo, config, split, train_examples, val_examples, output_train, output_val, seed, system_prompt)` — the body of `scripts/prepare_sft_data.py:main`, factored into a callable. (Also keep `to_messages`, `is_valid` from prepare_sft_data, or reuse `normalize_record` from sft_data — see Step 1 note.)
  - `normalize_record`, `load_jsonl`, `prepare_examples`, `PackedBatcher` — moved verbatim from `scripts/sft_data.py`.

- [ ] **Step 1: Create `axim/sft/data.py`**

Copy `scripts/sft_data.py` to `axim/sft/data.py`. Then make these edits:

a. Replace the imports block. The original starts with `import copy / import json / from typing import ... / import torch`. Add:
```python
from ..tokenizer import Tokenizer
```

b. Change `class AximTokenizer:` to subclass `Tokenizer` and drop the now-duplicate primitives:
```python
from ..tokenizer import Tokenizer

class AximTokenizer(Tokenizer):
    """Training-side tokenizer: axim.tokenizer.Tokenizer + render_conversation
    (assistant-loss masking) + render_for_inference. The encode/decode/
    encode_special/bos primitives are inherited from Tokenizer (single source)."""

    def __init__(self, enc, bos_token: str = "<|bos|>"):
        super().__init__(enc)  # sets self.enc, self._bos
        # bos_token is always <|bos|> in axim; kept for back-compat with existing calls.

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_bos_token_id(self) -> int:
        return self.bos
```
Delete the old `encode_special`, `encode`, `decode`, `__init__`'s `self.bos_token_id = ...` lines from AximTokenizer (they're now inherited). Keep `render_conversation` and `render_for_inference` verbatim — they use `self.encode_special`, `self.encode`, `self.encode("<|assistant_start|>")` which resolve to `Tokenizer`'s methods. If `render_conversation` references `self.bos_token_id`, change those to `self.bos`.

c. Move `normalize_record`, `load_jsonl`, `prepare_examples`, `PackedBatcher` verbatim.

d. Add `prepare_dataset(...)` — port the body of `scripts/prepare_sft_data.py:main` (after argparse) into a function. Reuse the existing `to_messages`/`is_valid` logic from prepare_sft_data (copy them in, or if `normalize_record` already covers the same shapes, use `normalize_record`). To preserve exact behavior, copy `to_messages` and `is_valid` from `scripts/prepare_sft_data.py` into `axim/sft/data.py` as module functions and have `prepare_dataset` call them. Signature:
```python
def prepare_dataset(repo, config=None, split="train", train_examples=2000,
                    val_examples=200, output_train=None, output_val=None,
                    seed=42, system_prompt=None):
    """Load a dataset slice from the HF Hub, normalize to {messages: [...]},
    filter, split, and write train/val JSONL. Ported from prepare_sft_data.py."""
    # ... body from prepare_sft_data.py:main lines 93-133 ...
```

- [ ] **Step 2: py_compile**

Run: `python -m py_compile axim/sft/data.py && echo OK`
Expected: `OK`.

- [ ] **Step 3: Verify import + that AximTokenizer inherits Tokenizer primitives**

Run:
```bash
python -c "
from axim.sft.data import AximTokenizer, prepare_dataset, normalize_record, PackedBatcher
from axim.tokenizer import Tokenizer
assert issubclass(AximTokenizer, Tokenizer)
print('sft.data ok; AximTokenizer subclasses Tokenizer')
"
```
Expected: `sft.data ok; AximTokenizer subclasses Tokenizer`.

- [ ] **Step 4: Commit**

```bash
git add axim/sft/data.py
git commit -m "feat: add axim.sft.data (sft_data + prepare_sft_data, AximTokenizer subclasses Tokenizer)"
```

---

### Task 9: axim.sft.train — move sft_train.py, wire `axim sft`

**Files:**
- Create: `axim/sft/train.py` (from `scripts/sft_train.py`)
- Modify: `axim/cli.py` (wire `_cmd_sft`)

**Interfaces:**
- Consumes: `axim.core.load_axim`/`write_axim`, `axim.sft.data`, `axim._nanochat.{gpt,common,optim}`.
- Produces: `axim.sft.train.main()` (the trainer entry) callable from `axim sft`.

- [ ] **Step 1: Copy `scripts/sft_train.py` to `axim/sft/train.py`**

Run: `cp scripts/sft_train.py axim/sft/train.py`

- [ ] **Step 2: Rewrite the imports / sys.path preamble in `axim/sft/train.py`**

The original begins:
```python
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))                       # for `axim`
sys.path.insert(0, str(REPO / "nanochat"))           # for `nanochat.gpt`
...
from axim.core import load_axim, write_axim
...
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import ...
from nanochat.optim import ...
```
Changes:
- Delete the `REPO = ...` and both `sys.path.insert(...)` lines.
- `from axim.core import load_axim, write_axim` → `from axim.core import load_axim, write_axim` (unchanged, works once installed).
- Every `from nanochat.X import ...` → `from axim._nanochat.X import ...`. Find them with `grep -n "from nanochat\|import nanochat" axim/sft/train.py` and edit each (there are several: `nanochat.common`, `nanochat.gpt`, `nanochat.optim`).
- Any `from sft_data import` → `from axim.sft.data import` (the old `test_inference` did `from sft_data import AximTokenizer`; check sft_train.py for the same and rewrite to `from axim.sft.data import AximTokenizer`).
- If sft_train.py constructs its own `GPTConfig` / `init_weights` / `load_state_dict` for the base model, leave it — but note Task 4's `load_model` exists; do NOT refactor sft_train to use it in this task (risk). Keep sft_train's loader as-is, just fix imports. (Conservative: behavior unchanged.)

- [ ] **Step 3: py_compile**

Run: `python -m py_compile axim/sft/train.py && echo OK`
Expected: `OK`. If it fails on an import, grep for the offending `nanochat`/`sft_data` reference and fix it.

- [ ] **Step 4: Verify `axim sft --help` shows the full flag set**

In `axim/cli.py`, replace the deferred `_cmd_sft`:
```python
def _cmd_sft(args):
    from .sft.train import main as sft_main
    # sft_train uses its own argparse; delegate argv after 'sft'
    sft_main()
```
But `sft_train.py`'s `main()` calls its own `_build_parser().parse_args()` on `sys.argv`. To make `axim sft --config ...` work, the cleanest approach: have `_cmd_sft` strip the leading `sft` from `sys.argv` and call `axim.sft.train.main()`. Replace `_cmd_sft` with:
```python
def _cmd_sft(args):
    import sys as _sys
    from .sft.train import main as sft_main
    # axim sft <flags> -> sft_train sees <flags> as its own argv
    _sys.argv = ["axim-sft"] + _sys.argv[2:]
    sft_main()
```
Then run: `axim sft --help`
Expected: prints the full sft_train flag set (the 30+ flags from `_build_parser`), no traceback.

- [ ] **Step 5: Commit**

```bash
git add axim/sft/train.py axim/cli.py
git commit -m "feat: add axim.sft.train (sft_train moved, imports -> axim._nanochat); wire `axim sft`"
```

---

### Task 10: wire `axim prepare-data`

**Files:**
- Modify: `axim/cli.py` (wire `_cmd_prepare_data`)

**Interfaces:**
- Consumes: `axim.sft.data.prepare_dataset` (created in Task 8).

- [ ] **Step 1: Wire `_cmd_prepare_data` in `axim/cli.py`**

Replace the deferred body:
```python
def _cmd_prepare_data(args):
    from .sft.data import prepare_dataset
    prepare_dataset(
        repo=args.repo, config=args.config, split=args.split,
        train_examples=args.train_examples, val_examples=args.val_examples,
        output_train=args.output_train, output_val=args.output_val,
        seed=args.seed, system_prompt=args.system_prompt,
    )
```

- [ ] **Step 2: py_compile + help check**

Run: `python -m py_compile axim/cli.py && axim prepare-data --help`
Expected: usage prints, no traceback.

- [ ] **Step 3: Commit**

```bash
git add axim/cli.py
git commit -m "feat: wire `axim prepare-data` -> axim.sft.data.prepare_dataset"
```

---

### Task 11: Delete `scripts/` and `requirements.txt`

**Files:**
- Delete: `scripts/` (entire directory)
- Delete: `requirements.txt`

**Interfaces:** none — this is the cutover. The CLI must already be fully wired (Tasks 7, 9, 10) before this runs.

- [ ] **Step 1: Confirm the CLI covers every old script**

Verify each old script's capability has a replacement:
- `api_server.py` → `axim serve` ✓ (Task 6/7)
- `inspect_axim.py` → `axim inspect` ✓ (Task 7)
- `export_to_axim.py` → `axim export` ✓ (Task 7)
- `download_model.py` → `axim download` ✓ (Task 7)
- `test_inference.py` → `axim infer` ✓ (Task 7)
- `prepare_sft_data.py` → `axim prepare-data` ✓ (Task 8/10)
- `sft_train.py` → `axim sft` ✓ (Task 9)
- `sft_data.py` → `axim.sft.data` ✓ (Task 8)

- [ ] **Step 2: Final smoke test of every subcommand's help**

Run:
```bash
for c in serve inspect export download infer prepare-data sft; do echo "=== $c ==="; axim $c --help || echo "FAILED $c"; done
```
Expected: all seven print usage, none print a traceback. Do NOT proceed to deletion if any fails.

- [ ] **Step 3: py_compile the whole package**

Run: `python -m py_compile axim/**/*.py axim/*.py && echo OK` (use a glob that works on your shell; PowerShell: `Get-ChildItem -Recurse axim -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }`)
Expected: `OK`, no errors.

- [ ] **Step 4: Delete `scripts/` and `requirements.txt`**

Run:
```bash
git rm -r scripts
git rm requirements.txt
```

- [ ] **Step 5: Reinstall + verify package still imports**

Run: `pip install -e . && python -c "import axim; from axim.sft import data, train; from axim import serve, cli; print('ok')"`
Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove scripts/ + requirements.txt (replaced by axim package + CLI)"
```

---

### Task 12: Move + rewrite the Colab notebook

**Files:**
- Move: `axim_sft_colab.ipynb` → `notebooks/axim_sft_colab.ipynb`
- (Reference: the existing notebook cells that call `python scripts/...`.)

**Interfaces:** none — documentation/orchestration only.

- [ ] **Step 1: Move the notebook**

Run: `git mv axim_sft_colab.ipynb notebooks/axim_sft_colab.ipynb`

- [ ] **Step 2: Rewrite cells to use the CLI**

Open `notebooks/axim_sft_colab.ipynb` and replace every cell that does `python scripts/X.py ...` with the matching `axim ...` invocation, and remove any cell that does `sys.path.insert` or clones nanochat. Concretely:
- Clone-this-repo cell: keep (clone this repo, not nanochat).
- Install cell: `pip install -e .` (or `pip install .`) — replaces any `pip install -r requirements.txt` / nanochat clone.
- Download cell: `axim download --repo Strawbercar/Axiom-V1-Base --output base.axim`
- Prepare-data cell: `axim prepare-data --repo HuggingFaceTB/smol-smoltalk --train-examples 50000 --val-examples 500 --output-train train.jsonl --output-val val.jsonl` (keep whatever dataset/repo the current notebook uses — read the existing cell and preserve its dataset choice).
- Train cell: `axim sft --config configs/sft_example.json --base-model base.axim --train train.jsonl --val val.jsonl --output tuned.axim --device cuda --dtype bf16` (match the existing notebook's flags).
- Serve cell (if any): `axim serve --axim tuned.axim --device cuda`
- Delete any `sys.path` manipulation cell and the nanochat-clone cell.

Because notebooks are JSON, edit cells via the NotebookEdit tool (read the notebook first with Read, then NotebookEdit each cell). Preserve non-command cells (markdown explanations) as-is, only updating commands.

- [ ] **Step 3: Validate notebook JSON**

Run: `python -c "import json; json.load(open('notebooks/axim_sft_colab.ipynb')); print('notebook json ok')"`
Expected: `notebook json ok`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/axim_sft_colab.ipynb
git commit -m "refactor: move notebook to notebooks/ and rewrite cells to use the axim CLI"
```

---

### Task 13: Rewrite README.md

**Files:**
- Modify: `README.md`

**Interfaces:** none.

- [ ] **Step 1: Rewrite README.md**

Replace the README content to reflect the new structure. Key sections (write full prose, not placeholders):

1. **Title + intro** — keep the existing voice ("Package and inference stuff for .axim model files. One binary, everything inside.").
2. **What is .axim? / Why not safetensors/GGUF/zip** — keep the existing subsections verbatim (they're still accurate).
3. **Format Spec (v1)** — keep verbatim (format unchanged).
4. **Installation** — replace with:
   ```bash
   git clone https://github.com/StrawberCar/Axiom-Inference-Engine.git
   cd Axiom-Inference-Engine
   pip install -e .
   ```
   Note: nanochat is vendored in `axim/_nanochat/` — no separate clone needed.
5. **Usage → CLI reference** — a table of the 7 subcommands with one-line descriptions and a example each:
   - `axim inspect <file.axim>`
   - `axim download --repo ... --output ...`
   - `axim export --model-dir ... --output ...`
   - `axim serve --axim ... [--device cuda]`
   - `axim infer --axim ... --prompt "..." [--chat] [--device cuda]`
   - `axim prepare-data --repo ... --output-train ... --output-val ...`
   - `axim sft --config ... --base-model ... --train ... --val ...`
6. **Usage → Python API**:
   ```python
   from axim import load_axim, load_model, Tokenizer, sample
   data = load_axim("model.axim")           # weights/config/tokenizer/metadata
   model, enc, cfg = load_model("model.axim", device="cuda")
   tok = Tokenizer(enc)
   ids = tok.render_chat([{"role": "user", "content": "hi"}])
   ```
7. **Run the API server** — `axim serve --axim model.axim --device cuda`, keep the verbose-log example, endpoints table, and curl/PowerShell examples from the current README.
8. **Web UI** — keep the current Simple/Advanced/streaming description; update the start command to `axim serve`.
9. **Fine-tune a model (SFT)** — keep the 3-step recipe but with `axim download` / `axim prepare-data` / `axim sft` commands and the `configs/sft_example.json` + `notebooks/axim_sft_colab.ipynb` references.
10. **File Structure** — the target layout from this plan's "File Structure (target)" section.
11. **Performance / Limitations / Roadmap** — keep the existing sections; the roadmap's "Streaming tokenizer" and "Better web UI with conversation history" stay checked.
12. **About** — keep the existing credits; add a line: "nanochat is vendored under `axim/_nanochat/` (MIT, (c) Andrej Karpathy) — we maintain our own fork."

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for the installable axim package + unified CLI"
```

---

### Task 14: Fix .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Edit `.gitignore`**

Remove these lines (no longer applicable):
- `nanochat/` (no longer an external clone)
- `.README.md` (stray)
- `*.ipynb` (the notebook is now a tracked artifact under `notebooks/`)

Add:
```
# packaging artifacts
*.egg-info/
dist/
build/
```

Keep: `__pycache__/`, `*.py[cod]`, `*.so`, `.claude`, `.remember`, `*.axim`, `out/`, `axiom_explainer.html`.

- [ ] **Step 2: Verify the notebook is still tracked**

Run: `git status --short && git ls-files notebooks/`
Expected: `notebooks/axim_sft_colab.ipynb` appears in `git ls-files`; `git status` shows no deletion of it.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: fix .gitignore (drop nanochat/ + stray entries, add packaging artifacts)"
```

---

### Task 15: Full smoke-test pass

**Files:** none (verification only).

- [ ] **Step 1: Fresh install + import everything**

Run: `pip install -e . && python -c "import axim; from axim import serve, cli; from axim.sft import data, train; from axim._nanochat import gpt, common, optim, flash_attention; print('all imports ok')"`
Expected: `all imports ok`.

- [ ] **Step 2: All seven subcommands' help**

Run: `for c in serve inspect export download infer prepare-data sft; do axim $c --help >/dev/null || echo "FAIL $c"; done; echo done`
Expected: `done` with no `FAIL` lines.

- [ ] **Step 3: py_compile every file**

Run: `python -m compileall -q axim && echo OK`
Expected: `OK`.

- [ ] **Step 4: inspect parity (if an .axim is available)**

If a model file exists: run `axim inspect model.axim` and confirm the output matches what `python scripts/inspect_axim.py model.axim` produced before deletion (same section breakdown). If no file, skip with note.

- [ ] **Step 5: GPU smoke (only if CUDA available)**

If `python -c "import torch; print(torch.cuda.is_available())"` prints `True`:
- `axim serve --axim <model> --device cuda --port 8123` in background, then `curl http://localhost:8080/health` (or the printed port) → `{"status":"ok"}`; then kill it.
- `axim infer --axim <model> --prompt "hello" --max-tokens 8 --device cuda` → prints generated text, no traceback.
If CUDA is not available, skip with note "skipped (no GPU) — per no-CPU-runs constraint".

- [ ] **Step 6: Note any skipped steps**

In the final commit message / summary, list which smoke steps were skipped and why (e.g. "inspect/infer smoke skipped — no .axim locally; help + py_compile + import all pass").

- [ ] **Step 7: Final commit if anything changed**

If Steps 1-5 made no changes, skip. Otherwise:
```bash
git add -A && git commit -m "test: full smoke pass after refactor"
```

---

## Self-Review (completed during planning)

**Spec coverage:** Every spec section maps to a task — package layout (T1-T10), dedup sources model/tokenizer/generate (T3-T5), unified CLI (T7, T9, T10), vendoring (T2), SFT restructure (T8-T10), packaging (T1, T11), notebook (T12), README (T13), gitignore (T14), out-of-scope respected (format untouched in T1-T14; no quantization tasks). ✓

**Type/signature consistency:** `load_model`/`load_from_dir` return `(model, enc, cfg)` everywhere they're used (T4, T6 serve, T7 infer). `Tokenizer` primitives (`encode`, `encode_special`, `decode`, `bos`) are defined once (T3) and inherited by `AximTokenizer` (T8). `sample(model, tok, prompt_ids, max_tokens, temperature, top_k, device, repetition_penalty, stop_strings, status)` is identical in T5 (definition), T6 (serve calls it), T7 (infer calls it). ✓

**Placeholder scan:** No "TBD"/"TODO". The only deferred items (`_cmd_sft`/`_cmd_prepare_data` in T7) are explicit `SystemExit`s that T9/T10 replace — intentional sequencing, not gaps. ✓

**Risk note carried into execution:** `axim infer --chat` now uses the `sample` loop (full forward passes) instead of nanochat's KV-cached `model.generate`. This is a deliberate behavior change toward consistency (serve and infer now use the same path; `--repetition-penalty` now actually applies in chat mode, which it silently didn't before). Flag this to the user in the execution summary.