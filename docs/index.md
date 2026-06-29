# Axiom Inference Engine

**One binary. Everything inside.**

The Axiom Inference Engine packages, serves, and fine-tunes `.axim` models — self-contained model files that bundle weights, config, tokenizer, and metadata into a single file.

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[CLI reference](cli/index.md){ .md-button }

---

## What you get

<div class="grid cards" markdown>

- :material-package-variant: **The `.axim` format**

    A single binary file with safetensors weights, JSON config, pickled tokenizer, and JSON metadata. No scattered files. No missing pieces.

- :material-console: **Unified CLI**

    `axim download`, `axim infer`, `axim serve`, `axim export`, `axim prepare-data`, and `axim sft` all share the same dispatch pattern.

- :material-api: **OpenAI-compatible API**

    `axim serve` exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/v1/debug/template` with streaming SSE support.

- :material-web: **Browser Web UI**

    `webui/index.html` is a self-contained dev console that talks directly to the running API server — no build step required.

- :material-tune: **SFT pipeline**

    Full-model fine-tuning with best-fit packing, assistant-only loss masking, Muon or AdamW optimizers, and periodic checkpoints.

</div>

---

## Why axim?

Most model packaging formats handle one part of the problem well but leave you to duct-tape the rest together. Safetensors is fast and memory-mappable, but it is only weights. GGUF is purpose-built for llama.cpp. Zip files work, but they are not structured for model loading.

`.axim` keeps the speed and safety of safetensors for the weights section, while adding a small header and section table so the config, tokenizer, and metadata travel with the model. Load one file and you have everything needed for inference, serving, or fine-tuning.

---

## Quick links

- [Install the package](getting-started/installation.md)
- [Run your first inference](getting-started/quickstart.md)
- [Read the `.axim` format spec](format/spec.md)
- [Understand why `.axim` exists](format/why-axim.md)
