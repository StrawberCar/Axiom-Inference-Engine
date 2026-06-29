# Why `.axim`?

There are plenty of ways to ship a language model. We picked `.axim` because the existing options all made us do extra work.

## vs. Safetensors

Safetensors is excellent. It is fast, safe, and the weights section inside an `.axim` file *is* a safetensors blob. But safetensors is **just weights**. You still need a `config.json` to know the architecture, a `tokenizer.pkl` to turn text into tokens, and maybe a metadata file with training provenance.

Before `.axim`, that meant juggling four or five files across downloads, uploads, and inference scripts. `.axim` keeps the safetensors speed and adds a tiny header so the rest of the model travels with it.

## vs. GGUF

GGUF is the de facto standard for llama.cpp-style inference, and it works well for models that fit llama.cpp's architectural assumptions.

nanoChat does not.

The models this engine targets use custom pieces — value embeddings, smear, backout, ReLU² activations — that llama.cpp does not understand. Converting to GGUF would mean either stripping those pieces out or fighting the format, neither of which is useful. `.axim` is built around the actual architecture instead of bending the architecture to fit a format.

## vs. Zip / Tar

A zip file could absolutely hold the same four blobs. It would even compress things.

But `.axim` is purpose-built: a fixed header, a typed section table, and aligned offsets mean you can inspect and load the file without a general-purpose archive parser. The weights section is exposed as raw safetensors data, so it can be memory-mapped directly. And honestly, a custom binary format is cooler.

## What `.axim` optimizes for

- **One file to move around** — download, copy, or serve from a single path.
- **Fast loading** — header + table is ~200 bytes, weights load through safetensors.
- **Honest architecture support** — no pretending a nanoChat model is something it is not.
- **Simple tooling** — a small Python module can read, write, and inspect the format in a few hundred lines.

It is not trying to replace every model format. It is the right container for the Axiom model line.
