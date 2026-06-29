# Roadmap and limitations

This page tracks what axim currently does and does not do.

## Current limitations

- **nanoChat-only** — The inference and SFT code is built around the nanoChat
  architecture (value embeddings, smear, backout, ReLU^2). Models using other
  backbones are not supported.
- **No KV cache** — The API server and sampling loop run a full forward pass for
  every token. Fine for small models on a GPU, slow on CPU or at long contexts.
- **No quantization** — Weights are stored at full precision (fp32/bf16). 4-bit or
  8-bit quantization is not implemented.
- **RustBPETokenizer-only** — The tokenizer must be the tiktoken/RustBPETokenizer
  used by nanoChat. Other tokenizers need adapter code.

## Roadmap

- [ ] KV cache for faster inference
- [ ] Quantization support (GGUF-style Q4_K_M, etc.)
- [x] Streaming tokenizer (decode tokens as they arrive)
- [ ] Multi-model support in the API server
- [ ] Model zoo / registry
- [ ] Digital signatures for `.axim` files
