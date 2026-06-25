# Axiom-Inference-Engine

Package and inference stuff for **.axim** model files. One binary, everything inside. No zip bullshit, no folder juggling, no "where did i put the tokenizer" moments.

This repo is basically made for my [Axiom lineup of models](https://huggingface.co/Strawbercar). As of right now that's just Axiom-V1-Base but hey, gotta start somewhere.

---

## What is .axim?

.axim is a binary format that stuffs everything you need into one file:

- **weights** (safetensors format — fast, memory-mappable)
- **config** (json with architecture details)
- **tokenizer** (pickled tiktoken encoding)
- **metadata** (training info, whatever else)

It's not zip. It's not a tar. It's a proper binary with a header and section table so you can jump straight to whatever you need without unpacking anything.

### Why not just use safetensors?

Safetensors is great for weights but it's *just* weights. You still need the config file, the tokenizer file, the metadata file... before you know it you've got 5 files scattered across your downloads folder and you're questioning your life choices.

### Why not GGUF?

GGUF is cool but it's built for llama.cpp architectures. My models use nanoChat (Karpathy's thing) which has custom stuff like value embeddings, smear, backout, relu² activations — none of which llama.cpp understands. So GGUF was a non-starter.

### Why not just a zip file?

Zips are fine but they feel... cheap? Like you're just dumping files in a bag. .axim is a *format*. It has a spec. It has dignity.

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
pip install -r requirements.txt
```

You'll also need nanoChat installed since that's what runs the models:
```bash
git clone https://github.com/karpathy/nanochat.git
```

---

## Usage

### Inspect an .axim file

```bash
python scripts/inspect.py model.axim
```

Output:
```
AXIM: model.axim
  version: 1
  sections: 4
  size: 5.54 GB

  [weights] model.safetensors
    offset: 168, size: 5536.5 MB
  [config] config.json
    offset: 5536507584, size: 1.1 KB
  [tokenizer] tokenizer.pkl
    offset: 5536508616, size: 412.1 KB
  [metadata] metadata.json
    offset: 5536920728, size: 1.4 KB

  model: nanochat
    layers: 24
    heads: 12
    hidden: 1536
    vocab: 32768
    seq_len: 2048
    params: 1,384,122,122 (1.38B)
```

### Load in Python

```python
from axim.core import load_axim

data = load_axim("model.axim")
weights = data["weights"]      # dict[str, torch.Tensor]
config = data["config"]        # dict
tokenizer = data["tokenizer"]  # tiktoken Encoding
metadata = data["metadata"]    # dict
```

### Export a model to .axim

```bash
python scripts/export_to_axim.py \
    --model-dir ./model-safetensors \
    --tokenizer-pkl ./tokenizer.pkl \
    --output model.axim
```

Or download tokenizer from HuggingFace:
```bash
python scripts/export_to_axim.py \
    --model-dir ./model-safetensors \
    --repo-id Strawbercar/Axiom-V1-Base \
    --output model.axim
```

### Run the API server

```bash
python scripts/api_server.py --axim model.axim --device cuda
```

Then hit it with standard OpenAI-style requests:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "axim",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

Supports streaming too, just add `"stream": true`.

### Web UI

Open `webui/index.html` in a browser. It's a dead simple chat interface that talks to the API server. No build step, no npm, no webpack, just a single HTML file.

Point it at your API endpoint (default is `http://localhost:8000/v1/chat/completions`) and chat away.

---

## File Structure

```
Axiom-Inference-Engine/
├── axim/
│   ├── __init__.py          # package init
│   └── core.py              # read/write/inspect .axim files
├── scripts/
│   ├── export_to_axim.py    # export nanoChat models to .axim
│   ├── api_server.py        # OpenAI-compatible API server
│   ├── inspect.py           # quick .axim inspector
│   └── example_load.py      # example inference script
├── webui/
│   └── index.html           # simple chat web interface
├── requirements.txt         # python deps
└── README.md                # you're reading it
```

---

## Performance

Loading an .axim file is basically as fast as loading safetensors because the weights section *is* safetensors. The extra overhead is:

- Reading a ~200 byte header + section table
- Parsing a few KB of JSON
- Unpickling the tokenizer

So like, microseconds. The weights load via `safetensors.load()` which is already optimized.

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
- [ ] Streaming tokenizer (decode tokens as they arrive)
- [ ] Multi-model support in API server
- [ ] Better web UI with conversation history
- [ ] Model zoo / registry
- [ ] Digital signatures for .axim files

No promises though. This is a side project built on a shoestring budget.

---

## About

Made with around $100 by Strawbercar, released under the Axiom Research Project.

**Special thanks to:**
- **Andrej Karpathy** — for nanoChat and being an inspiration to every indie ML dev
- **HuggingFace** — for transformers, safetensors, and the hub
- **The llama.cpp team** — for proving that small, fast inference is possible
- **The tiktoken/rustbpe folks** — for tokenizers that actually work
- **Anyone who's ever released open source ML tools** — you know who you are

---

## License

MIT or whatever. Use it, fork it, improve it. Just don't be a dick about it.

If you build something cool with this, let me know. I'd love to see it.
