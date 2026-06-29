# Installation

## Requirements

- **Python** >= 3.10
- A working Python environment with `pip`
- Optional but recommended: a CUDA-capable GPU for inference and training

The package is compatible with **Windows**, **Linux**, and **macOS**.

## Install from source

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/StrawberCar/Axiom-Inference-Engine.git
cd Axiom-Inference-Engine
pip install -e .
```

This creates the `axim` console script and makes `python -m axim` work.

!!! info "Editable install"
    `pip install -e .` is the recommended install path while the project is in active development. It links the package into your environment so source changes take effect immediately.

## Optional extras

Install extra dependencies when you need them:

```bash
# For the SFT pipeline (accelerate, tqdm)
pip install -e ".[sft]"

# For Flash Attention support
pip install -e ".[flash]"
```

!!! tip "Flash Attention"
    The `[flash]` extra pulls in `flash-attn`. This requires a compatible CUDA toolchain and may take a while to build. Only install it if your GPU supports the kernel.

## About the vendored nanochat

`axim/_nanochat/` is a vendored fork of Andrej Karpathy's nanochat (MIT license). It is included in the repository, so **no separate clone is required**. The package ships with everything needed to build, run, and fine-tune nanoChat-based models.

## Verify the install

```bash
axim --help
```

You should see the unified CLI and its subcommands: `serve`, `inspect`, `export`, `download`, `infer`, `prepare-data`, and `sft`.
