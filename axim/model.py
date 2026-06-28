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