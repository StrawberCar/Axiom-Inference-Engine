"""
Example: load an .axim file and run inference.

Usage:
    python example_load.py model.axim
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from axim.core import load_axim

# need nanochat installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat"))
from nanochat.gpt import GPT, GPTConfig


def main():
    if len(sys.argv) < 2:
        print("usage: python example_load.py <file.axim>")
        return
    
    path = sys.argv[1]
    print(f"loading {path}...")
    
    data = load_axim(path)
    cfg = data["config"]
    
    print(f"model: {cfg.get('model_type', '?')}")
    print(f"params: {sum(v.numel() for v in data['weights'].values()):,}")
    
    config = GPTConfig(
        vocab_size=cfg["vocab_size"],
        n_layer=cfg["num_hidden_layers"],
        n_head=cfg["num_attention_heads"],
        n_kv_head=cfg["num_key_value_heads"],
        n_embd=cfg["hidden_size"],
        sequence_len=cfg["max_position_embeddings"],
        window_pattern=cfg["window_pattern"],
    )
    
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(data["weights"], strict=True, assign=True)
    model.eval()
    
    print("model loaded. do something with it.")


if __name__ == "__main__":
    main()
