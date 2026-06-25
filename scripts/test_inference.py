"""
Test inference on a model from raw safetensors files (no .axim needed).

Usage:
    python test_inference.py --model-dir ./model-safetensors --prompt "hello world" --max-tokens 50
    python test_inference.py --model-dir ./model-safetensors --device cuda --temperature 0.1 --repetition-penalty 1.2
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_from_dir(model_dir, device):
    from safetensors.torch import load_file
    
    model_dir = Path(model_dir)
    
    weights_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    
    # find tokenizer.pkl
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
    
    print(f"loading weights from {weights_path}")
    weights = load_file(str(weights_path))
    
    print(f"loading config from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"loading tokenizer from {tok_path}")
    with open(tok_path, "rb") as f:
        import pickle
        tokenizer = pickle.load(f)
    
    # need nanochat for model class
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat"))
    from nanochat.gpt import GPT, GPTConfig
    
    cfg = config
    gpt_config = GPTConfig(
        vocab_size=cfg["vocab_size"],
        n_layer=cfg["num_hidden_layers"],
        n_head=cfg["num_attention_heads"],
        n_kv_head=cfg["num_key_value_heads"],
        n_embd=cfg["hidden_size"],
        sequence_len=cfg["max_position_embeddings"],
        window_pattern=cfg.get("window_pattern", []),
    )
    
    model = GPT(gpt_config)
    model.init_weights()
    model.load_state_dict(weights, strict=True, assign=True)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, cfg


def _load_from_axim(axim_path, device):
    from axim.core import load_axim
    
    print(f"loading .axim from {axim_path}")
    data = load_axim(axim_path)
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat"))
    from nanochat.gpt import GPT, GPTConfig
    
    cfg = data["config"]
    config = GPTConfig(
        vocab_size=cfg["vocab_size"],
        n_layer=cfg["num_hidden_layers"],
        n_head=cfg["num_attention_heads"],
        n_kv_head=cfg["num_key_value_heads"],
        n_embd=cfg["hidden_size"],
        sequence_len=cfg["max_position_embeddings"],
        window_pattern=cfg.get("window_pattern", []),
    )
    
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(data["weights"], strict=True, assign=True)
    model = model.to(device)
    model.eval()
    
    return model, data["tokenizer"], cfg


@torch.inference_mode()
def generate(model, tokenizer, prompt_text, max_tokens=50, temperature=0.7, top_k=50, repetition_penalty=1.0, device="cpu"):
    # ALWAYS prepend BOS token — the model was trained with it at every document start
    bos = tokenizer.encode_single_token("<|bos|>")
    ids = [bos] + tokenizer.encode_ordinary(prompt_text)
    
    # Use the model's own generate() method — it handles smear, KV cache, etc. correctly
    generated = []
    for token in model.generate(ids, max_tokens=max_tokens, temperature=temperature, top_k=top_k):
        generated.append(token)
    
    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="test model inference")
    parser.add_argument("--model-dir", type=str, default=None, help="directory with model.safetensors + config.json")
    parser.add_argument("--axim", type=str, default=None, help=".axim file to load instead")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="prompt text")
    parser.add_argument("--max-tokens", type=int, default=50, help="max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="repetition penalty (>1.0 = less repetition)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()
    
    if not args.model_dir and not args.axim:
        print("ERROR: specify --model-dir or --axim")
        sys.exit(1)
    
    device = torch.device(args.device)
    
    # try loading from directory first, then axim fallback
    if args.model_dir:
        try:
            model, tokenizer, cfg = _load_from_dir(args.model_dir, device)
        except Exception as e:
            print(f"failed to load from directory: {e}")
            if args.axim:
                print("falling back to .axim file...")
                model, tokenizer, cfg = _load_from_axim(args.axim, device)
            else:
                raise
    else:
        model, tokenizer, cfg = _load_from_axim(args.axim, device)
    
    total = sum(p.numel() for p in model.parameters())
    print(f"\nloaded {total:,} params ({total/1e9:.2f}B)")
    print(f"device: {device}")
    print(f"prompt: {args.prompt!r}")
    print(f"generating {args.max_tokens} tokens (temp={args.temperature}, top_k={args.top_k}, rep_penalty={args.repetition_penalty})...\n")
    
    output = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k, args.repetition_penalty, device)
    
    print("=" * 50)
    print(args.prompt + output)
    print("=" * 50)


if __name__ == "__main__":
    main()
