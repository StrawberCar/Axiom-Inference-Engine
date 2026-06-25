"""
Export a trained axiom model to .axim format.

Usage:
    python export_to_axim.py --model-dir ./model-safetensors --tokenizer-pkl tokenizer.pkl --output model.axim
    python export_to_axim.py --model-dir ./model-safetensors --repo-id user/model --output model.axim
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from safetensors.torch import load_file
from axim.core import write_axim


def main():
    parser = argparse.ArgumentParser(description="export to .axim")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--tokenizer-pkl", type=str, default=None)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
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
        try:
            from huggingface_hub import hf_hub_download
            tok_path = hf_hub_download(repo_id=args.repo_id, filename="tokenizer.pkl")
        except Exception as e:
            print(f"failed to download tokenizer: {e}")
            sys.exit(1)
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


if __name__ == "__main__":
    main()
