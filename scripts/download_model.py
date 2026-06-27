"""
Download a base model from the HuggingFace Hub into a single .axim file.

Primary path: the repo already contains a ready .axim -> just download it.
Fallback:    the repo has model.safetensors (+ optional shards) + config.json +
             tokenizer.pkl -> merge shards if needed and pack into .axim.

Usage:
    python scripts/download_model.py --repo Strawbercar/Axiom-V1-Base --output base.axim
    python scripts/download_model.py --repo user/model --filename model.axim --output base.axim

Set HF_TOKEN (or run `huggingface-cli login`) for gated repos.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from huggingface_hub import list_repo_files, hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file

from axim.core import write_axim


def main():
    p = argparse.ArgumentParser(description="download a base model from the HF Hub -> .axim")
    p.add_argument("--repo", required=True, help="HF repo id, e.g. Strawbercar/Axiom-V1-Base")
    p.add_argument("--output", required=True, help="output .axim path")
    p.add_argument("--filename", default=None,
                   help="specific .axim filename in the repo (default: first *.axim found)")
    p.add_argument("--model-dir", default=None,
                   help="(fallback only) dir to stage safetensors; default: a temp dir next to --output")
    p.add_argument("--keep-stage", action="store_true",
                   help="keep the staged safetensors dir after packing (default: delete it)")
    args = p.parse_args()

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
        shutil.move(dl, out)
        print(f"done -> {out} ({out.stat().st_size / 1e9:.2f} GB)")
        return

    # ---- fallback: safetensors + config + tokenizer.pkl -> pack into .axim ----
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

    # tokenizer.pkl: prefer one staged alongside the weights, else fetch from the repo
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


if __name__ == "__main__":
    main()