"""Unified CLI for axim: `axim <command>` or `python -m axim <command>`."""

import argparse
import sys
from pathlib import Path


def _cmd_serve(args):
    from .serve import serve
    serve(args.axim, host=args.host, port=args.port, device=args.device)


def _cmd_inspect(args):
    from .core import print_axim_info
    print_axim_info(args.path)


def _cmd_export(args):
    import json
    from safetensors.torch import load_file
    from .core import write_axim

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
        from huggingface_hub import hf_hub_download
        tok_path = hf_hub_download(repo_id=args.repo_id, filename="tokenizer.pkl")
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


def _cmd_download(args):
    # Ported verbatim from scripts/download_model.py:main (HF Hub -> .axim).
    import os, json, shutil
    from huggingface_hub import list_repo_files, hf_hub_download, snapshot_download
    from safetensors.torch import load_file, save_file
    from .core import write_axim

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
        real = os.path.realpath(dl)
        if not os.path.exists(real):
            raise SystemExit(f"downloaded file could not be resolved: {dl}")
        shutil.copy2(real, out)
        print(f"done -> {out} ({out.stat().st_size / 1e9:.2f} GB)")
        return
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


def _cmd_infer(args):
    import torch
    from .model import load_model, load_from_dir
    from .tokenizer import Tokenizer
    from .generate import sample

    device = torch.device(args.device)
    if args.axim:
        model, enc, cfg = load_model(args.axim, device)
    elif args.model_dir:
        model, enc, cfg = load_from_dir(args.model_dir, device)
    else:
        print("ERROR: specify --axim or --model-dir")
        sys.exit(1)

    tok = Tokenizer(enc)
    total = sum(p.numel() for p in model.parameters())
    print(f"loaded {total:,} params ({total/1e9:.2f}B) on {device}")

    if args.chat:
        msgs = []
        if args.system_prompt:
            msgs.append({"role": "system", "content": args.system_prompt})
        msgs.append({"role": "user", "content": args.prompt})
        prompt_ids = tok.render_chat(msgs, max_len=cfg["max_position_embeddings"])
        prompt_ids.append(tok.encode_special("<|assistant_start|>"))
        mode = "chat (stops on <|assistant_end|>)"
    else:
        prompt_ids = tok.encode(args.prompt)
        prompt_ids = [tok.bos] + prompt_ids if prompt_ids[0] != tok.bos else prompt_ids
        mode = "raw completion"

    print(f"prompt: {args.prompt!r}")
    print(f"mode: {mode}")
    print(f"generating up to {args.max_tokens} tokens (temp={args.temperature}, top_k={args.top_k}, rep_penalty={args.repetition_penalty})...\n")

    status = {}
    toks = list(sample(model, tok, prompt_ids, args.max_tokens, args.temperature,
                       args.top_k, device, repetition_penalty=args.repetition_penalty,
                       status=status))
    out = tok.decode(toks)
    print("=" * 50)
    print(args.prompt + out)
    print("=" * 50)


def _cmd_prepare_data(args):
    from .sft.data import prepare_dataset
    prepare_dataset(
        repo=args.repo, config=args.config, split=args.split,
        train_examples=args.train_examples, val_examples=args.val_examples,
        output_train=args.output_train, output_val=args.output_val,
        seed=args.seed, system_prompt=args.system_prompt,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="axim", description="axim — .axim model package + inference engine")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("serve", help="Serve an .axim model over an OpenAI-compatible API")
    sp.add_argument("--axim", required=True)
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=_cmd_serve)

    sp = sub.add_parser("inspect", help="Inspect an .axim file")
    sp.add_argument("path")
    sp.set_defaults(func=_cmd_inspect)

    sp = sub.add_parser("export", help="Export a model directory to .axim")
    sp.add_argument("--model-dir", required=True)
    sp.add_argument("--tokenizer-pkl", default=None)
    sp.add_argument("--repo-id", default=None)
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=_cmd_export)

    sp = sub.add_parser("download", help="Download a base model from the HF Hub to .axim")
    sp.add_argument("--repo", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--filename", default=None)
    sp.add_argument("--model-dir", default=None)
    sp.add_argument("--keep-stage", action="store_true")
    sp.set_defaults(func=_cmd_download)

    sp = sub.add_parser("infer", help="Run inference on a model")
    sp.add_argument("--axim", default=None)
    sp.add_argument("--model-dir", default=None)
    sp.add_argument("--prompt", default="Once upon a time")
    sp.add_argument("--max-tokens", type=int, default=50)
    sp.add_argument("--temperature", type=float, default=0.7)
    sp.add_argument("--top-k", type=int, default=50)
    sp.add_argument("--repetition-penalty", type=float, default=1.0)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--chat", action="store_true")
    sp.add_argument("--system-prompt", default=None)
    sp.set_defaults(func=_cmd_infer)

    sp = sub.add_parser("prepare-data", help="Prepare an SFT dataset JSONL from the HF Hub")
    sp.add_argument("--repo", required=True)
    sp.add_argument("--config", default=None)
    sp.add_argument("--split", default="train")
    sp.add_argument("--train-examples", type=int, default=2000)
    sp.add_argument("--val-examples", type=int, default=200)
    sp.add_argument("--output-train", required=True)
    sp.add_argument("--output-val", required=True)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--system-prompt", default=None)
    sp.set_defaults(func=_cmd_prepare_data)

    sp = sub.add_parser("sft", help="Fine-tune an .axim model (SFT)")
    # `axim sft` is short-circuited in main() before this parser runs, handing
    # off to sft_train's own argparse (full flag set + --help). This subparser
    # exists only so `axim --help` lists `sft` with its one-line description.
    sp.set_defaults(func=None)

    return p


def main():
    import sys as _sys
    # `axim sft` delegates to sft_train's own argparse (full flag set, --help,
    # etc.). Short-circuit before the parent parser so `axim sft --help` prints
    # sft_train's flags, not just the parent subparser's --config.
    if len(_sys.argv) >= 2 and _sys.argv[1] == "sft":
        from .sft.train import main as sft_main
        _sys.argv = ["axim-sft"] + _sys.argv[2:]
        sft_main()
        return
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()