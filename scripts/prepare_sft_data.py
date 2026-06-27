"""
Prepare an SFT dataset from the HuggingFace Hub into train/val JSONL in the
nanoChat chat format.

Loads a dataset slice, normalizes any common record shape to {"messages": [...]},
filters to valid user/assistant conversations, shuffles, splits into train/val,
and writes two JSONL files ready for sft_train.py.

Usage:
    python scripts/prepare_sft_data.py --repo HuggingFaceH4/smoltalk \
        --config smol-magpie-ultra --train-examples 2000 --val-examples 200 \
        --output-train train.jsonl --output-val val.jsonl --seed 42
"""

import os
import sys
import json
import argparse
import random

from datasets import load_dataset


def to_messages(row):
    """Normalize one dataset row to {'messages': [...]} in nanoChat chat format.

    Recognized shapes (detected by keys):
      - messages              -> chat (passthrough)
      - conversations         -> ShareGPT (from/value -> role/content)
      - instruction+output    -> Alpaca (+optional input, system)
      - problem+solution      -> OpenMath (+optional reasoning prepended)
      - question+answer
      - prompt+completion
      - text                  -> single assistant turn with empty user
    """
    if row.get("messages") is not None:
        return {"messages": row["messages"]}
    if row.get("conversations") is not None:
        role_map = {"human": "user", "user": "user", "gpt": "assistant",
                    "assistant": "assistant", "system": "system"}
        msgs = [{"role": role_map.get(t["from"], t["from"]), "content": t["value"]}
                for t in row["conversations"]]
        return {"messages": msgs}
    if row.get("instruction") is not None and row.get("output") is not None:
        p = row["instruction"]
        if row.get("input"):
            p = (p or "") + "\n\n" + row["input"]
        msgs = []
        if row.get("system"):
            msgs.append({"role": "system", "content": row["system"]})
        msgs += [{"role": "user", "content": p}, {"role": "assistant", "content": row["output"]}]
        return {"messages": msgs}
    if row.get("problem") is not None and row.get("solution") is not None:
        ans = row["solution"]
        if row.get("reasoning"):
            ans = row["reasoning"] + "\n\n" + ans
        return {"messages": [{"role": "user", "content": row["problem"]},
                            {"role": "assistant", "content": ans}]}
    if row.get("question") is not None and row.get("answer") is not None:
        return {"messages": [{"role": "user", "content": row["question"]},
                            {"role": "assistant", "content": row["answer"]}]}
    if row.get("prompt") is not None and row.get("completion") is not None:
        return {"messages": [{"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["completion"]}]}
    if row.get("text") is not None:
        return {"messages": [{"role": "user", "content": ""},
                            {"role": "assistant", "content": row["text"]}]}
    raise ValueError(f"unrecognized row keys: {list(row.keys())}")


def is_valid(conv):
    msgs = conv["messages"]
    if not msgs:
        return False
    return (any(m["role"] == "user" for m in msgs)
            and any(m["role"] == "assistant" for m in msgs))


def main():
    p = argparse.ArgumentParser(description="prepare SFT dataset JSONL from the HF Hub")
    p.add_argument("--repo", required=True, help="HF dataset repo id")
    p.add_argument("--config", default=None, help="dataset config/subset name (empty = default)")
    p.add_argument("--split", default="train", help="source split (e.g. train, test). slice is appended automatically")
    p.add_argument("--train-examples", type=int, default=2000)
    p.add_argument("--val-examples", type=int, default=200)
    p.add_argument("--output-train", required=True)
    p.add_argument("--output-val", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--system-prompt", default=None,
                   help="prepend a system message to records that lack one")
    args = p.parse_args()

    n_total = args.train_examples + args.val_examples
    split = args.split if "[" in args.split else f"{args.split}[:{n_total}]"
    print(f"loading {args.repo} / {args.config or '(default)'} split={split} ...")
    ds_kwargs = {"path": args.repo, "split": split}
    if args.config:
        ds_kwargs["name"] = args.config
    ds = load_dataset(**ds_kwargs)
    print(f"  loaded {len(ds):,} candidate rows; converting ...")

    rows = []
    for r in ds:
        try:
            conv = to_messages(dict(r))
        except Exception:
            continue
        if args.system_prompt and (not conv["messages"] or conv["messages"][0]["role"] != "system"):
            conv = {"messages": [{"role": "system", "content": args.system_prompt}] + conv["messages"]}
        if is_valid(conv):
            rows.append(conv)
        if len(rows) >= n_total:
            break
    print(f"  {len(rows):,} valid conversations after normalization.")

    if len(rows) < n_total:
        print(f"  NOTE: requested {n_total} but only {len(rows)} valid; using what we have.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = min(args.val_examples, len(rows))
    val = rows[:n_val]
    train = rows[n_val:n_val + args.train_examples]

    for path, data in ((args.output_train, train), (args.output_val, val)):
        with open(path, "w", encoding="utf-8") as f:
            for m in data:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"  wrote {len(data):,} -> {path}")

    if train:
        print("\nexample train row:")
        print(json.dumps(train[0], ensure_ascii=False, indent=2)[:600])


if __name__ == "__main__":
    main()