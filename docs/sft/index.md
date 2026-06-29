# SFT overview

The `axim sft` pipeline turns a base `.axim` model into a chat-tuned `.axim`
model. The output is a drop-in replacement for the base model: it can be served
with `axim serve`, inspected with `axim inspect`, loaded with `load_model`, or
used in the web UI.

## What the pipeline does

1. Load a base `.axim` (weights + config + tokenizer + metadata).
2. Normalize chat-formatted JSONL training data.
3. Render conversations to token ids with assistant-only loss masking.
4. Pack multiple conversations into fixed-length training rows (`bestfit` packing).
5. Fine-tune with MuonAdamW or AdamW and a warmup-stable-decay learning-rate
   schedule.
6. Periodically evaluate, generate samples, and write checkpoints.
7. Save the final merged model back to `.axim`.

## Quick path (notebook / Colab)

The notebook at `notebooks/axim_sft_colab.ipynb` runs the whole pipeline one cell
at a time on a single GPU:

```bash
# 1. Download a base model
axim download --repo Strawbercar/Axiom-V1-Base --output base.axim

# 2. Build train/val JSONL from a Hugging Face dataset
axim prepare-data --repo HuggingFaceTB/smol-smoltalk \
    --train-examples 50000 --val-examples 500 \
    --output-train train.jsonl --output-val val.jsonl

# 3. Fine-tune
axim sft --config configs/sft_example.json \
    --base-model base.axim --train train.jsonl --val val.jsonl \
    --output tuned.axim --device cuda --dtype bfloat16
```

## Sections

- [Preparing data](preparing-data.md) — dataset formats and `axim prepare-data`.
- [Training](training.md) — the `axim sft` command and training behavior.
- [Config reference](config-reference.md) — every field in `configs/sft_example.json`.
