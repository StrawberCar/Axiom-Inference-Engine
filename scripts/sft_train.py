"""
SFT (supervised fine-tuning) for models in the .axim format.

Loads a base model packaged as .axim (weights + config + tokenizer + metadata),
fine-tunes it on a JSONL chat dataset, and writes the result back to .axim -
so the fine-tuned model is a drop-in replacement for the base everywhere the
.axim tooling is used (api_server, test_inference, webui, ...).

Designed to be self-contained and easy to configure: sensible defaults, a
JSON/YAML config file for repeatable runs, and CLI flags that override the
config. Runs single-process out of the box (CPU/CUDA/MPS) and also supports
multi-GPU via torchrun.

Quick start:
    python scripts/sft_train.py --base-model model.axim --train data/chat.jsonl \\
        --output out/finetuned.axim --device cuda --epochs 3

Config file:
    python scripts/sft_train.py --config configs/sft_example.json

Data formats (one JSON object per line; auto-detected):
    {"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
    {"prompt": "...", "completion": "..."}
    {"instruction": "...", "input"?: "...", "output": "..."}
    {"text": "..."}                       # raw text -> single assistant turn
"""

import os
import sys
import gc
import json
import time
import math
import pickle
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

# ---- repo path setup (so `axim` and `nanochat` import cleanly) -------------
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))                       # for `axim`
sys.path.insert(0, str(REPO / "nanochat"))           # for `nanochat.gpt`

import torch
import torch.nn.functional as F

# axim.core has no dependency on nanochat's COMPUTE_DTYPE, so it's safe to import
# at module level (the helpers below use load_axim/write_axim).
from axim.core import load_axim, write_axim

# NOTE: nanochat.common computes COMPUTE_DTYPE at import time from the
# NANOCHAT_DTYPE env var / CUDA capability, so we must set the env var BEFORE
# importing anything from nanochat. We parse args first (pure stdlib) to get
# --dtype, set the env, then import nanochat.


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SFT fine-tuner for .axim models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O
    p.add_argument("--config", type=str, default=None, help="JSON/YAML config file; CLI flags override it")
    p.add_argument("--base-model", type=str, default=None, help="base .axim to fine-tune (or a dir with model.safetensors+config.json)")
    p.add_argument("--resume-from", type=str, default=None, help="resume from a .axim (weights only) or a checkpoint dir (weights+optim)")
    p.add_argument("--train", type=str, default=None, help="training JSONL")
    p.add_argument("--val", type=str, default=None, help="validation JSONL (optional; if absent, a slice of --train is held out)")
    p.add_argument("--val-frac", type=float, default=None, help="fraction of --train to hold out for val when --val is not given")
    p.add_argument("--output", type=str, default=None, help="output .axim path (final model)")
    p.add_argument("--out-dir", type=str, default=None, help="checkpoint dir (periodic + resume state). default: alongside --output")
    # Device / precision
    p.add_argument("--device", type=str, default=None, help="cuda|cpu|mps (empty = autodetect)")
    p.add_argument("--dtype", type=str, default=None, help="bfloat16|float16|float32 (empty = autodetect via nanochat)")
    p.add_argument("--compile", type=int, default=0, help="torch.compile the model (1=yes, 0=no). off by default for portability")
    p.add_argument("--cast-dtype", type=int, default=0, help="cast model params/buffers to COMPUTE_DTYPE after loading (1=yes, 0=no). Use this on GPUs that can't run the stored dtype, e.g. a free Colab T4 with bf16 weights -> fp16 compute. The output .axim stores the cast dtype")
    # Data
    p.add_argument("--max-seq-len", type=int, default=None, help="training context length (must be <= base model's sequence_len)")
    p.add_argument("--train-on", type=str, default=None, choices=["assistant", "all"], help="which tokens to supervise")
    p.add_argument("--pack", type=str, default=None, choices=["bestfit", "single"], help="packing strategy")
    p.add_argument("--system-prompt", type=str, default=None, help="default system prompt prepended to records that lack one")
    p.add_argument("--seed", type=int, default=None, help="rng seed for shuffling/splits")
    # Batch / horizon
    p.add_argument("--batch-size", type=int, default=None, help="micro-batch size (rows per forward)")
    p.add_argument("--grad-accum", type=int, default=None, help="gradient accumulation steps (total batch = batch_size * grad_accum * world)")
    p.add_argument("--epochs", type=float, default=None, help="number of passes over the training data (-1 = use --steps)")
    p.add_argument("--steps", type=int, default=None, help="max optimizer steps (-1 = use --epochs)")
    p.add_argument("--max-grad-norm", type=float, default=None, help="gradient clipping max norm (0 = disable)")
    # Optimizer
    p.add_argument("--optimizer", type=str, default=None, choices=["muon", "adamw"], help="muon = nanochat's MuonAdamW (best for this arch); adamw = plain torch AdamW")
    p.add_argument("--lr", type=float, default=None, help="base learning rate (for adamw; or overrides all muon LRs when set)")
    p.add_argument("--matrix-lr", type=float, default=None, help="muon: LR for matrix (transformer) params")
    p.add_argument("--embedding-lr", type=float, default=None, help="muon: LR for token embeddings")
    p.add_argument("--unembedding-lr", type=float, default=None, help="muon: LR for lm_head")
    p.add_argument("--scalar-lr", type=float, default=None, help="muon: LR for per-layer scalars")
    p.add_argument("--weight-decay", type=float, default=None, help="weight decay (matrices for muon; all for adamw)")
    p.add_argument("--emb-lr-mult", type=float, default=None, help="adamw: multiplier on embedding LR (relative to --lr)")
    # LR schedule
    p.add_argument("--init-lr-frac", type=float, default=None, help="LR at step 0 as a fraction of base LR (warmup start)")
    p.add_argument("--warmup-ratio", type=float, default=None, help="fraction of training devoted to linear LR warmup")
    p.add_argument("--warmdown-ratio", type=float, default=None, help="fraction of training devoted to linear LR warmdown")
    p.add_argument("--final-lr-frac", type=float, default=None, help="final LR as a fraction of base LR (warmdown floor)")
    # Eval / logging / checkpointing
    p.add_argument("--eval-every", type=int, default=None, help="evaluate val loss every N steps (-1 = disable)")
    p.add_argument("--eval-steps", type=int, default=None, help="number of val batches per eval")
    p.add_argument("--sample-every", type=int, default=None, help="generate sample completions every N steps (-1 = disable)")
    p.add_argument("--sample-prompts", type=int, default=None, help="number of val prompts to sample at each sample-every")
    p.add_argument("--sample-tokens", type=int, default=None, help="max tokens per sample generation")
    p.add_argument("--save-every", type=int, default=None, help="save a numbered .axim snapshot every N steps (-1 = off). Each is a full copy of the model - heavy")
    p.add_argument("--checkpoint-every", type=int, default=None, help="overwrite a resumable checkpoint.axim+checkpoint.pt every N steps (-1 = off). Required for --resume-from to restore optimizer state")
    p.add_argument("--save-optim", type=int, default=None, help="persist optimizer state with the final model for later resume (0=no, 1=yes). ~2x model size for AdamW; smaller for Muon")
    p.add_argument("--wandb", type=str, default=None, help="wandb run name ('dummy' or empty disables wandb)")
    p.add_argument("--log-every", type=int, default=None, help="console log every N steps")
    return p


# Hardcoded defaults: used when neither CLI nor config sets a value.
DEFAULTS: Dict[str, Any] = dict(
    base_model="model.axim",
    resume_from=None,
    train=None, val=None, val_frac=0.02, output="finetuned.axim", out_dir=None,
    device="", dtype="", compile=0, cast_dtype=0,
    max_seq_len=None, train_on="assistant", pack="bestfit", system_prompt=None, seed=42,
    batch_size=4, grad_accum=8, epochs=3, steps=-1, max_grad_norm=1.0,
    optimizer="muon", lr=None, matrix_lr=0.02, embedding_lr=0.3, unembedding_lr=0.004,
    scalar_lr=0.5, weight_decay=0.0, emb_lr_mult=1.0,
    init_lr_frac=0.5, warmup_ratio=0.03, warmdown_ratio=0.2, final_lr_frac=0.0,
    eval_every=200, eval_steps=20, sample_every=500, sample_prompts=3, sample_tokens=64,
    save_every=-1, checkpoint_every=-1, save_optim=0, wandb="dummy", log_every=1,
)


def load_config_file(path: str) -> Dict[str, Any]:
    """Load a JSON or YAML config file (YAML if PyYAML is installed)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text)
        except ImportError:
            raise SystemExit("YAML config requires PyYAML: pip install pyyaml (or use a .json config)")
    return json.loads(text)


def resolve(args, name: str):
    """Priority: explicit CLI value (not None) > config file > hardcoded default."""
    cli_val = getattr(args, name.replace("-", "_"), None)
    if cli_val is not None:
        return cli_val
    if args._config is not None and name in args._config:
        return args._config[name]
    return DEFAULTS.get(name)


def main():
    parser = _build_parser()
    args = parser.parse_args()
    args._config = load_config_file(args.config) if args.config else {}

    # Resolve every knob through the priority chain.
    C = {name: resolve(args, name) for name in DEFAULTS}

    # ---- set dtype env BEFORE importing nanochat ---------------------------
    if C["dtype"]:
        os.environ["NANOCHAT_DTYPE"] = C["dtype"]

    # ---- now import the heavy stuff ---------------------------------------
    from nanochat.common import (compute_init, compute_cleanup, print0,
                                 autodetect_device_type, COMPUTE_DTYPE,
                                 COMPUTE_DTYPE_REASON, is_ddp_initialized)
    from nanochat.gpt import GPT, GPTConfig
    from sft_data import (AximTokenizer, load_jsonl, prepare_examples,
                         PackedBatcher, normalize_record)

    # =====================================================================
    # Device / distributed setup
    # =====================================================================
    device_type = autodetect_device_type() if not C["device"] else C["device"]
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master = ddp_rank == 0
    print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
    if ddp:
        print0(f"DDP world size: {ddp_world_size}")
    pin = (device_type == "cuda")

    torch.manual_seed(C["seed"])

    # wandb
    use_wandb = C["wandb"] and C["wandb"] != "dummy" and master
    if use_wandb:
        import wandb
        wandb_run = wandb.init(project="axim-sft", name=C["wandb"], config=C)
    else:
        class _Dummy:
            def log(self, *a, **k): pass
            def finish(self): pass
        wandb_run = _Dummy()

    # =====================================================================
    # Load the base model
    # =====================================================================
    base_path = C["base_model"]
    print0(f"loading base model from {base_path}")
    cfg, weights, enc, base_meta = _load_base(base_path)
    tokenizer = AximTokenizer(enc, bos_token=cfg.get("bos_token", "<|bos|>"))

    # max_seq_len: default to the base model's, but never exceed it
    base_seq_len = cfg["max_position_embeddings"]
    max_seq_len = C["max_seq_len"] or base_seq_len
    if max_seq_len > base_seq_len:
        print0(f"NOTE: requested max_seq_len={max_seq_len} > base sequence_len={base_seq_len}; clamping")
        max_seq_len = base_seq_len

    gpt_config = GPTConfig(
        vocab_size=cfg["vocab_size"],
        n_layer=cfg["num_hidden_layers"],
        n_head=cfg["num_attention_heads"],
        n_kv_head=cfg["num_key_value_heads"],
        n_embd=cfg["hidden_size"],
        sequence_len=base_seq_len,
        window_pattern=cfg.get("window_pattern", "SSSL"),
    )
    orig_model = GPT(gpt_config)
    orig_model.init_weights()
    orig_model.load_state_dict(weights, strict=True, assign=True)
    orig_model = orig_model.to(device)
    n_params = sum(p.numel() for p in orig_model.parameters())
    print0(f"loaded {n_params:,} params ({n_params/1e9:.2f}B) | n_layer={gpt_config.n_layer} | seq_len={max_seq_len}")

    # ---------------------------------------------------------------------
    # Resume: optionally restore weights (+ optimizer state) from a checkpoint
    # ---------------------------------------------------------------------
    start_step = 0
    if C["resume_from"]:
        start_step = _resume(orig_model, C["resume_from"], device, master)

    # Optionally cast params/buffers to the compute dtype. Needed when the
    # stored dtype can't run on this GPU (e.g. bf16 weights on a free Colab T4,
    # which is SM75 and has no bf16 compute). Applied AFTER load + resume so
    # resumed weights are also cast.
    if C["cast_dtype"]:
        orig_model = orig_model.to(COMPUTE_DTYPE)
        print0(f"cast model params/buffers to {COMPUTE_DTYPE}")

    model = torch.compile(orig_model, dynamic=False) if C["compile"] else orig_model
    model.train()

    # =====================================================================
    # Optimizer
    # =====================================================================
    optimizer, opt_kind = _build_optimizer(orig_model, C, ddp)

    # =====================================================================
    # Data
    # =====================================================================
    train_rows = load_jsonl(C["train"])
    print0(f"loaded {len(train_rows):,} training rows from {C['train']}")
    if C["val"]:
        val_rows = load_jsonl(C["val"])
        print0(f"loaded {len(val_rows):,} val rows from {C['val']}")
    else:
        # hold out a deterministic slice of the training data
        import random
        rng = random.Random(C["seed"])
        rng.shuffle(train_rows)
        n_val = max(1, int(len(train_rows) * C["val_frac"])) if len(train_rows) > 1 else 0
        val_rows = train_rows[:n_val]
        train_rows = train_rows[n_val:]
        print0(f"held out {len(val_rows):,} rows for validation (val_frac={C['val_frac']})")

    train_ex = prepare_examples(train_rows, tokenizer, max_seq_len, C["train_on"] == "all", C["system_prompt"])
    val_ex = prepare_examples(val_rows, tokenizer, max_seq_len, C["train_on"] == "all", C["system_prompt"])
    print0(f"rendered {len(train_ex):,} train / {len(val_ex):,} val conversations (train_on={C['train_on']}, pack={C['pack']})")
    if not train_ex:
        raise SystemExit("no usable training examples after rendering - check your data/format")

    # In DDP, shard the training conversations across ranks so each rank sees a
    # distinct slice (true data-parallel). Validation is evaluated per-rank.
    if ddp:
        train_ex = train_ex[ddp_rank::ddp_world_size]
        print0(f"DDP: rank {ddp_rank} sharded to {len(train_ex):,} train conversations")

    train_batcher = PackedBatcher(train_ex, C["batch_size"], max_seq_len, C["pack"],
                                  tokenizer.get_bos_token_id(), shuffle=True, seed=C["seed"])

    # Total horizon
    tokens_per_micro = C["batch_size"] * max_seq_len
    world_tokens_per_micro = tokens_per_micro * ddp_world_size
    total_batch = world_tokens_per_micro * C["grad_accum"]
    per_rank_per_step = tokens_per_micro * C["grad_accum"]
    est_tokens_per_epoch = sum(len(ids) for ids, _ in train_ex)  # tokens this rank sees per epoch
    steps_per_epoch = max(1, est_tokens_per_epoch // per_rank_per_step) if est_tokens_per_epoch else 1
    if C["steps"] and C["steps"] > 0:
        max_steps = C["steps"]
    else:
        max_steps = max(1, int(math.ceil(C["epochs"] * steps_per_epoch)))
    print0(f"micro-batch: {C['batch_size']} x {max_seq_len} = {tokens_per_micro:,} tokens | grad_accum={C['grad_accum']} "
           f"| total batch={total_batch:,} tokens | max_steps={max_steps} (~{steps_per_epoch}/epoch)")

    # =====================================================================
    # LR schedule  (see get_lr_multiplier below - the key knob to tune)
    # =====================================================================
    def get_muon_momentum(it: int) -> float:
        frac = min(it / 300, 1.0)
        return (1 - frac) * 0.85 + frac * 0.95

    # =====================================================================
    # Training loop
    # =====================================================================
    out_dir = Path(C["out_dir"]) if C["out_dir"] else Path(C["output"]).resolve().parent / "sft_checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_axim = out_dir / "checkpoint.axim"
    ckpt_optim = out_dir / "checkpoint.pt"

    scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
    if scaler is not None:
        print0("GradScaler enabled for fp16 training")

    train_iter = train_batcher.batches(device, pin_memory=pin, repeat=True)
    smooth_loss = 0.0
    ema_beta = 0.9
    best_val = float("inf")
    total_time = 0.0
    step = start_step

    print0("starting training...")
    # Baseline val loss BEFORE any updates, so you can see whether training actually
    # improves it (val_loss is the real metric - the per-step "loss" is noisy and
    # structurally inflated by being measured pre-step during active optimization).
    if val_ex:
        base_val = _eval_loss(orig_model, val_ex, C, tokenizer, device, ddp_world_size)
        best_val = base_val
        print0(f"baseline val_loss {base_val:.4f} | ppl {math.exp(min(20.0, base_val)):.2f} "
               f"(this is the base model's held-out loss; watch val_loss go DOWN from here)")
    while step < max_steps:
        t0 = time.time()
        optimizer_loss = 0.0
        for _ in range(C["grad_accum"]):
            x, y = next(train_iter)
            loss = model(x, y)
            optimizer_loss += loss.detach().item()
            loss = loss / C["grad_accum"]
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        optimizer_loss /= C["grad_accum"]

        # LR schedule (progress in [0,1])
        progress = (step - start_step) / max(1, (max_steps - start_step))
        lrm = get_lr_multiplier(progress, C)
        if opt_kind == "muon":
            muon_mom = get_muon_momentum(step)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group.get("kind") == "muon":
                    group["momentum"] = muon_mom
        else:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm

        if C["max_grad_norm"] and C["max_grad_norm"] > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(orig_model.parameters(), C["max_grad_norm"])

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        model.zero_grad(set_to_none=True)

        if device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        step += 1
        if step > 10:
            total_time += dt

        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * optimizer_loss
        debiased = smooth_loss / (1 - ema_beta ** step)

        if step % C["log_every"] == 0 or step == max_steps:
            tok_per_sec = int(total_batch / dt) if dt > 0 else 0
            print0(f"step {step:05d} ({100*progress:.1f}%) | loss {debiased:.4f} | lrm {lrm:.3f} "
                   f"| dt {dt*1000:.0f}ms | tok/s {tok_per_sec:,} | lr {optimizer.param_groups[0]['lr']:.2e}")

        wandb_run.log({"step": step, "train/loss": debiased, "train/lrm": lrm, "train/dt": dt})

        # ---- periodic eval ----
        if C["eval_every"] > 0 and (step % C["eval_every"] == 0 or step == max_steps) and val_ex:
            val_loss = _eval_loss(orig_model, val_ex, C, tokenizer, device, ddp_world_size)
            ppl = math.exp(min(20.0, val_loss)) if val_loss < 20 else float("inf")
            print0(f"step {step:05d} | val_loss {val_loss:.4f} | ppl {ppl:.2f}")
            wandb_run.log({"step": step, "val/loss": val_loss, "val/ppl": ppl})
            if val_loss < best_val:
                best_val = val_loss

        # ---- periodic generation samples ----
        if C["sample_every"] > 0 and (step % C["sample_every"] == 0) and val_rows:
            _sample(orig_model, tokenizer, val_rows, C, device, max_seq_len)

        # ---- periodic numbered snapshot (heavy: full model copy) ----
        if C["save_every"] > 0 and step % C["save_every"] == 0:
            tagged = out_dir / f"step{step}.axim"
            _save_axim(tagged, orig_model, cfg, enc, base_meta, step, None, C, master=master)
            print0(f"  saved snapshot -> {tagged}")

        # ---- periodic resumable checkpoint (overwritten; needed for --resume-from) ----
        if C["checkpoint_every"] > 0 and step % C["checkpoint_every"] == 0:
            _save_axim(ckpt_axim, orig_model, cfg, enc, base_meta, step, None, C, master=master)
            _save_optim(ckpt_optim, optimizer, step, master=master)
            print0(f"  saved resumable checkpoint -> {ckpt_axim}")

        if step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()

    # =====================================================================
    # Final save
    # =====================================================================
    final_val = float("inf")
    if val_ex:
        final_val = _eval_loss(orig_model, val_ex, C, tokenizer, device, ddp_world_size)
        print0(f"final val_loss {final_val:.4f} | best {best_val:.4f}")
    out_path = Path(C["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_axim(out_path, orig_model, cfg, enc, base_meta, step, final_val, C, master=master)
    if master and C["save_optim"]:
        # persist optimizer state so the run can be resumed from `out_dir` later
        _save_optim(ckpt_optim, optimizer, step, master=master)
        _save_axim(ckpt_axim, orig_model, cfg, enc, base_meta, step, final_val, C, master=master)
        print0(f"  wrote resumable checkpoint alongside final model -> {out_dir}")
    print0(f"done. wrote {out_path} (step {step}, val_loss {final_val if val_ex else 'n/a'})")

    wandb_run.finish()
    compute_cleanup()


# =============================================================================
# Helpers
# =============================================================================

def _load_base(base_path: str):
    """Load (config, weights, tokenizer_enc, metadata) from a .axim or a safetensors dir."""
    p = Path(base_path)
    if p.is_dir():
        from safetensors.torch import load_file
        weights = load_file(str(p / "model.safetensors"))
        with open(p / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(p / "tokenizer.pkl", "rb") as f:
            enc = pickle.load(f)
        meta = {}
        m = p / "nanochat_metadata.json"
        if m.exists():
            with open(m, "r", encoding="utf-8") as f:
                meta = json.load(f)
        return cfg, weights, enc, meta
    # .axim
    data = load_axim(p)
    return data["config"], data["weights"], data["tokenizer"], data.get("metadata", {})


def _resume(model, resume_from: str, device, master: bool) -> int:
    """Load weights from a .axim or a checkpoint dir. Returns the start step."""
    from nanochat.common import print0
    p = Path(resume_from)
    if p.is_dir():
        axim_path = p / "checkpoint.axim"
        optim_path = p / "checkpoint.pt"
        if not axim_path.exists():
            # fall back to a numbered step*.axim if present
            cands = sorted(p.glob("step*.axim"))
            if not cands:
                raise SystemExit(f"no checkpoint found in {p}")
            axim_path = cands[-1]
        print0(f"resuming weights from {axim_path}")
        data = load_axim(axim_path)
        model.load_state_dict(data["weights"], strict=True, assign=True)
        step = 0
        if optim_path.exists():
            print0(f"resuming optimizer state from {optim_path}")
            state = torch.load(optim_path, map_location="cpu")
            step = state.get("step", 0)
            # optimizer state is rebuilt by _build_optimizer after this; caller must load it
            model._resume_optim_state = state.get("optim")
        return step
    # single .axim
    print0(f"resuming weights from {p}")
    data = load_axim(p)
    model.load_state_dict(data["weights"], strict=True, assign=True)
    return 0


def _build_optimizer(model, C: Dict[str, Any], ddp: bool):
    """Build the optimizer. Returns (optimizer, kind). kind in {'muon','adamw'}."""
    from nanochat.common import print0
    kind = C["optimizer"]
    # On CPU/MPS, Muon's torch.compile kernels may be unavailable -> fall back to adamw.
    if kind == "muon":
        try:
            opt = model.setup_optimizer(
                unembedding_lr=C["unembedding_lr"], embedding_lr=C["embedding_lr"],
                matrix_lr=C["matrix_lr"], weight_decay=C["weight_decay"], scalar_lr=C["scalar_lr"],
            )
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"] * C["init_lr_frac"]
                group["lr"] = group["initial_lr"]
            print0(f"optimizer: MuonAdamW (matrix_lr={C['matrix_lr']}, embedding_lr={C['embedding_lr']}, "
                   f"unembedding_lr={C['unembedding_lr']}, wd={C['weight_decay']})")
            # load resumed optimizer state if present
            state = getattr(model, "_resume_optim_state", None)
            if state is not None:
                try:
                    opt.load_state_dict(state)
                    print0("  restored optimizer momentum buffers from checkpoint")
                except Exception as e:
                    print0(f"  WARNING: could not restore optimizer state: {e}")
                if hasattr(model, "_resume_optim_state"):
                    del model._resume_optim_state
            return opt, "muon"
        except Exception as e:
            print0(f"WARNING: Muon optimizer unavailable ({e}); falling back to AdamW")
            kind = "adamw"
    # plain AdamW with two param groups (embeddings vs the rest)
    lr = C["lr"] if C["lr"] is not None else 3e-4
    emb_params, other_params = [], []
    for name, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        if "wte" in name or "value_embeds" in name:
            emb_params.append(p_)
        else:
            other_params.append(p_)
    groups = [
        {"params": other_params, "lr": lr, "weight_decay": C["weight_decay"]},
        {"params": emb_params, "lr": lr * C["emb_lr_mult"], "weight_decay": C["weight_decay"]},
    ]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"] * C["init_lr_frac"]
        group["lr"] = group["initial_lr"]
    print0(f"optimizer: AdamW (lr={lr}, emb_lr={lr*C['emb_lr_mult']:.2e}, wd={C['weight_decay']})")
    state = getattr(model, "_resume_optim_state", None)
    if state is not None:
        try:
            opt.load_state_dict(state)
            print0("  restored optimizer state from checkpoint")
        except Exception as e:
            print0(f"  WARNING: could not restore optimizer state: {e}")
        if hasattr(model, "_resume_optim_state"):
            del model._resume_optim_state
    return opt, "adamw"


@torch.no_grad()
def _eval_loss(model, val_ex, C, tokenizer, device, world_size):
    """Mean cross-entropy over supervised val positions (ignores -1)."""
    from sft_data import PackedBatcher
    model.eval()
    batcher = PackedBatcher(val_ex, C["batch_size"], C["max_seq_len"] or 2048, C["pack"],
                            tokenizer.get_bos_token_id(), shuffle=False)
    total_nats = torch.zeros((), device=device)
    total_count = torch.zeros((), device=device)
    n = 0
    for x, y in batcher.batches(device):
        loss2d = model(x, y, loss_reduction="none").view(-1)
        yv = y.view(-1)
        valid = yv >= 0
        total_nats += (loss2d[valid]).sum()
        total_count += valid.sum()
        n += 1
        if n >= C["eval_steps"]:
            break
    if total_count.item() == 0:
        model.train()
        return float("inf")
    val = (total_nats / total_count).item()
    model.train()
    return val


@torch.inference_mode()
def _sample(model, tokenizer, val_rows, C, device, max_seq_len):
    """Generate a few completions from val conversations to eyeball quality."""
    from sft_data import normalize_record
    from nanochat.common import print0
    model.eval()
    import random
    rng = random.Random(123)
    prompts = rng.sample(val_rows, min(C["sample_prompts"], len(val_rows)))
    for row in prompts:
        try:
            conv = normalize_record(row, default_system=C["system_prompt"])
            prompt_ids = tokenizer.render_for_inference(conv, max_tokens=max_seq_len)
        except Exception:
            continue
        if len(prompt_ids) < 2:
            continue
        # surface the last user turn for context
        last_user = ""
        for m in conv["messages"]:
            if m["role"] == "user" and isinstance(m["content"], str):
                last_user = m["content"]
        out = []
        for tok in model.generate(prompt_ids, max_tokens=C["sample_tokens"], temperature=0.7, top_k=50):
            if tok == tokenizer.encode_special("<|assistant_end|>"):
                break
            out.append(tok)
        text = tokenizer.decode(out).replace("\n", " ")[:160]
        print0(f"  [sample] Q: {last_user[:80]!r} -> A: {text!r}")
    model.train()


def _save_axim(path, model, cfg, enc, base_meta, step, val_loss, C, master=False):
    """Write the current model back to .axim (rank 0 only in DDP)."""
    from nanochat.common import is_ddp_initialized
    if not master and is_ddp_initialized():
        return  # only rank 0 writes
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    # clone config and stamp SFT info into it + metadata
    out_cfg = dict(cfg)
    out_cfg["training_step"] = step
    if val_loss is not None:
        out_cfg["val_loss"] = val_loss
    out_cfg["finetuned"] = True
    meta = dict(base_meta or {})
    meta["sft"] = {
        "step": step,
        "val_loss": val_loss,
        "base_model": C["base_model"],
        "train_file": C["train"],
        "optimizer": C["optimizer"],
        "config": {k: C[k] for k in ("max_seq_len", "train_on", "pack", "batch_size",
                                     "grad_accum", "epochs", "lr", "matrix_lr", "embedding_lr",
                                     "unembedding_lr", "weight_decay", "warmup_ratio",
                                     "warmdown_ratio", "final_lr_frac")},
    }
    tok_bytes = pickle.dumps(enc)
    write_axim(path, sd, out_cfg, tok_bytes, meta)


def _save_optim(path, optimizer, step, master=False):
    from nanochat.common import is_ddp_initialized
    if not master and is_ddp_initialized():
        return
    torch.save({"step": step, "optim": optimizer.state_dict()}, path)


# =============================================================================
# LR schedule  *** THE KEY KNOB - see notes below ***
# =============================================================================
def get_lr_multiplier(progress: float, C: Dict[str, Any]) -> float:
    """Return the LR multiplier in [final_lr_frac, 1.0] for a given progress in [0, 1].

    Default schedule: linear warmup (0 -> 1) over --warmup-ratio of training,
    constant at 1.0 in the middle, then linear warmdown (1 -> final_lr_frac) over
    --warmdown-ratio of training.

    *** This is the single most impactful knob to tune for SFT. ***
    Trade-offs:
      - warmup_ratio: longer warmup = stabler early training (SFT gradients can
        be noisy on small datasets) but spends fewer steps at peak LR. 0.02-0.05
        is typical for SFT; raise it if you see early divergence / NaNs.
      - warmdown_ratio: longer warmdown = smoother convergence to a flatter
        minimum (often better final quality) but less time at peak LR. 0.1-0.3
        is typical; nanochat's own SFT uses 0.5.
      - final_lr_frac: the LR floor at the end of warmdown. 0.0 (decay to zero)
        is the safe default; a small floor (0.05-0.1) can keep the model
        adapting slightly through the end.

    TODO(you): if you have a schedule that worked well for your base pretraining,
    swap it in here. Common alternatives:
      - cosine decay:    return 0.5*(1+cos(pi*t)) for the cooldown phase
      - constant + step decay: hold 1.0 then drop to a fraction at a milestone
      - WSD (warmup-stable-decay): the default above is already WSD
    All that matters for the rest of the trainer is that this returns a float
    multiplier; the optimizer code multiplies each group's initial_lr by it.
    """
    warmup = C["warmup_ratio"]
    warmdown = C["warmdown_ratio"]
    final_frac = C["final_lr_frac"]
    if progress < warmup:
        return (progress + 1e-8) / max(warmup, 1e-8)
    if progress <= 1.0 - warmdown:
        return 1.0
    decay = (progress - (1.0 - warmdown)) / max(warmdown, 1e-8)
    return (1 - decay) * 1.0 + decay * final_frac


if __name__ == "__main__":
    main()