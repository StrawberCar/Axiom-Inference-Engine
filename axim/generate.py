"""Autoregressive sampling — the single sampling loop shared by serve + infer.

Stops on the model's own <|assistant_end|> / <|bos|> and additionally on any
user-supplied stop_strings (substring match on decoded-so-far; the stop string
itself is NOT emitted, OpenAI-style). HF-style repetition penalty, top-k, and
greedy vs sampling. If `status` (a dict) is passed, it is populated with the
finish reason after the loop exits.
"""

import torch
import torch.nn.functional as F


@torch.inference_mode()
def sample(model, tok, prompt_ids, max_tokens, temperature, top_k, device,
           repetition_penalty=1.0, stop_strings=None, status=None):
    """Yield generated token ids one at a time."""
    def _finish(reason, cause):
        if status is not None:
            status["reason"] = reason
            status["cause"] = cause

    stops = [s for s in (stop_strings or []) if s]
    aend = tok.encode_special("<|assistant_end|>")
    bos = tok.bos
    greedy = (temperature is None or temperature <= 0)

    rng = None
    if not greedy:
        rng = torch.Generator(device=device)
        rng.manual_seed(42)

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, T)
    generated = []
    stop_cause = None
    for _ in range(max_tokens):
        logits = model.forward(ids)[:, -1, :]  # (1, vocab)

        if repetition_penalty and repetition_penalty != 1.0 and generated:
            for t in set(generated):
                logits[0, t] = logits[0, t] / repetition_penalty if logits[0, t] > 0 \
                    else logits[0, t] * repetition_penalty

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        if greedy:
            next_id = int(torch.argmax(logits, dim=-1).item())
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1, generator=rng).item())

        if next_id == aend or next_id == bos:
            stop_cause = "eos"
            break
        if stops:
            decoded = tok.decode(generated + [next_id])
            hit = next((s for s in stops if s in decoded), None)
            if hit is not None:
                stop_cause = ("stop_string", hit)
                break

        generated.append(next_id)
        ids = torch.cat((ids, torch.tensor([[next_id]], dtype=torch.long, device=device)), dim=1)
        yield next_id

    if stop_cause is None:
        _finish("length", f"max_tokens={max_tokens} reached")
    elif stop_cause == "eos":
        _finish("stop", "<|assistant_end|> generated")
    else:
        _finish("stop", f"stop string {stop_cause[1]!r} matched")