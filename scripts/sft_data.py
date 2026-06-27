"""
Data pipeline for .axim SFT.

Self-contained: wraps the pickled tiktoken `enc` stored inside an .axim file and
re-implements nanochat's conversation rendering (so we don't depend on
nanochat.tokenizer, which imports `rustbpe` - not in this repo's requirements).

Supports several JSONL record shapes, normalizes them all to nanochat's
{"messages": [...]} chat format, renders to (ids, loss_mask), then packs into
fixed-length training batches with -1 (ignore_index) masking on padded/unsupervised
positions.

Conversation format (nanochat):
    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..." | [{"type": "text"|"python"|"python_output", "text": "..."}]}
    ]}
A leading "system" message is allowed and is merged into the first user message
(like nanochat does). The loss mask is 1 only on assistant tokens.
"""

import copy
import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

# Special tokens, in registration order (offset = mergeable_ranks + index here).
# Must match the tokenizer that was trained into the .axim file.
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
]


class AximTokenizer:
    """Minimal wrapper around a tiktoken Encoding pulled from an .axim file.

    Reimplements the pieces of nanochat's RustBPETokenizer that SFT needs:
    encode / decode / encode_special / get_bos_token_id / render_conversation.
    """

    def __init__(self, enc, bos_token: str = "<|bos|>"):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    # ---- basic encoding -------------------------------------------------
    def encode_special(self, text: str) -> int:
        # encode_single_token raises KeyError if `text` is not a registered special token
        return self.enc.encode_single_token(text)

    def encode(self, text: str) -> List[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    # ---- conversation rendering -----------------------------------------
    def render_conversation(self, conversation: Dict[str, Any],
                             max_tokens: int = 2048,
                             train_on_all: bool = False) -> Tuple[List[int], List[int]]:
        """Render a chat conversation into (ids, mask).

        mask[i] == 1  => position i is supervised (an assistant token).
        mask[i] == 0  => position i is ignored by the loss (user/prompt/special/pad).

        If train_on_all=True, every content token is supervised (useful for
        continued pre-training on chat-formatted data).
        """
        ids: List[int] = []
        mask: List[int] = []

        def add_tokens(token_ids, mask_val: int):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # Merge a leading system message into the following user message (nanochat behavior)
        conv = copy.deepcopy(conversation)
        messages = conv["messages"]
        assert len(messages) >= 1, f"conversation has <1 message: {messages}"
        if messages[0]["role"] == "system":
            assert messages[1]["role"] == "user", "system message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]

        bos = self.bos_token_id
        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")
        python_start = self.encode_special("<|python_start|>")
        python_end = self.encode_special("<|python_end|>")
        output_start = self.encode_special("<|output_start|>")
        output_end = self.encode_special("<|output_end|>")

        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, (
                f"message {i} is from {message['role']} but should be from {must_be_from}; "
                "conversations must alternate user/assistant starting with user"
            )
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "user messages must be strings"
                add_tokens(user_start, 0)
                add_tokens(self.encode(content), 1 if train_on_all else 0)
                add_tokens(user_end, 0)
            else:  # assistant
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    add_tokens(self.encode(content), 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        ptype = part["type"]
                        if ptype == "text":
                            add_tokens(value_ids, 1)
                        elif ptype == "python":
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif ptype == "python_output":
                            # tool output comes from the environment at inference time => don't supervise
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"unknown part type: {ptype}")
                else:
                    raise ValueError(f"unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_inference(self, conversation: Dict[str, Any], max_tokens: int = 2048) -> List[int]:
        """Render a conversation up to the assistant_start token (for prompting)."""
        conv = copy.deepcopy(conversation)
        messages = conv["messages"]
        # drop a trailing assistant turn so we prime the model to generate it
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        ids, _ = self.render_conversation({"messages": messages}, max_tokens=max_tokens)
        ids.append(self.encode_special("<|assistant_start|>"))
        return ids


# =============================================================================
# Record normalization - accept several JSONL shapes, emit {"messages": [...]}
# =============================================================================

def normalize_record(obj: Dict[str, Any], default_system: Optional[str] = None) -> Dict[str, Any]:
    """Turn one JSONL object into the canonical {"messages": [...]} conversation.

    Recognized shapes (detected by keys):
      - {"messages": [...]}                  -> nanochat/OpenAI chat (passed through)
      - {"prompt": str, "completion": str}  -> user=prompt, assistant=completion
      - {"instruction": str, "output": str, "input"?: str}
                                            -> Alpaca-style -> user (templated), assistant=output
      - {"text": str}                        -> single assistant turn; the "user" is empty
    """
    messages = obj.get("messages")
    if messages is not None:
        conv = {"messages": messages}
        if default_system and (not messages or messages[0]["role"] != "system"):
            conv = {"messages": [{"role": "system", "content": default_system}] + list(messages)}
        return conv

    if "prompt" in obj and "completion" in obj:
        msgs = []
        if default_system:
            msgs.append({"role": "system", "content": default_system})
        msgs.append({"role": "user", "content": obj["prompt"]})
        msgs.append({"role": "assistant", "content": obj["completion"]})
        return {"messages": msgs}

    if "instruction" in obj and "output" in obj:
        prompt = obj["instruction"]
        if obj.get("input"):
            prompt = (prompt or "") + "\n\n" + obj["input"]
        msgs = []
        if default_system:
            msgs.append({"role": "system", "content": default_system})
        msgs.append({"role": "user", "content": prompt})
        msgs.append({"role": "assistant", "content": obj["output"]})
        return {"messages": msgs}

    if "text" in obj:
        # raw text: treat as a single assistant completion with an empty user turn
        msgs = []
        if default_system:
            msgs.append({"role": "system", "content": default_system})
        msgs.append({"role": "user", "content": ""})
        msgs.append({"role": "assistant", "content": obj["text"]})
        return {"messages": msgs}

    raise ValueError(
        f"unrecognized record shape (keys: {list(obj.keys())}). "
        "expected one of: messages / prompt+completion / instruction+output / text"
    )


# =============================================================================
# JSONL loading
# =============================================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"bad JSON at {path}:{lineno}: {e}") from e
    return rows


def prepare_examples(rows: List[Dict[str, Any]], tokenizer: AximTokenizer,
                     max_seq_len: int, train_on_all: bool,
                     default_system: Optional[str] = None) -> List[Tuple[List[int], List[int]]]:
    """Normalize + render every row to (ids, mask). Drops empty/trivial rows."""
    out = []
    for r in rows:
        conv = normalize_record(r, default_system=default_system)
        ids, mask = tokenizer.render_conversation(conv, max_tokens=max_seq_len, train_on_all=train_on_all)
        if len(ids) < 2:
            continue  # nothing to learn from
        out.append((ids, mask))
    return out


# =============================================================================
# Packing / batching -> (inputs, targets) tensors with -1 masking
# =============================================================================

class PackedBatcher:
    """Yield fixed-length (inputs, targets) batches from rendered conversations.

    Packing modes:
      - "bestfit": BOS-aligned best-fit packing. Multiple conversations are
        bin-packed into each row; the remainder is padded with BOS tokens whose
        targets are masked to -1 (no tokens are ever dropped). Mirrors nanochat's
        SFT packing.
      - "single":  one conversation per row, truncated to max_seq_len and padded
        with BOS (masked). Simpler and keeps each example isolated.

    targets use int64 with -1 at padded/unsupervised positions (cross-entropy
    ignore_index). inputs are the targets shifted right by one (with a BOS in
    the first column), matching the standard next-token LM formulation.
    """

    def __init__(self, examples: List[Tuple[List[int], List[int]]],
                 batch_size: int, seq_len: int, pack: str = "bestfit",
                 bos_token_id: int = 0, shuffle: bool = True, seed: int = 42):
        assert pack in ("bestfit", "single")
        self.examples = list(examples)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pack = pack
        self.bos = bos_token_id
        self.shuffle = shuffle
        self.seed = seed

    def __len__(self) -> int:
        # rough estimate of number of batches (bestfit packs tightly)
        total = sum(len(ids) for ids, _ in self.examples)
        per_batch = self.batch_size * self.seq_len
        return max(1, total // per_batch)

    def _row_iter(self) -> Iterator[Tuple[List[int], List[int]]]:
        """Yield (row_ids, row_mask) of length seq_len+1 (one extra so the
        shifted inputs[:, :-1] / targets[:, 1:] come out to exactly seq_len)."""
        capacity = self.seq_len + 1

        if self.pack == "single":
            for ids, mask in self.examples:
                row = ids[:capacity]
                m = mask[:capacity]
                pad = capacity - len(row)
                if pad > 0:
                    row = row + [self.bos] * pad
                    m = m + [0] * pad
                yield row, m
            return

        # bestfit: keep a buffer of conversations, greedily fit the largest one
        # that still fits into the remaining row space (nanochat's approach).
        buf = list(self.examples)
        if self.shuffle:
            import random
            rng = random.Random(self.seed)
            rng.shuffle(buf)

        while buf:
            row: List[int] = []
            row_mask: List[int] = []
            while len(row) < capacity:
                remaining = capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(buf):
                    cl = len(conv)
                    if cl <= remaining and cl > best_len:
                        best_idx = i
                        best_len = cl
                if best_idx >= 0:
                    conv, cmask = buf.pop(best_idx)
                    row.extend(conv)
                    row_mask.extend(cmask)
                else:
                    # nothing fits -> pad the rest with BOS (masked)
                    pad = capacity - len(row)
                    row.extend([self.bos] * pad)
                    row_mask.extend([0] * pad)
                    break
            yield row, row_mask

    def batches(self, device: torch.device, pin_memory: bool = False,
                repeat: bool = False) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield (inputs, targets). inputs: (B, T) int32. targets: (B, T) int64 with -1 masked.

        If repeat=True, cycle forever, re-shuffling the buffer each pass (for training).
        If repeat=False, stop after one pass (for finite eval).
        """
        while True:
            rows: List[List[int]] = []
            masks: List[List[int]] = []
            for row, m in self._row_iter():
                rows.append(row)
                masks.append(m)
                if len(rows) == self.batch_size:
                    yield self._build(rows, masks, device, pin_memory)
                    rows, masks = [], []
            if rows:
                # pad the final short batch with BOS rows (all masked) so shapes stay constant
                while len(rows) < self.batch_size:
                    rows.append([self.bos] * (self.seq_len + 1))
                    masks.append([0] * (self.seq_len + 1))
                yield self._build(rows, masks, device, pin_memory)
            if not repeat:
                return
            # next epoch: _row_iter re-shuffles from self.examples automatically

    def _build(self, rows, masks, device, pin_memory):
        batch = torch.tensor(rows, dtype=torch.long, pin_memory=pin_memory)
        inputs = batch[:, :-1].to(device=device, dtype=torch.int32, non_blocking=pin_memory).contiguous()
        targets = batch[:, 1:].to(device=device, dtype=torch.int64, non_blocking=pin_memory).contiguous()
        # apply the loss mask (shifted by 1 to align with targets) and pad masking
        mask_t = torch.tensor(masks, dtype=torch.int8)
        mask_targets = mask_t[:, 1:].to(device=device)
        targets[mask_targets == 0] = -1
        return inputs, targets