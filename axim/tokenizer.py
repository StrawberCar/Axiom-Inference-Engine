"""Tokenizer wrapper + chat rendering — the single source for tokenization.

Wraps the pickled tiktoken Encoding pulled from an .axim file and reimplements
the pieces nanochat's RustBPETokenizer provides that inference needs:
encode / decode / encode_special / bos / render_chat.
"""

import copy
from typing import List


class Tokenizer:
    """Minimal wrapper around a tiktoken Encoding.

    Used by `axim.serve`, `axim` infer, and subclassed by `axim.sft.data.AximTokenizer`
    (which adds the training-side render_conversation with assistant masking).
    """

    def __init__(self, enc):
        self.enc = enc
        self._bos = enc.encode_single_token("<|bos|>")

    @property
    def bos(self) -> int:
        return self._bos

    def encode_special(self, text: str) -> int:
        # encode_single_token raises KeyError if `text` is not a registered special token
        return self.enc.encode_single_token(text)

    def encode(self, text: str) -> List[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)

    def render_chat(self, messages, max_len: int = 2048) -> List[int]:
        """Render a chat conversation into the inference prompt scaffold.

        Produces: <|bos|> <|user_start|> ... <|user_end|> <|assistant_start|> ... <|assistant_end|> ...
        A leading system message is merged into the first user turn's content
        (nanochat convention). Does NOT append a trailing <|assistant_start|> —
        the caller appends it to prime generation.
        """
        ids: List[int] = []
        msgs = copy.deepcopy(messages)

        if msgs and msgs[0]["role"] == "system":
            msgs[1]["content"] = msgs[0]["content"] + "\n\n" + msgs[1]["content"]
            msgs = msgs[1:]

        us = self.encode_special("<|user_start|>")
        ue = self.encode_special("<|user_end|>")
        ast = self.encode_special("<|assistant_start|>")
        aen = self.encode_special("<|assistant_end|>")

        ids.append(self._bos)
        for i, msg in enumerate(msgs):
            want = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == want, f"expected {want!r} at index {i}, got {msg['role']!r}"
            if msg["role"] == "user":
                ids += [us] + self.encode(msg["content"]) + [ue]
            else:
                ids += [ast] + self.encode(msg["content"]) + [aen]

        return ids[:max_len]