"""
axim — Axiom Model Package Format + inference engine.

Public API:
    load_axim, write_axim, print_axim_info  — .axim format I/O
    Tokenizer                                — tiktoken wrapper + chat rendering
    load_model, load_from_dir                — build a nanochat GPT from .axim / safetensors
    sample                                   — autoregressive sampling loop
"""

from .core import load_axim, write_axim, print_axim_info, SectionType, AXIMHeader
from .tokenizer import Tokenizer
from .model import load_model, load_from_dir
from .generate import sample

__all__ = [
    "load_axim", "write_axim", "print_axim_info", "SectionType", "AXIMHeader",
    "Tokenizer", "load_model", "load_from_dir", "sample",
]