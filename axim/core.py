"""
AXIM - Axiom Model Package Format v1

Binary format that stuffs model weights (safetensors), config (json),
tokenizer (pickle), and metadata (json) into one file.

aka not a zip lmfao..
"""

import struct
import json
import pickle
import os
from pathlib import Path
from typing import Dict, Any, Optional, BinaryIO
from enum import IntEnum
from dataclasses import dataclass
from io import BytesIO

import torch
from safetensors.torch import save_file, load_file, load as load_safetensors


MAGIC = b"AXIM\x00"
VERSION = 1


class SectionType(IntEnum):
    WEIGHTS = 1
    CONFIG = 2
    TOKENIZER = 3
    METADATA = 4


NAMES = {
    SectionType.WEIGHTS: "weights",
    SectionType.CONFIG: "config",
    SectionType.TOKENIZER: "tokenizer",
    SectionType.METADATA: "metadata",
}


@dataclass
class Section:
    type: SectionType
    offset: int
    size: int
    name: str


@dataclass
class AXIMHeader:
    magic: bytes
    version: int
    flags: int
    sections: list


def _read_header(f: BinaryIO) -> AXIMHeader:
    magic = f.read(5)
    if magic != MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    
    version, flags = struct.unpack("<II", f.read(8))
    if version != VERSION:
        raise ValueError(f"bad version: {version}, expected {VERSION}")
    
    num = struct.unpack("<I", f.read(4))[0]
    sections = []
    
    for _ in range(num):
        t, off, size, nlen = struct.unpack("<IQQI", f.read(24))
        name = f.read(nlen).decode("utf-8")
        sections.append(Section(type=SectionType(t), offset=off, size=size, name=name))
    
    return AXIMHeader(magic=magic, version=version, flags=flags, sections=sections)


def load_axim(path: str | Path) -> Dict[str, Any]:
    """Load an .axim file. Returns dict with keys: weights, config, tokenizer, metadata."""
    path = Path(path)
    result = {}
    
    with open(path, "rb") as f:
        header = _read_header(f)
        
        for sec in header.sections:
            f.seek(sec.offset)
            data = f.read(sec.size)
            
            if sec.type == SectionType.WEIGHTS:
                result["weights"] = load_safetensors(data)
            elif sec.type == SectionType.CONFIG:
                result["config"] = json.loads(data.decode("utf-8"))
            elif sec.type == SectionType.TOKENIZER:
                result["tokenizer"] = pickle.loads(data)
            elif sec.type == SectionType.METADATA:
                result["metadata"] = json.loads(data.decode("utf-8"))
    
    return result


def write_axim(
    path: str | Path,
    weights: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    tokenizer_data: bytes,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Write an .axim file from components."""
    path = Path(path)
    
    sections_data = []

    # serialize weights to a safetensors blob (bytes). Newer safetensors versions
    # reject a BytesIO in save_file(), so write to a temp file and read it back.
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp:
        tmp_path = tmp.name
    try:
        save_file(weights, tmp_path)
        with open(tmp_path, "rb") as f:
            weights_bytes = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    sections_data.append((SectionType.WEIGHTS, "model.safetensors", weights_bytes))
    
    cfg_bytes = json.dumps(config, indent=2).encode("utf-8")
    sections_data.append((SectionType.CONFIG, "config.json", cfg_bytes))
    
    sections_data.append((SectionType.TOKENIZER, "tokenizer.pkl", tokenizer_data))
    
    if metadata:
        meta_bytes = json.dumps(metadata, indent=2).encode("utf-8")
        sections_data.append((SectionType.METADATA, "metadata.json", meta_bytes))
    
    num = len(sections_data)
    table_size = 4 + num * 24 + sum(len(name.encode("utf-8")) for _, name, _ in sections_data)
    
    header_and_table = 13 + table_size
    pad = (8 - header_and_table % 8) % 8
    data_off = header_and_table + pad
    
    sections = []
    cur = data_off
    for stype, name, data in sections_data:
        sec_pad = (8 - cur % 8) % 8
        cur += sec_pad
        sections.append({"type": stype, "name": name, "offset": cur, "size": len(data), "data": data})
        cur += len(data)
    
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, 0))
        f.write(struct.pack("<I", num))
        
        for sec in sections:
            nb = sec["name"].encode("utf-8")
            f.write(struct.pack("<IQQI", sec["type"], sec["offset"], sec["size"], len(nb)))
            f.write(nb)
        
        f.write(b"\x00" * pad)
        
        for sec in sections:
            f.seek(sec["offset"])
            f.write(sec["data"])
    
    total = path.stat().st_size
    print(f"wrote {path} ({total / 1e9:.2f} GB)")
    for sec in sections:
        print(f"  {sec['name']}: {sec['size'] / 1e6:.1f} MB")


def print_axim_info(path: str | Path):
    """Print info about an .axim file."""
    path = Path(path)
    
    with open(path, "rb") as f:
        header = _read_header(f)
    
    print(f"AXIM: {path}")
    print(f"  version: {header.version}")
    print(f"  sections: {len(header.sections)}")
    print(f"  size: {path.stat().st_size / 1e9:.2f} GB")
    print()
    
    for sec in header.sections:
        tname = NAMES.get(sec.type, f"unknown({sec.type})")
        print(f"  [{tname}] {sec.name}")
        print(f"    offset: {sec.offset}, size: {sec.size / 1e6:.1f} MB")
    
    data = load_axim(path)
    if "config" in data:
        cfg = data["config"]
        print(f"\n  model: {cfg.get('model_type', '?')}")
        print(f"    layers: {cfg.get('num_hidden_layers', '?')}")
        print(f"    heads: {cfg.get('num_attention_heads', '?')}")
        print(f"    hidden: {cfg.get('hidden_size', '?')}")
        print(f"    vocab: {cfg.get('vocab_size', '?')}")
        print(f"    seq_len: {cfg.get('max_position_embeddings', '?')}")
    
    if "weights" in data:
        total = sum(v.numel() for v in data["weights"].values())
        print(f"    params: {total:,} ({total/1e9:.2f}B)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print_axim_info(sys.argv[1])
    else:
        print("usage: python core.py <file.axim>")
