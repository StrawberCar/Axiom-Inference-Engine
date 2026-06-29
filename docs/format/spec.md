# AXIM v1 Format Specification

`.axim` is a binary package format for nanoChat-style models. A single file contains model weights, architecture config, tokenizer, and metadata.

## Design goals

- **Single file**: everything needed for inference or fine-tuning is inside.
- **Memory-mappable weights**: the weights section is a raw safetensors blob, so it loads as fast as safetensors.
- **Small metadata overhead**: the header + section table is typically a few hundred bytes.
- **Simple parsing**: the file is read sequentially once the section table is known.

## File layout

```
[Header]            13 bytes
  magic: "AXIM\0"    5 bytes
  version: uint32     4 bytes
  flags: uint32       4 bytes

[Section Table]     variable
  num_sections: uint32
  sections[]:
    type:   uint32    (1=weights, 2=config, 3=tokenizer, 4=metadata)
    offset: uint64    absolute byte offset of section data
    size:   uint64    length of section data in bytes
    name_len: uint32  length of section name in UTF-8 bytes
    name:   UTF-8     variable-length section name

[Section Data]      aligned to 8-byte boundaries
  weights:   safetensors blob
  config:    JSON text
  tokenizer: pickle bytes
  metadata:  JSON text
```

### Header byte layout

| Offset | Size | Field | Value |
|--------|------|-------|-------|
| 0 | 5 | magic | `AXIM\0` (bytes) |
| 5 | 4 | version | `1` (uint32, little-endian) |
| 9 | 4 | flags | `0` (uint32, little-endian) |

Total header size: **13 bytes**.

### Section table entry

Each entry is exactly **24 bytes** of fixed fields plus the variable-length name:

| Offset in entry | Size | Field |
|-----------------|------|-------|
| 0 | 4 | type (uint32) |
| 4 | 8 | offset (uint64) |
| 12 | 8 | size (uint64) |
| 20 | 4 | name_len (uint32) |
| 24 | name_len | name (UTF-8) |

## Section types

| Value | Name | Stored data |
|-------|------|-------------|
| 1 | `WEIGHTS` | safetensors blob — `model.safetensors` |
| 2 | `CONFIG` | JSON text — `config.json` |
| 3 | `TOKENIZER` | pickle bytes — `tokenizer.pkl` |
| 4 | `METADATA` | JSON text — `metadata.json` |

## Alignment

After the section table, **padding bytes** are inserted so the first section starts on an **8-byte boundary**. Likewise, each section is padded so the next section also starts on an 8-byte boundary. Sections themselves are stored contiguously after the table.

!!! tip "Why 8-byte alignment?"
    Keeping section starts on 8-byte boundaries makes the file friendlier to memory mapping and direct pointer arithmetic, and it keeps the safetensors blob aligned the way safetensors expects.

## Reading an `.axim` file

1. Verify the 5-byte magic is `AXIM\0`.
2. Read `version` and `flags` (currently version must be `1`, flags must be `0`).
3. Read `num_sections`.
4. For each section, read `type`, `offset`, `size`, and `name`.
5. Seek to each section's `offset`, read `size` bytes, and deserialize based on `type`:
   - `WEIGHTS` → safetensors load
   - `CONFIG` → JSON parse
   - `TOKENIZER` → pickle load
   - `METADATA` → JSON parse

## Writing an `.axim` file

1. Serialize each section independently:
   - Weights via `safetensors.torch.save_file` into a temporary file, then read back as bytes.
   - Config as UTF-8 JSON.
   - Tokenizer as raw pickle bytes.
   - Metadata as UTF-8 JSON, if present.
2. Compute the total header + section table size.
3. Add padding so the first data section is aligned to 8 bytes.
4. Compute each section's absolute offset, adding padding between sections.
5. Write the header, section table, padding, and then each section's raw bytes.

## Limits and notes

- Section names are stored as raw UTF-8 bytes, not null-terminated.
- All integer fields are little-endian.
- The format currently supports only version `1` and flags `0`.
- The weights section can be loaded directly with `safetensors.torch.load` because it is a complete safetensors file in memory.
