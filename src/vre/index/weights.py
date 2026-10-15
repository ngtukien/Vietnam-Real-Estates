"""Nạp trọng số e5-large ít bộ nhớ, dùng chung cho encoder lô GPU (`vre.index.embed`) và encoder truy vấn
(`vre.rag.query_encoder`). Đây là loader DUY NHẤT của dự án cho model này.

Vì sao không dùng `AutoModel.from_pretrained`: transformers 5.x mở safetensors bằng backend "mmap" =
`torch.UntypedStorage.from_file(shared=False)`, tức một view copy-on-write tính cả file 2,24 GB vào commit. Trên Windows
khi commit trống < ~2,24 GB, `MapViewOfFile` trả NULL (lỗi 1455) mà torch không kiểm tra ⇒ storage có base NULL ⇒
access violation khi đọc tensor đầu tiên (plans/reports/p3-encoder-crash.md). Ở đây: dựng model trên `meta`, đọc file
bằng `readinto` theo lát ≤ 64 MB và ép fp16 ngay ⇒ đỉnh ≈ model fp16 (1,1 GB) + một lát, không có view COW nào, và
thiếu commit thì nổ `MemoryError` của Python thay vì sập tiến trình.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

from vre.config import EMBED_MODEL_SLUG, EMBED_MODELS

SLICE_BYTES = 64 * 1024 * 1024
_NUMPY_DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64}


def _weights_path() -> str:
    from huggingface_hub import try_to_load_from_cache

    from vre.index import tokenize

    path = try_to_load_from_cache(tokenize.model_name(), "model.safetensors")
    if not isinstance(path, str):
        raise RuntimeError(f"không có model.safetensors của {tokenize.model_name()} trong HF_HOME (không tự tải)")
    return path


def load_state_fp16(path: str) -> dict[str, Any]:
    """Đọc safetensors thành tensor CPU; số thực ép fp16 theo từng lát (không giữ bản fp32 nào)."""
    import torch

    state: dict[str, Any] = {}
    with open(path, "rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(size))
        base = 8 + size
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            src = _NUMPY_DTYPES[meta["dtype"]]
            begin, end = meta["data_offsets"]
            floating = np.issubdtype(src, np.floating)
            out = torch.empty(meta["shape"], dtype=torch.float16 if floating else torch.from_numpy(np.zeros(0, src)).dtype)
            flat = out.view(-1)
            step = max(1, SLICE_BYTES // np.dtype(src).itemsize)
            fh.seek(base + begin)
            buf = bytearray(min(end - begin, step * np.dtype(src).itemsize))
            for start in range(0, flat.numel(), step):
                count = min(step, flat.numel() - start)
                view = memoryview(buf)[:count * np.dtype(src).itemsize]
                fh.readinto(view)
                flat[start:start + count] = torch.from_numpy(np.frombuffer(view, dtype=src))
            state[name] = out
    return state


def build_model_fp16() -> Any:
    """Model e5-large fp16 + SDPA trên CPU, chế độ eval, nạp bằng `load_state_fp16` (không mmap, không `from_file`)."""
    import torch
    from transformers import AutoConfig, AutoModel

    name, _, pooling, _, _ = EMBED_MODELS[EMBED_MODEL_SLUG]
    if pooling != "mean":
        raise ValueError(f"pooling {pooling!r} chưa hỗ trợ")
    config = AutoConfig.from_pretrained(name, local_files_only=True)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = AutoModel.from_config(config, dtype=torch.float16)
    state = load_state_fp16(_weights_path())
    buffers = {k for k, _ in model.named_buffers()}
    model.load_state_dict({k: v for k, v in state.items() if k not in buffers}, strict=False, assign=True)
    positions = config.max_position_embeddings
    model.embeddings.register_buffer("position_ids", torch.arange(positions).unsqueeze(0), persistent=False)
    model.embeddings.register_buffer("token_type_ids", torch.zeros((1, positions), dtype=torch.long), persistent=False)
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("trọng số e5-large thiếu tham số sau khi nạp")
    return model.eval()
