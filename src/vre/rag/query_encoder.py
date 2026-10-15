"""Encoder TRUY VẤN e5-large cho QA/API: CPU, fp16, nạp trọng số ít bộ nhớ (data-contract §9.1: tiền tố `query: `,
mean pooling, L2).

Vì sao không dùng thẳng `vre.index.embed.Encoder` (encoder lô GPU của lượt chạy dài):
- Trên Windows (WDDM) bộ nhớ CUDA của tiến trình được TÍNH VÀO commit: context CUDA ~1,9 GB + model 1,1 GB. Đo
  2026-10-15: một lệnh `ask` qua encoder GPU đạt 5,9 GB private commit, máy còn 1,9 GB commit trống.
- `from_pretrained` (cả CPU lẫn GPU) dựng fp32 rồi mới ép fp16, và safetensors mmap copy-on-write tính cả file
  2,24 GB vào commit ⇒ đỉnh 5,7–7,4 GB.
Ở đây: dựng model trên `meta`, đọc safetensors bằng `readinto` theo lát ≤ 64 MB và ép fp16 ngay ⇒ đỉnh ≈ model fp16
(1,1 GB) + một lát. Một truy vấn ngắn chạy ~0,2 s trên CPU nên không cần GPU. Tokenizer + tên model dùng chung
`vre.index.tokenize` để truy vấn khớp đúng cách chỉ mục được xây.
"""

from __future__ import annotations

import json
import struct
import threading
from typing import Any

import numpy as np

from vre.config import EMBED_MODEL_SLUG, EMBED_MODELS

SLICE_BYTES = 64 * 1024 * 1024
_NUMPY_DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64}
_lock = threading.Lock()
_encoder: QueryEncoder | None = None


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


class QueryEncoder:
    def __init__(self) -> None:
        import torch
        from transformers import AutoConfig, AutoModel

        from vre.index import tokenize

        self.torch = torch
        name, self.dim, pooling, _, self.prefix = EMBED_MODELS[EMBED_MODEL_SLUG]
        if pooling != "mean":
            raise ValueError(f"pooling {pooling!r} chưa hỗ trợ")
        self.tokenizer = tokenize.load_tokenizer()
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
        self.model = model.eval()
        self.max_length = 512

    def encode(self, text: str) -> np.ndarray:
        torch = self.torch
        enc = self.tokenizer([self.prefix + text], truncation=True, max_length=self.max_length, return_tensors="pt")
        with torch.inference_mode():
            hidden = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            vector = torch.nn.functional.normalize(pooled.float(), dim=-1)[0]
        return vector.numpy().astype(np.float32)


def get_query_encoder() -> QueryEncoder:
    global _encoder
    with _lock:
        if _encoder is None:
            _encoder = QueryEncoder()
        return _encoder


def embed(text: str) -> list[float]:
    """Vector truy vấn 1024 chiều, L2 = 1. An toàn đa luồng (một model, khoá khi suy luận)."""
    encoder = get_query_encoder()
    with _lock:
        return encoder.encode(text).tolist()
