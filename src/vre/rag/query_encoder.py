"""Encoder TRUY VẤN e5-large cho QA/API: CPU, fp16, nạp trọng số ít bộ nhớ (data-contract §9.1: tiền tố `query: `,
mean pooling, L2).

Vì sao không dùng thẳng `vre.index.embed.Encoder` (encoder lô GPU của lượt chạy dài):
- Trên Windows (WDDM) bộ nhớ CUDA của tiến trình được TÍNH VÀO commit: context CUDA ~1,9 GB + model 1,1 GB. Đo
  2026-10-15: một lệnh `ask` qua encoder GPU đạt 5,9 GB private commit, máy còn 1,9 GB commit trống.
- `from_pretrained` (cả CPU lẫn GPU) dựng fp32 rồi mới ép fp16, và safetensors mmap copy-on-write tính cả file
  2,24 GB vào commit ⇒ đỉnh 5,7–7,4 GB.
Ở đây: nạp bằng `vre.index.weights.build_model_fp16` (meta + `readinto` theo lát ≤ 64 MB, ép fp16 ngay) ⇒ đỉnh ≈ model
fp16 (1,1 GB) + một lát. Một truy vấn ngắn chạy ~0,2 s trên CPU nên không cần GPU. Tokenizer + tên model dùng chung
`vre.index.tokenize` để truy vấn khớp đúng cách chỉ mục được xây.
"""

from __future__ import annotations

import threading

import numpy as np

from vre.config import EMBED_MODEL_SLUG, EMBED_MODELS

_lock = threading.Lock()
_encoder: QueryEncoder | None = None


class QueryEncoder:
    def __init__(self) -> None:
        import torch

        from vre.index import tokenize
        from vre.index.weights import build_model_fp16

        self.torch = torch
        _, self.dim, _, _, self.prefix = EMBED_MODELS[EMBED_MODEL_SLUG]
        self.tokenizer = tokenize.load_tokenizer()
        self.model = build_model_fp16()
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
