"""Embedding E5 (Notebook 1). Vector được lưu vào Qdrant qua src/qdrant_store.py."""

import threading

import numpy as np

from src.config import EMBEDDING_MODEL

_model = None
# Chatbot gọi từ nhiều luồng: một luồng nạp mô hình, encode lần lượt (GPU không chạy song song được nhiều lô).
_lock = threading.RLock()


def embedder():
    """E5 đa ngôn ngữ 384 chiều; tự dùng GPU nếu có. Trên GPU chạy fp16: nhanh khoảng 3 lần,
    vector lệch không đáng kể so với fp32 (cosine giữa hai bản ≥ 0,999)."""
    global _model
    with _lock:
        if _model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            kwargs = {"model_kwargs": {"torch_dtype": torch.float16}} if torch.cuda.is_available() else {}
            _model = SentenceTransformer(EMBEDDING_MODEL, **kwargs)
    return _model


def dimension() -> int:
    """Số chiều vector của EMBEDDING_MODEL, dùng khi tạo collection Qdrant."""
    return embedder().get_sentence_embedding_dimension()


def embed_passages(texts: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
    # E5 cần tiền tố "passage: " cho văn bản lưu và "query: " cho câu hỏi.
    model = embedder()
    with _lock:
        return model.encode(["passage: " + t for t in texts], batch_size=batch_size,
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=progress).astype(np.float32)


def embed_query(question: str) -> np.ndarray:
    model = embedder()
    with _lock:
        return model.encode("query: " + question, normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)
