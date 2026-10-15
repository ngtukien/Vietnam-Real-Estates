"""Encoder tự viết (fp16, SDPA, lô theo token) khớp `SentenceTransformer.encode` (cosine ≥ 0,999), cả `passage: ` và `query: `,
và chuỗi id ghép từ chunk khớp tokenize cả văn bản đã lưu."""

import random

import numpy as np
import pytest

pytestmark = pytest.mark.gpu
N_SENTENCES, MIN_COSINE = 256, 0.999
WORDS = ("bán nhà mặt tiền quận bảy sổ hồng chính chủ giá tốt căn hộ hai phòng ngủ gần chợ trường học "
         "đất nền thổ cư hẻm xe hơi thang máy hồ bơi view sông nội thất đầy đủ cho thuê văn phòng").split()


@pytest.fixture(scope="module")
def models():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("không có CUDA")
    from sentence_transformers import SentenceTransformer

    from vre.config import load_settings
    from vre.index.embed import get_encoder
    from vre.index.tokenize import model_name

    settings = load_settings()
    reference = SentenceTransformer(model_name(), device="cuda", local_files_only=True)
    return settings, get_encoder(settings, "cuda"), reference


def _sentences() -> list[str]:
    rng = random.Random(42)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.choice((3, 12, 40, 120, 300)))) for _ in range(N_SENTENCES)]


@pytest.mark.parametrize("kind,prefix", [("passage", "passage: "), ("query", "query: ")])
def test_encoder_matches_sentence_transformers(models, kind, prefix):
    _, encoder, reference = models
    texts = _sentences()
    ours = encoder.encode_texts(texts, kind=kind).astype(np.float32)
    ref = reference.encode([prefix + t for t in texts], normalize_embeddings=True, batch_size=32)
    cosine = (ours * ref).sum(1) / np.linalg.norm(ours, axis=1)
    assert ours.shape == (N_SENTENCES, 1024) and ours.dtype == np.float32
    assert np.allclose(np.linalg.norm(ours, axis=1), 1.0, atol=1e-2)
    assert cosine.min() >= MIN_COSINE, f"cosine min {cosine.min():.5f}"


def test_chunk_ids_match_stored_text(models):
    from vre.index import chunk, tokenize

    settings, encoder, reference = models
    rows = [{"listing_id": i, "title": f"Bán nhà số {i}", "property_type": "Nhà", "province": "Hồ Chí Minh",
             "district": "Quận 7", "ward": "Tân Phong", "price_vnd": 7.45e9, "area_m2": 80.0, "bedrooms": 3,
             "description": text} for i, text in enumerate(_sentences()[:64])]
    tok = tokenize.load_tokenizer()  # tokenizer transformers: chuẩn để so
    fast = tokenize.load_fast_tokenizer()  # tokenizer mà worker chunk thật sự dùng
    table, sequences, _ = chunk.chunk_rows(rows, fast, tokenize.special_ids(fast), settings)
    for ids, text in zip(sequences, table["text"], strict=True):
        assert ids == tok("passage: " + text)["input_ids"]
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    np.cumsum([len(s) for s in sequences], out=offsets[1:])
    flat = np.fromiter((t for s in sequences for t in s), dtype=np.int64, count=int(offsets[-1]))
    ours = encoder.encode_ids(flat, offsets).astype(np.float32)
    ref = reference.encode(["passage: " + t for t in table["text"]], normalize_embeddings=True, batch_size=32)
    cosine = (ours * ref).sum(1) / np.linalg.norm(ours, axis=1)
    assert cosine.min() >= MIN_COSINE, f"cosine min {cosine.min():.5f}"
