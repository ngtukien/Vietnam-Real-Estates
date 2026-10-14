"""Tokenizer XLM-R của e5-large và định dạng token tạm `index/tok/part_NNNN.npz` (data-contract §4).

Token của một part được lưu phẳng (int32) kèm mảng offsets để tiến trình GPU đọc thẳng từ đĩa,
không pickle danh sách Python lớn qua queue giữa các tiến trình.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from vre.config import EMBED_MODEL_SLUG, EMBED_MODELS


@dataclass(frozen=True)
class SpecialIds:
    """Các id cố định ghép quanh mỗi chunk: [bos] + tiền tố + header + nl + body + [eos]."""

    bos: int
    eos: int
    pad: int
    passage_prefix: tuple[int, ...]
    query_prefix: tuple[int, ...]
    newline: tuple[int, ...]


def model_name() -> str:
    return EMBED_MODELS[EMBED_MODEL_SLUG][0]


@lru_cache(maxsize=1)
def load_tokenizer():
    """Tokenizer fast của model đã cache; chỉ đọc cache cục bộ (không bao giờ tải)."""
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return AutoTokenizer.from_pretrained(model_name(), local_files_only=True)


def special_ids(tokenizer) -> SpecialIds:
    _, _, _, passage, query = EMBED_MODELS[EMBED_MODEL_SLUG]

    def ids(text: str) -> tuple[int, ...]:
        return tuple(tokenizer(text, add_special_tokens=False)["input_ids"])

    return SpecialIds(bos=tokenizer.bos_token_id, eos=tokenizer.eos_token_id, pad=tokenizer.pad_token_id,
                      passage_prefix=ids(passage), query_prefix=ids(query), newline=ids("\n"))


def save_tokens(path: Path, sequences: list[list[int]]) -> None:
    """Ghi nguyên tử: `.tmp` rồi `os.replace` (part dở dang không bao giờ trông như đã xong)."""
    lengths = np.fromiter((len(s) for s in sequences), dtype=np.int64, count=len(sequences))
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = np.fromiter((t for s in sequences for t in s), dtype=np.int32, count=int(offsets[-1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, ids=flat, offsets=offsets)
    os.replace(tmp, path)


def load_tokens(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(ids phẳng int32, offsets int64 dài n+1): chuỗi i = ids[offsets[i]:offsets[i+1]]."""
    with np.load(path) as data:
        return data["ids"], data["offsets"]
