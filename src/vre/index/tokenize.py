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


class FastTokenizer:
    """Tokenizer Rust `tokenizers` từ `tokenizer.json` của model, cùng giao diện gọi với tokenizer transformers mà chunk cần.

    Dùng trong tiến trình con tokenize: `transformers` kéo theo torch + sklearn + scipy ⇒ ~1,3 GB commit mỗi worker
    (đo 2026-10-15); bản này ~0,3 GB. Id và offsets trùng khớp tokenizer transformers (kiểm trên 3.005 văn bản thật + ca biên).
    """

    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(path))
        self.bos_token_id, self.eos_token_id = self._tok.token_to_id("<s>"), self._tok.token_to_id("</s>")
        self.pad_token_id = self._tok.token_to_id("<pad>")

    def __call__(self, texts, add_special_tokens=False, return_offsets_mapping=False):
        single = isinstance(texts, str)
        encodings = self._tok.encode_batch([texts] if single else list(texts), add_special_tokens=add_special_tokens)
        out = {"input_ids": [e.ids for e in encodings]}
        if return_offsets_mapping:
            out["offset_mapping"] = [e.offsets for e in encodings]
        return {k: v[0] for k, v in out.items()} if single else out


@lru_cache(maxsize=1)
def load_fast_tokenizer() -> FastTokenizer:
    """Tokenizer cho tiến trình chunk (không import torch). Chỉ đọc snapshot đã cache (HF_HUB_OFFLINE)."""
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return FastTokenizer(Path(snapshot_download(model_name(), local_files_only=True)) / "tokenizer.json")


def special_ids(tokenizer) -> SpecialIds:
    _, _, _, passage, query = EMBED_MODELS[EMBED_MODEL_SLUG]

    def ids(text: str) -> tuple[int, ...]:
        return tuple(tokenizer(text, add_special_tokens=False)["input_ids"])

    return SpecialIds(bos=tokenizer.bos_token_id, eos=tokenizer.eos_token_id, pad=tokenizer.pad_token_id,
                      passage_prefix=ids(passage), query_prefix=ids(query), newline=ids("\n"))


def save_tokens(path: Path, flat: np.ndarray, lengths: np.ndarray) -> None:
    """Ghi nguyên tử (`.tmp` rồi `os.replace`): id phẳng int32 + độ dài từng chuỗi → npz `ids` + `offsets`."""
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    if offsets[-1] != len(flat):
        raise ValueError(f"tổng độ dài {offsets[-1]} ≠ số id {len(flat)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, ids=flat, offsets=offsets)
    os.replace(tmp, path)


def load_tokens(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(ids phẳng int32, offsets int64 dài n+1): chuỗi i = ids[offsets[i]:offsets[i+1]]."""
    with np.load(path) as data:
        return data["ids"], data["offsets"]
