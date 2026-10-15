"""Phân giải tên tự do (dự án, đường) về tên thật trong dữ liệu (entity resolution cho bộ lọc).

Tên dự án trong payload không chuẩn: có hậu tố quận ('Sun Feliza Suites Cầu Giấy'), tiền tố loại khu
('Khu đô thị Ecopark'), bí danh trong ngoặc ('Hà Nội Paragon (Paragon Tower)'). So khớp `MatchValue` đúng chuỗi
với cách viết của người dùng/LLM thì 0 kết quả (plans/reports/p3-filter-drop.md). Ở đây gập cả hai phía
(NFC, chữ thường, bỏ dấu, bỏ tiền tố/ngoặc/hậu tố quận-tỉnh) rồi lấy các tên thật mà khoá gập của chúng
bằng / chứa / nằm trong khoá của câu hỏi (dãy từ liên tiếp). Kết quả là danh sách tên thật để
`filters.conditions` phát điều kiện `in` — cùng một danh sách cho Qdrant, SQL và pandas (B4).

Từ vựng (tên, quận, tỉnh, số tin) lấy một lần từ bảng sạch bằng DuckDB (vài nghìn dòng, ~0,1 s), cache ở
`<cache_dir>/name_vocab_<field>.parquet`, dựng lại khi bảng sạch mới hơn cache.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

log = logging.getLogger(__name__)

NAME_FIELDS = ("project", "street")
# Địa danh đã qua `normalize_place` nhưng có thể thiếu dấu ('Thu Duc'): chỉ nhận khoá gập BẰNG tên thật.
PLACE_FIELDS = ("province", "district", "ward")
MAX_CANDIDATES = 20
# Tiền tố loại khu (sau khi gập) — bỏ ở đầu tên, cả hai phía.
_PREFIXES = {
    "project": ("chung cu", "khu do thi", "khu dan cu", "kdt", "kdc", "du an", "phan khu", "khu can ho", "can ho"),
    "street": ("duong", "pho"),
    **{place: () for place in PLACE_FIELDS},
}
# Từ chung chung: tên thật chỉ gồm những từ này không được khớp kiểu "nằm trong câu hỏi" (quá rộng).
_GENERIC = frozenset({"city", "park", "tower", "towers", "residence", "residences", "home", "homes", "garden",
                      "gardens", "plaza", "center", "central", "house", "the", "green", "river", "sky", "star",
                      "new", "town", "villa", "villas", "apartment", "apartments", "khu", "so", "1", "2", "3"})
# Đường: chỉ khớp bằng hoặc câu hỏi nằm trong tên (tên đường ngắn, "nằm trong" dễ khớp nhầm).
_CONTAINED_OK = {"project": True, "street": False, **{place: False for place in PLACE_FIELDS}}
_PARENS = re.compile(r"\([^)]*\)")
_NON_WORD = re.compile(r"[^0-9a-z]+")


def fold(text: str) -> str:
    """NFC → chữ thường → bỏ dấu (đ → d) → bỏ ngoặc → chỉ giữ chữ/số, cách nhau một khoảng trắng."""
    text = unicodedata.normalize("NFC", text).lower().replace("đ", "d")
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    return " ".join(_NON_WORD.sub(" ", _PARENS.sub(" ", text)).split())


def _strip_prefix(key: str, field: str) -> str:
    for prefix in _PREFIXES[field]:
        if key.startswith(prefix + " ") and len(key) > len(prefix) + 1:
            return key[len(prefix) + 1:]
    return key


def _strip_suffix(key: str, places: list[str]) -> str:
    """Bỏ hậu tố là tên quận/tỉnh (dài nhất trước); không bỏ nếu chỉ còn rỗng."""
    for place in places:
        if place and key.endswith(" " + place):
            return key[: -len(place) - 1]
    return key


def _contains(outer: tuple[str, ...], inner: tuple[str, ...]) -> bool:
    n = len(inner)
    return 0 < n <= len(outer) and any(outer[i:i + n] == inner for i in range(len(outer) - n + 1))


@dataclass(frozen=True)
class _Entry:
    name: str
    key: tuple[str, ...]
    district: str | None
    province: str | None
    n: int


class NameIndex:
    """Chỉ mục tên của một trường: khoá gập → tên thật; tra theo từ (inverted index)."""

    def __init__(self, field: str, vocab: pd.DataFrame) -> None:
        if field not in NAME_FIELDS + PLACE_FIELDS:
            raise ValueError(f"field phải là một trong {NAME_FIELDS + PLACE_FIELDS}, nhận {field!r}")
        self.field = field
        self.exact = field in PLACE_FIELDS  # địa danh: chỉ khớp khoá gập bằng nhau, không bỏ hậu tố
        places = {fold(p) for col in ("district", "province") for p in vocab[col].dropna().unique()}
        self._places = sorted((p for p in places if p), key=len, reverse=True)
        self._entries: list[_Entry] = []
        self._by_token: dict[str, list[int]] = defaultdict(list)
        rows = zip(vocab["name"].tolist(), vocab["district"].tolist(), vocab["province"].tolist(),
                   vocab["n"].tolist(), strict=True)
        for name, district, province, n in rows:
            district = district if isinstance(district, str) else None
            province = province if isinstance(province, str) else None
            own = [fold(x) for x in (district, province) if x]
            key = fold(name) if self.exact else _strip_suffix(_strip_prefix(fold(name), field),
                                                              sorted(own, key=len, reverse=True))
            if not key:
                continue
            entry = _Entry(name, tuple(key.split()), district, province, int(n))
            for token in set(entry.key):
                self._by_token[token].append(len(self._entries))
            self._entries.append(entry)

    def _query_keys(self, value: str, district: str | None, province: str | None) -> list[tuple[str, ...]]:
        base = _strip_prefix(fold(value), self.field)
        if self.exact:
            return [tuple(base.split())] if base else []
        own = sorted((fold(x) for x in (district, province) if x), key=len, reverse=True)
        keys = [base, _strip_suffix(base, own), _strip_suffix(base, self._places)]
        return list(dict.fromkeys(tuple(k.split()) for k in keys if k))

    def resolve(self, value: str, district: str | None = None, province: str | None = None,
                limit: int = MAX_CANDIDATES) -> list[str]:
        """Tên thật khớp `value`, xếp hạng: khoá bằng nhau → tên chứa câu hỏi → tên nằm trong câu hỏi;
        trong mỗi hạng, tin cùng quận/tỉnh của bộ lọc trước, rồi nhiều tin trước. Không có ⇒ []."""
        queries = self._query_keys(value, district, province)
        ids = {i for q in queries for t in q for i in self._by_token.get(t, ())}
        ranked: dict[str, tuple[int, int, int]] = {}
        for i in ids:
            e = self._entries[i]
            tier = None
            for q in queries:
                if e.key == q:
                    tier = 0
                elif self.exact:
                    continue
                elif _contains(e.key, q):
                    tier = 1 if tier is None else min(tier, 1)
                elif (_CONTAINED_OK[self.field] and _contains(q, e.key)
                      and not set(e.key) <= _GENERIC):
                    tier = 2 if tier is None else min(tier, 2)
            if tier is None:
                continue
            outside = int((district is not None and e.district != district)
                          or (province is not None and e.province != province))
            rank = (tier, outside, -e.n)
            if e.name not in ranked or rank < ranked[e.name]:
                ranked[e.name] = rank
        return sorted(ranked, key=lambda name: (ranked[name], name))[:limit]


# ---- từ vựng: dựng lười từ bảng sạch, cache parquet; test tiêm từ vựng riêng

_lock = threading.Lock()
_override: dict[str, NameIndex | None] = {}


def vocab_frame(field: str) -> pd.DataFrame | None:
    """(name, district, province, n) của trường `field` từ bảng sạch; cache parquet. Không có bảng sạch ⇒ None."""
    from vre.config import get_settings

    settings = get_settings()
    source = settings.clean_path
    cache = settings.cache_dir / f"name_vocab_{field}.parquet"
    if cache.exists() and (not source.exists() or cache.stat().st_mtime >= source.stat().st_mtime):
        return pd.read_parquet(cache)
    if not source.exists():
        return None
    from vre.data.duck import connect, sql_path

    con = connect(settings, views=False)
    try:
        con.execute("SET memory_limit = '1GB'")  # chỉ đọc 3 cột, vài nghìn nhóm
        frame = con.execute(
            f"SELECT {field} AS name, district, province, count(*) AS n FROM read_parquet('{sql_path(source)}') "
            f"WHERE {field} IS NOT NULL GROUP BY ALL").df()
    finally:
        con.close()
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    log.info("dựng từ vựng %s: %d dòng → %s", field, len(frame), cache)
    return frame


@lru_cache(maxsize=len(NAME_FIELDS + PLACE_FIELDS))
def _loaded(field: str) -> NameIndex | None:
    frame = vocab_frame(field)
    if frame is None:
        log.warning("không có bảng sạch để dựng từ vựng %s: bộ lọc %s so khớp đúng chuỗi", field, field)
        return None
    return NameIndex(field, frame)


def name_index(field: str) -> NameIndex | None:
    with _lock:
        if field in _override:
            return _override[field]
        return _loaded(field)


def use_vocabulary(field: str, vocab: pd.DataFrame | None) -> None:
    """Tiêm từ vựng (test/smoke); None ⇒ coi như không có từ vựng (so khớp đúng chuỗi)."""
    with _lock:
        _override[field] = None if vocab is None else NameIndex(field, vocab)
    resolve.cache_clear()


def reset() -> None:
    with _lock:
        _override.clear()
    _loaded.cache_clear()
    resolve.cache_clear()


@lru_cache(maxsize=1024)
def resolve(field: str, value: str, district: str | None = None, province: str | None = None) -> tuple[str, ...] | None:
    """Tên thật cho `value` (≤ MAX_CANDIDATES). None ⇒ không có từ vựng (người gọi so khớp đúng chuỗi);
    () ⇒ không tên nào khớp (người gọi bỏ điều kiện này và báo trong `note`)."""
    index = name_index(field)
    if index is None:
        return None
    return tuple(index.resolve(value, district, province))
