"""Retrieval và Generation (Notebook 2). Hợp đồng: search_dense, search_bm25, rrf, ask_rag."""

import re
from time import perf_counter

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src import qdrant_store
from src.common import fmt_vnd
from src.config import RERANK_MODEL, RRF_K, TOP_K
from src.data import normalize_place
from src.llm import LLMResult, ask_llm
from src.vector import embed_query


# ---------------------------------------------------------------- nạp chỉ mục (P2-01)


def tokenize_vi(text: str) -> list[str]:
    """Tách từ tiếng Việt bằng pyvi: 'trường học' -> 'trường_học'; bỏ dấu câu."""
    from pyvi import ViTokenizer

    return [t for t in ViTokenizer.tokenize(text.lower()).split() if re.search(r"\w", t)]


# ---------------------------------------------------------------- hiểu câu hỏi (P2-02)


def parse_money_mil(value) -> float | None:
    """Đổi tiền về triệu đồng: '5 tỷ' -> 5000; '1 tỷ 2' -> 1200; '850 triệu' -> 850."""
    if value is None or isinstance(value, (int, float)):
        return value
    text = str(value).lower().replace(",", ".").strip()
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(tỷ|ty)\s*(\d+)?", text)
    if match:
        extra = match.group(3)
        return float(match.group(1)) * 1000 + (float(extra) * 10 ** (3 - len(extra)) if extra else 0)
    match = re.search(r"(\d+(?:\.\d+)?)\s*(triệu|tr)", text)
    if match:
        return float(match.group(1))
    return float(re.search(r"\d+(?:\.\d+)?", text).group())


class ExtractedFilters(BaseModel):
    """Đầu ra LLM: giữ nguyên cách viết tiền trong câu hỏi ('5 tỷ'), code sẽ đổi đơn vị."""

    province: str | None = Field(None, description="Tỉnh/thành, ví dụ 'Hồ Chí Minh', 'Hà Nội'")
    district: str | None = Field(None, description="Quận/huyện/thành phố thuộc tỉnh, ví dụ 'Thủ Đức', 'Quận 7'")
    property_type: str | None = Field(None, description="Một trong: Căn hộ chung cư, Nhà, Đất, Biệt thự/Nhà liền kề, Shophouse")
    bedrooms: int | None = Field(None, description="Số phòng ngủ; '2PN' -> 2")
    min_price: str | None = Field(None, description="Giá tối thiểu đúng như câu hỏi viết, ví dụ '3 tỷ'")
    max_price: str | None = Field(None, description="Giá tối đa đúng như câu hỏi viết, ví dụ '5 tỷ'")
    min_area: float | None = Field(None, description="Diện tích tối thiểu, m²")
    max_area: float | None = Field(None, description="Diện tích tối đa, m²")


class QueryFilters(BaseModel):
    """Bộ lọc đã kiểm tra: tiền theo triệu đồng, địa danh đã chuẩn hoá như dữ liệu sạch."""

    province: str | None = None
    district: str | None = None
    property_type: str | None = None
    bedrooms: int | None = None
    min_price_mil: float | None = None
    max_price_mil: float | None = None
    min_area: float | None = None
    max_area: float | None = None

    @field_validator("min_price_mil", "max_price_mil", mode="before")
    @classmethod
    def _money(cls, value):
        return parse_money_mil(value)

    @field_validator("province", "district", mode="before")
    @classmethod
    def _place(cls, value, info):
        return normalize_place(value, info.field_name)

    @classmethod
    def from_extracted(cls, raw: dict) -> "QueryFilters":
        return cls.model_validate({**raw, "min_price_mil": raw.get("min_price"),
                                   "max_price_mil": raw.get("max_price")})


FILTER_PROMPT = """Trích bộ lọc tìm kiếm bất động sản từ câu hỏi. Chỉ điền trường được nói rõ trong câu hỏi,
còn lại để null. Không suy đoán tỉnh từ tên quận nếu câu hỏi không nói.

Câu hỏi: {question}"""


def parse_filters(question: str) -> tuple[QueryFilters, LLMResult]:
    result = ask_llm(FILTER_PROMPT.format(question=question), output=ExtractedFilters)
    return QueryFilters.from_extracted(result.data), result


# ---------------------------------------------------------------- lọc


def matches(df: pd.DataFrame, filters: QueryFilters | None) -> pd.Series:
    """Mặt nạ pandas tương đương bộ lọc Qdrant (qdrant_store.to_filter), dùng cho post-filter."""
    mask = pd.Series(True, index=df.index)
    if filters is None:
        return mask
    f = filters
    for key, value in (("province", f.province), ("district", f.district), ("property_type", f.property_type)):
        if value:
            mask &= df[key] == value
    if f.bedrooms is not None:
        mask &= df["bedrooms"] == f.bedrooms
    if f.min_price_mil is not None:
        mask &= df["price"] >= f.min_price_mil * 1e6
    if f.max_price_mil is not None:
        mask &= df["price"] <= f.max_price_mil * 1e6
    if f.min_area is not None:
        mask &= df["area"] >= f.min_area
    if f.max_area is not None:
        mask &= df["area"] <= f.max_area
    return mask.fillna(False)


# ---------------------------------------------------------------- tìm kiếm


def cosine(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _by_listing(hits: pd.DataFrame, k: int) -> pd.DataFrame:
    """Mỗi tin chỉ giữ chunk điểm cao nhất, để một tin dài không chiếm hết top-k."""
    hits = hits.sort_values("score", ascending=False).drop_duplicates("listing_id")
    return hits.head(k).reset_index(drop=True)


def search_dense(question: str, k: int = TOP_K, filters: QueryFilters | None = None,
                 threshold: float | None = None) -> pd.DataFrame:
    """Dense search trên Qdrant, có pre-filter theo payload và ngưỡng cosine tuỳ chọn."""
    hits = qdrant_store.search(embed_query(question).tolist(), "dense", k * 4, filters)
    if threshold is not None:
        hits = hits[hits["score"] >= threshold]
    return _by_listing(hits, k)


def search_bm25(question: str, k: int = TOP_K, filters: QueryFilters | None = None) -> pd.DataFrame:
    """BM25 trên Qdrant (sparse vector, IDF tính phía server) với từ đã tách bằng pyvi;
    khớp đúng từ như 'sổ hồng', '2PN'."""
    hits = qdrant_store.search(qdrant_store.bm25_query(tokenize_vi(question)), "bm25", k * 4, filters)
    return _by_listing(hits, k)


def rrf(rankings: list[list], k: int = RRF_K) -> list[tuple]:
    """rankings: danh sách các bảng xếp hạng, mỗi bảng là list ID theo thứ tự tốt dần (Phụ lục A.3)."""
    scores = {}
    for ranking in rankings:
        for pos, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0) + 1 / (k + pos)
    return sorted(scores.items(), key=lambda x: -x[1])


def search_hybrid(question: str, k: int = TOP_K, filters: QueryFilters | None = None,
                  candidates: int = 20) -> pd.DataFrame:
    """Gộp dense và BM25 bằng RRF theo listing_id; giữ chunk tốt nhất của mỗi tin."""
    dense = search_dense(question, candidates, filters)
    bm25 = search_bm25(question, candidates, filters)
    fused = rrf([dense["listing_id"].tolist(), bm25["listing_id"].tolist()])[:k]
    pool = pd.concat([dense, bm25]).sort_values("score", ascending=False).drop_duplicates("listing_id")
    pool = pool.set_index("listing_id")
    return pd.DataFrame([{**pool.loc[lid].to_dict(), "listing_id": lid, "rrf": score}
                         for lid, score in fused])


def hits_table(hits: pd.DataFrame, score: str = "score", width: int = 70) -> pd.DataFrame:
    """Bảng kết quả gọn cho output trên lớp: ID, điểm, giá, diện tích, địa chỉ, đầu đoạn văn."""
    return pd.DataFrame({
        "listing_id": hits["listing_id"].astype(int),
        "điểm": hits[score].round(4),
        "giá": hits["price"].map(fmt_vnd),
        "m²": hits["area"],
        "địa chỉ": hits[["ward", "district", "province"]].apply(
            lambda r: ", ".join(x for x in r if isinstance(x, str)), axis=1),
        "đoạn": hits["text"].str.replace("\n", " · ").str.slice(0, width),
    })


def post_filter(hits: pd.DataFrame, filters: QueryFilters) -> pd.DataFrame:
    return hits[matches(hits, filters)].reset_index(drop=True)


_reranker = None


def rerank(question: str, hits: pd.DataFrame, top: int = TOP_K) -> pd.DataFrame:
    """Cross-encoder chấm lại từng cặp (câu hỏi, chunk): chậm hơn nhưng chính xác hơn bi-encoder."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL)
    scores = _reranker.predict([(question, text) for text in hits["text"]])
    return hits.assign(rerank=scores).nlargest(top, "rerank").reset_index(drop=True)


# ---------------------------------------------------------------- sinh câu trả lời (P2-09, P2-10)

ANSWER_SYSTEM = """Bạn là trợ lý thị trường bất động sản Việt Nam. Chỉ trả lời từ NGỮ CẢNH được cung cấp.
Mỗi nhận định phải kèm nguồn dạng [Tin#ID]. Nếu ngữ cảnh không đủ để trả lời hoặc để tính toán, nói rõ là
không đủ dữ liệu thay vì đoán. Giá ghi theo tỷ hoặc triệu đồng. Trả lời ngắn gọn, tiếng Việt."""


def format_context(hits: pd.DataFrame) -> list[str]:
    return [f"[Tin#{int(row.listing_id)}] {row.text}" for row in hits.itertuples()]


def build_prompt(question: str, contexts: list[str]) -> str:
    return "NGỮ CẢNH:\n" + "\n\n".join(contexts) + f"\n\nCÂU HỎI: {question}"


def generate(question: str, contexts: list[str]) -> LLMResult:
    return ask_llm(build_prompt(question, contexts), system=ANSWER_SYSTEM)


def cited_ids(answer: str) -> list[int]:
    return [int(x) for x in dict.fromkeys(re.findall(r"Tin#(\d+)", answer))]


def retrieve(question: str, k: int = TOP_K, use_filters: bool = True) -> dict:
    """Bộ lọc từ LLM -> hybrid search có pre-filter. Nếu bộ lọc quá chặt (0 kết quả), bỏ lọc
    và ghi lại trong 'note'."""
    filters, parsed = parse_filters(question) if use_filters else (None, None)
    hits, note = search_hybrid(question, k, filters), ""
    if hits.empty and filters is not None:
        hits, note = search_hybrid(question, k), "bộ lọc không có kết quả, đã bỏ lọc"
    return dict(hits=hits, filters=filters, note=note, calls=[parsed] if parsed else [])


def ask_rag(question: str, k: int = TOP_K, use_filters: bool = True) -> dict:
    """Basic RAG: retrieve -> prompt (chỉ thị, ngữ cảnh, câu hỏi) -> câu trả lời có trích dẫn [Tin#ID]."""
    start = perf_counter()
    found = retrieve(question, k, use_filters)
    contexts = format_context(found["hits"])
    answer = generate(question, contexts)
    calls = found["calls"] + [answer]
    filters = found["filters"]
    return dict(system="basic_rag", question=question, answer=answer.text,
                sources=found["hits"]["listing_id"].astype(int).tolist(), contexts=contexts,
                filters=filters.model_dump(exclude_none=True) if filters else {}, note=found["note"],
                latency=perf_counter() - start,
                input_tokens=sum(r.input_tokens for r in calls),
                output_tokens=sum(r.output_tokens for r in calls))
