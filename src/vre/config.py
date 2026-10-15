"""Cấu hình dự án — SSOT là data-contract §9.1. Thêm/đổi khoá ⇒ sửa §9.1 trước rồi báo phiên p2.

Biến môi trường mang tiền tố `VRE_` (vd `VRE_DUCKDB_THREADS=8`). Tham số LLM/RAG/chat/Neo4j giữ tên env cũ
(`GROQ_API_KEY`, `LLM_MODEL`, `NEO4J_PASSWORD`…). `.env` dưới `data_root` chỉ được đọc khi `VRE_LOAD_DOTENV != "0"`.
Mọi đường dẫn dữ liệu là tuyệt đối, tính từ `data_root` (gốc DỮ LIỆU, không phải gốc code — hai worktree, một bộ dữ liệu).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DATA_ROOT = Path(r"D:\Vietnam-Real-Estates")

# slug: (tên HF, số chiều, pooling, tiền tố tài liệu, tiền tố truy vấn). Một model duy nhất, đã chốt 2026-10-15.
EMBED_MODELS: dict[str, tuple[str, int, str, str, str]] = {
    "e5_large": ("intfloat/multilingual-e5-large", 1024, "mean", "passage: ", "query: "),
}
EMBED_MODEL_SLUG = "e5_large"

# provider: (model sinh, model dự phòng khi 429/503, model chấm, giá USD/1M token vào, ra)
LLM_DEFAULTS: dict[str, tuple[str, str, str | None, float, float]] = {
    "groq": ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-20b", 0.15, 0.60),
    "openai": ("gpt-5.4-mini", "", None, 4.00, 20.00),
    "anthropic": ("claude-opus-5-5", "", None, 4.00, 20.00),
    # Gemini qua endpoint tương thích OpenAI, gói miễn phí (0 USD); không có giám khảo mặc định (eval chấm theo luật).
    # Không có model dự phòng: `gemini-2.5-flash-lite` trả 404 "no longer available to new users" (đo 2026-10-16).
    "gemini": ("gemini-flash-lite-latest", "", None, 0.0, 0.0),
}


def _env(*names: str) -> AliasChoices:
    """Đọc từ `VRE_<NAME>` hoặc tên env cũ (giữ tương thích `.env` hiện có)."""
    return AliasChoices(*(f"VRE_{n}" for n in names), *names)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VRE_", extra="ignore", populate_by_name=True)

    data_root: Path = DEFAULT_DATA_ROOT

    duckdb_threads: int = 12
    duckdb_memory_gb: int = 12
    ingest_workers: int = 8

    chunk_size: int = 400
    chunk_overlap: int = 100
    max_chunks_per_listing: int = 64
    header_max_tokens: int = 96
    part_size: int = 50000

    embed_max_length: int = 512
    embed_token_budget: int = 32768
    embed_max_batch: int = 256
    embed_min_token_budget: int = 4096

    tokenize_workers: int = 2  # commit là giới hạn; GPU vẫn là nút cổ chai (CR-04)
    tokenize_prefetch_parts: int = 3
    bm25_workers: int = 4  # CR-04

    qdrant_url: str = Field("http://127.0.0.1:6333", validation_alias=_env("QDRANT_URL"))
    qdrant_grpc_port: int = 6334
    qdrant_upload_parallel: int = 1  # tiến trình upload riêng, wait=True: 1 luồng gRPC đủ (CR-02)
    qdrant_upload_batch: int = 256
    qdrant_max_indexing_threads: int = 12
    qdrant_indexing_threshold: int = 20000
    qdrant_green_timeout_s: int = 10800

    collection: str = "vre_chunks_e5_large"
    collection_alias: str = "vre_chunks"
    dryrun_collection: str = "vre_dryrun"
    dryrun_parts: int = 2

    neo4j_uri: str = Field("bolt://127.0.0.1:7687", validation_alias=_env("NEO4J_URI"))
    neo4j_user: str = Field("neo4j", validation_alias=_env("NEO4J_USER"))
    neo4j_password: str = Field("local_dev_password", validation_alias=_env("NEO4J_PASSWORD"))

    xgb_nthread: int = 12
    xgb_device_tabular: str = "cpu"
    xgb_device_full: str = "cuda"
    rent_min_rows: int = 20000

    pca_dim: int = 32
    pca_fit_sample: int = 300000

    min_free_disk_gb_start: int = 45
    min_free_disk_gb_part: int = 5

    night_max_attempts: int = 3
    night_retry_wait_s: int = 120
    handoff_poll_s: int = 300

    # ---- tham số LLM/RAG/chat cũ (Phase 3 dùng), giữ tên env cũ
    seed: int = Field(42, validation_alias=_env("SEED"))
    llm_provider: str = Field("groq", validation_alias=_env("LLM_PROVIDER"))
    llm_model: str = Field("", validation_alias=_env("LLM_MODEL"))
    llm_fallback_models: str | None = Field(None, validation_alias=_env("LLM_FALLBACK_MODELS"))
    llm_effort: str = Field("low", validation_alias=_env("LLM_EFFORT"))
    llm_timeout: float = Field(120.0, validation_alias=_env("LLM_TIMEOUT"))
    judge_model: str = Field("", validation_alias=_env("JUDGE_MODEL"))
    llm_price_input: float | None = Field(None, validation_alias=_env("LLM_PRICE_INPUT"))
    llm_price_output: float | None = Field(None, validation_alias=_env("LLM_PRICE_OUTPUT"))
    groq_api_key: str = Field("", validation_alias=_env("GROQ_API_KEY"))
    openai_api_key: str = Field("", validation_alias=_env("OPENAI_API_KEY"))
    anthropic_api_key: str = Field("", validation_alias=_env("ANTHROPIC_API_KEY"))
    gemini_api_key: str = Field("", validation_alias=_env("GEMINI_API_KEY"))
    rerank: bool = Field(False, validation_alias=_env("RERANK"))
    rerank_model: str = Field("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", validation_alias=_env("RERANK_MODEL"))
    rerank_candidates: int = Field(20, validation_alias=_env("RERANK_CANDIDATES"))
    qdrant_oversampling: float = Field(2.0, validation_alias=_env("QDRANT_OVERSAMPLING"))
    rrf_k: int = Field(60, validation_alias=_env("RRF_K"))
    top_k: int = Field(5, validation_alias=_env("TOP_K"))
    cypher_timeout: float = Field(10.0, validation_alias=_env("CYPHER_TIMEOUT"))
    cypher_max_rows: int = Field(50, validation_alias=_env("CYPHER_MAX_ROWS"))
    chat_max_concurrent: int = Field(4, validation_alias=_env("CHAT_MAX_CONCURRENT"))
    chat_rate_per_min: int = Field(20, validation_alias=_env("CHAT_RATE_PER_MIN"))
    chat_max_query_chars: int = Field(500, validation_alias=_env("CHAT_MAX_QUERY_CHARS"))
    chat_api_token: str = Field("", validation_alias=_env("CHAT_API_TOKEN"))

    @model_validator(mode="after")
    def _llm_defaults(self) -> Settings:
        model, fallback, judge, price_in, price_out = LLM_DEFAULTS.get(self.llm_provider, LLM_DEFAULTS["groq"])
        self.llm_model = self.llm_model or model
        self.judge_model = self.judge_model or judge or self.llm_model
        if self.llm_fallback_models is None:
            self.llm_fallback_models = fallback
        self.llm_price_input = price_in if self.llm_price_input is None else self.llm_price_input
        self.llm_price_output = price_out if self.llm_price_output is None else self.llm_price_output
        return self

    # ---- đường dẫn (tuyệt đối, dưới data_root)
    @property
    def raw_dir(self) -> Path:
        return self.data_root / "data" / "raw" / "vietnam-real-estates"

    @property
    def clean_path(self) -> Path:
        return self.data_root / "data" / "clean" / "listings.parquet"

    @property
    def dup_map_path(self) -> Path:
        return self.data_root / "data" / "clean" / "dup_map.parquet"

    @property
    def index_dir(self) -> Path:
        return self.data_root / "index"

    @property
    def results_dir(self) -> Path:
        return self.data_root / "results"

    @property
    def models_dir(self) -> Path:
        return self.data_root / "models"

    @property
    def logs_dir(self) -> Path:
        return self.data_root / "logs"

    @property
    def handoff_dir(self) -> Path:
        return self.data_root / "results" / "handoff"

    @property
    def features_dir(self) -> Path:
        return self.data_root / "data" / "features"

    @property
    def cache_dir(self) -> Path:
        return self.data_root / ".cache"

    @property
    def compose_file(self) -> Path:
        return self.data_root / "docker-compose.yaml"

    @property
    def llm_fallback_list(self) -> list[str]:
        return [m.strip() for m in (self.llm_fallback_models or "").split(",") if m.strip()]

    def fallbacks_for(self, model: str) -> list[str]:
        """Model dự phòng (429/5xx) cho `model`: bỏ chính nó và bỏ model "phía bên kia" — sinh câu trả lời không
        bao giờ lui về `judge_model` (giám khảo sẽ chấm câu do chính nó viết), giám khảo không lui về `llm_model`."""
        other = self.llm_model if model == self.judge_model else self.judge_model
        return [m for m in self.llm_fallback_list if m not in (model, other)]


def dotenv_enabled() -> bool:
    return os.environ.get("VRE_LOAD_DOTENV", "1") != "0"


def load_settings(**overrides: object) -> Settings:
    """Tạo Settings mới; `.env` dưới data_root chỉ đọc khi `VRE_LOAD_DOTENV != "0"`."""
    root = Path(os.environ.get("VRE_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
    env_file = root / ".env" if dotenv_enabled() else None
    return Settings(_env_file=env_file, _env_file_encoding="utf-8", **overrides)  # type: ignore[call-arg, arg-type]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
