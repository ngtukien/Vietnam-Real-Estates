"""Pydantic models = OpenAPI `components.schemas` (api-contract §8). Tên class là hợp đồng với web — đổi thì sửa
api-contract trước, rồi `python -m api.export_openapi`.

Quy ước: trường `T | None` không có mặc định ⇒ luôn có mặt trong response (giá trị `null` khi thiếu dữ liệu).
Trường request có mặc định ⇒ tuỳ chọn (`T?` trong hợp đồng). Request cấm khoá lạ (`extra="forbid"`).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

MAX_QUESTION_CHARS = 500
MONTH_PATTERN = r"^\d{4}-(0[1-9]|1[0-2])$"
MAX_DESCRIPTION_EXCERPT = 400

Month = Annotated[str, StringConstraints(pattern=MONTH_PATTERN)]
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Scalar = str | bool | int | float | None


# ---------------------------------------------------------------- enums (§1)
class SystemId(StrEnum):
    LLM_ONLY = "llm_only"
    BASIC_RAG = "basic_rag"
    GRAPH = "graph"
    HYBRID = "hybrid"


class Route(StrEnum):
    BASIC_RAG = "basic_rag"
    GRAPH = "graph"
    HYBRID = "hybrid"
    VALUATION = "valuation"


class PropertyType(StrEnum):
    NHA = "Nhà"
    DAT = "Đất"
    CAN_HO = "Căn hộ chung cư"
    BIET_THU = "Biệt thự/Nhà liền kề"
    SHOPHOUSE = "Shophouse"


class DealType(StrEnum):
    SALE = "sale"
    RENT = "rent"


class ErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    UNAUTHORIZED = "unauthorized"
    NOT_FOUND = "not_found"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    RATE_LIMITED = "rate_limited"
    BUSY = "busy"
    SYSTEM_UNAVAILABLE = "system_unavailable"
    UPSTREAM_LLM_ERROR = "upstream_llm_error"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


class NodeLabel(StrEnum):
    LISTING = "Listing"
    PROPERTY_TYPE = "PropertyType"
    PROVINCE = "Province"
    DISTRICT = "District"
    WARD = "Ward"
    STREET = "Street"
    PROJECT = "Project"
    MONTH = "Month"
    AMENITY = "Amenity"
    CLUSTER = "Cluster"
    MARKET_STAT = "MarketStat"


class RelType(StrEnum):
    OF_TYPE = "OF_TYPE"
    IN_WARD = "IN_WARD"
    IN_DISTRICT = "IN_DISTRICT"
    IN_PROVINCE = "IN_PROVINCE"
    ON_STREET = "ON_STREET"
    IN_PROJECT = "IN_PROJECT"
    POSTED_IN = "POSTED_IN"
    HAS_AMENITY = "HAS_AMENITY"
    IN_CLUSTER = "IN_CLUSTER"
    STAT_OF = "STAT_OF"


class GeoLevel(StrEnum):
    PROVINCE = "province"
    DISTRICT = "district"
    WARD = "ward"


class BenchmarkCategory(StrEnum):
    LOOKUP = "lookup"
    CONSTRAINT = "constraint"
    AGGREGATE = "aggregate"
    MULTIHOP = "multihop"
    TEMPORAL = "temporal"
    COMPARISON = "comparison"
    PROJECT_STREET = "project_street"
    VALUATION = "valuation"
    UNANSWERABLE = "unanswerable"


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- kiểu chung (§2)
class QueryFilters(BaseModel):
    """Mọi trường nullable; null = không lọc. Dùng cả trong request (khoá vắng = null) và payload SSE."""

    model_config = ConfigDict(extra="forbid")

    province: str | None = None
    district: str | None = None
    ward: str | None = None
    property_type: PropertyType | None = None
    deal_type: DealType | None = None
    bedrooms: Annotated[int, Field(ge=0, le=100)] | None = None
    min_price_mil: Annotated[float, Field(ge=0)] | None = None
    max_price_mil: Annotated[float, Field(ge=0)] | None = None
    min_area_m2: Annotated[float, Field(ge=0)] | None = None
    max_area_m2: Annotated[float, Field(ge=0)] | None = None
    project: str | None = None
    street: str | None = None
    month_from: Month | None = None
    month_to: Month | None = None


class Address(_Response):
    street: str | None
    ward: str | None
    district: str | None
    province: str | None


class ListingCard(_Response):
    listing_id: int
    dup_cluster_id: int
    title: str | None
    property_type: PropertyType
    deal_type: DealType
    price_vnd: float | None
    price_m2_mil: float | None
    area_m2: float | None
    bedrooms: int | None
    bathrooms: int | None
    address: Address
    project: str | None
    published_month: str | None
    description_excerpt: Annotated[str, Field(max_length=MAX_DESCRIPTION_EXCERPT)]
    amenities: list[str]
    score: float | None
    cited: bool


class ListingDetail(ListingCard):
    description: str | None
    floors: int | None
    frontage_m: float | None
    road_width_m: float | None
    house_direction: str | None
    balcony_direction: str | None
    published_at: str | None
    n_raw_rows: int
    similar_in_cluster: int


class CypherResult(_Response):
    cypher: str
    columns: list[str]
    rows: Annotated[list[dict[str, Scalar]], Field(max_length=50)]
    error: str | None
    attempts: Annotated[int, Field(ge=1, le=3)]


class ShapItem(_Response):
    feature: str
    value: str | float | None
    contribution_log: float


class ErrorBody(_Response):
    code: ErrorCode
    message: str
    request_id: str
    details: dict[str, Any] | None


class ErrorEnvelope(_Response):
    error: ErrorBody


# ---------------------------------------------------------------- request
class ChatRequest(_Request):
    question: Question
    system: SystemId = SystemId.HYBRID
    filters: QueryFilters | None = None
    top_k: Annotated[int, Field(ge=1, le=20)] = 5
    use_rerank: bool = False
    stream_tokens: bool = True


class CompareRequest(_Request):
    question: Question
    systems: Annotated[list[SystemId], Field(min_length=1, max_length=4)] = Field(default_factory=lambda: list(SystemId))
    filters: QueryFilters | None = None
    top_k: Annotated[int, Field(ge=1, le=20)] = 5

    @field_validator("systems")
    @classmethod
    def _unique(cls, value: list[SystemId]) -> list[SystemId]:
        if len(set(value)) != len(value):
            raise ValueError("systems không được trùng")
        return value


class ValuationInput(_Request):
    property_type: PropertyType
    deal_type: DealType = DealType.SALE
    province: NonEmpty
    district: NonEmpty
    ward: str | None = None
    area_m2: Annotated[float, Field(gt=0, le=100_000)]
    bedrooms: Annotated[int, Field(ge=0, le=100)] | None = None
    bathrooms: Annotated[int, Field(ge=0, le=100)] | None = None
    floors: Annotated[int, Field(ge=0, le=200)] | None = None
    frontage_m: Annotated[float, Field(gt=0, le=1000)] | None = None
    road_width_m: Annotated[float, Field(gt=0, le=1000)] | None = None
    house_direction: str | None = None
    month: Month | None = None
    amenities: Annotated[list[str], Field(max_length=50)] = Field(default_factory=list)
    asking_price_vnd: Annotated[float, Field(gt=0)] | None = None


# ---------------------------------------------------------------- response REST (§3)
class HealthResponse(_Response):
    status: Literal["ok", "degraded"]
    qdrant: bool
    neo4j: bool
    llm_configured: bool
    version: str


class SystemInfo(_Response):
    id: SystemId
    name: str
    available: bool
    unavailable_reason: str | None


class IndexInfo(_Response):
    collection: str
    model_slug: str
    n_listings: int
    n_points: int


class DataPeriod(_Response):
    min_month: str
    max_month: str


class Limits(_Response):
    max_question_chars: int
    rate_per_min: int


class InfoResponse(_Response):
    systems: list[SystemInfo]
    index: IndexInfo
    graph_ok: bool
    data_period: DataPeriod
    limits: Limits


class FilterOptions(_Response):
    provinces: list[str]
    districts: dict[str, list[str]]
    property_types: list[PropertyType]
    months: list[str]


class PriceInterval(_Response):
    low: float
    high: float


class ModelInfo(_Response):
    variant: Literal["full", "tabular"]
    path: str
    trained_at: str
    test_mape: float


class ValuationResult(_Response):
    predicted_price_m2_mil: float
    predicted_price_vnd: float
    interval_price_vnd: PriceInterval
    baseline_ward_median_price_m2_mil: float | None
    comparables_n: int
    verdict: Literal["below_market", "fair", "above_market"] | None
    shap: Annotated[list[ShapItem], Field(max_length=8)]
    model: ModelInfo


class TypeStat(_Response):
    property_type: PropertyType
    n: int
    median_price_m2_mil: float | None


class ProvinceStat(_Response):
    province: str
    n: int
    median_price_m2_mil: float | None


class StatsOverview(_Response):
    n_listings: int
    n_raw_rows: int
    n_duplicates_removed: int
    by_property_type: list[TypeStat]
    by_province_top: list[ProvinceStat]


class MarketScope(_Response):
    province: str
    district: str | None
    property_type: PropertyType | None
    deal_type: DealType


class MonthStat(_Response):
    month: str
    n: int
    median_price_m2_mil: float | None


class AreaStat(_Response):
    key: str
    name: str
    n: int
    median_price_m2_mil: float | None


class MarketStats(_Response):
    scope: MarketScope
    monthly: list[MonthStat]
    by_area: list[AreaStat]


class GeoItem(_Response):
    key: str
    name: str
    lat: float | None
    lon: float | None
    n_listings: int
    median_price_m2_mil: float | None


class GeoAreas(_Response):
    level: GeoLevel
    items: list[GeoItem]
    coverage: Annotated[float, Field(ge=0, le=1)]


class GraphNode(_Response):
    id: str
    label: NodeLabel
    name: str
    props: dict[str, Scalar]


class GraphEdge(_Response):
    source: str
    target: str
    type: RelType


class Subgraph(_Response):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool


class BenchmarkResponse(BaseModel):
    """Nội dung `results/eval_summary.json` + `ablation`, `embedding_benchmark` (object tự do, data-contract §7.7)."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------- payload SSE (§4) — đã gồm request_id, system
class _Event(_Response):
    request_id: str
    system: SystemId


class EventMeta(_Event):
    question: str
    started_at: str


class EventRoute(_Event):
    route: Route
    reason: str


class EventFilters(_Event):
    filters: QueryFilters
    source: Literal["llm", "manual", "merged"]
    note: str | None


class EventCypher(CypherResult):
    request_id: str
    system: SystemId


class EventValuation(ValuationResult):
    request_id: str
    system: SystemId


class EventSources(_Event):
    sources: list[ListingCard]


class EventToken(_Event):
    text: str


class EventAnswer(_Event):
    answer: str
    cited: list[int]
    unsupported_citations: list[int]
    fallback: bool
    route: Route | None


class EventUsage(_Event):
    latency_ms: float
    input_tokens: int
    output_tokens: int
    usd: float
    cached: bool


class EventError(_Event):
    code: ErrorCode
    message: str


class EventDone(_Event):
    status: Literal["ok", "error"]


class EventCompareDone(_Response):
    request_id: str
    systems_ok: list[SystemId]
    systems_failed: list[SystemId]


EVENT_MODELS: dict[str, type[BaseModel]] = {
    "meta": EventMeta,
    "route": EventRoute,
    "filters": EventFilters,
    "cypher": EventCypher,
    "valuation": EventValuation,
    "sources": EventSources,
    "token": EventToken,
    "answer": EventAnswer,
    "usage": EventUsage,
    "error": EventError,
    "done": EventDone,
}

# Model không gắn vào route nào nhưng phải có trong components.schemas (api-contract §8).
EXTRA_SCHEMA_MODELS: tuple[type[BaseModel], ...] = (
    *EVENT_MODELS.values(), EventCompareDone, ErrorEnvelope, ChatRequest, CompareRequest, CypherResult, ListingCard,
)
EXTRA_SCHEMA_ENUMS: tuple[type[StrEnum], ...] = (
    SystemId, Route, PropertyType, DealType, ErrorCode, NodeLabel, RelType, GeoLevel, BenchmarkCategory,
)
