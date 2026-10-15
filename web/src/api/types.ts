/**
 * TẠM THỜI VIẾT TAY theo api-contract.md §1–§4, §8 (cùng cấu trúc `components["schemas"]` mà openapi-typescript sinh ra).
 * Khi Lane B có `api/openapi.json`: chạy `npm --prefix web run gen:types` để sinh lại file này rồi commit.
 * Không thêm tên trường ngoài hợp đồng.
 */

export type paths = Record<string, never>;
export type webhooks = Record<string, never>;
export type $defs = Record<string, never>;
export type operations = Record<string, never>;

type Nullable<T> = T | null;
type Scalar = string | number | boolean | null;

export interface components {
  schemas: {
    // ---- Enums §1 ----
    SystemId: "llm_only" | "basic_rag" | "graph" | "hybrid";
    Route: "basic_rag" | "graph" | "hybrid" | "valuation";
    PropertyType: "Nhà" | "Đất" | "Căn hộ chung cư" | "Biệt thự/Nhà liền kề" | "Shophouse";
    DealType: "sale" | "rent";
    ErrorCode:
      | "invalid_request"
      | "unauthorized"
      | "not_found"
      | "payload_too_large"
      | "unsupported_media_type"
      | "rate_limited"
      | "busy"
      | "system_unavailable"
      | "upstream_llm_error"
      | "timeout"
      | "internal_error";
    NodeLabel:
      | "Listing"
      | "PropertyType"
      | "Province"
      | "District"
      | "Ward"
      | "Street"
      | "Project"
      | "Month"
      | "Amenity"
      | "Cluster"
      | "MarketStat";
    RelType:
      | "OF_TYPE"
      | "IN_WARD"
      | "IN_DISTRICT"
      | "IN_PROVINCE"
      | "ON_STREET"
      | "IN_PROJECT"
      | "POSTED_IN"
      | "HAS_AMENITY"
      | "IN_CLUSTER"
      | "STAT_OF";
    GeoLevel: "province" | "district" | "ward";
    BenchmarkCategory:
      | "lookup"
      | "constraint"
      | "aggregate"
      | "multihop"
      | "temporal"
      | "comparison"
      | "project_street"
      | "valuation"
      | "unanswerable";

    // ---- Kiểu chung §2 ----
    QueryFilters: {
      province: Nullable<string>;
      district: Nullable<string>;
      ward: Nullable<string>;
      property_type: Nullable<components["schemas"]["PropertyType"]>;
      deal_type: Nullable<components["schemas"]["DealType"]>;
      bedrooms: Nullable<number>;
      min_price_mil: Nullable<number>;
      max_price_mil: Nullable<number>;
      min_area_m2: Nullable<number>;
      max_area_m2: Nullable<number>;
      project: Nullable<string>;
      street: Nullable<string>;
      month_from: Nullable<string>;
      month_to: Nullable<string>;
    };
    Address: {
      street: Nullable<string>;
      ward: Nullable<string>;
      district: Nullable<string>;
      province: Nullable<string>;
    };
    ListingCard: {
      listing_id: number;
      dup_cluster_id: number;
      title: Nullable<string>;
      property_type: components["schemas"]["PropertyType"];
      deal_type: components["schemas"]["DealType"];
      price_vnd: Nullable<number>;
      price_m2_mil: Nullable<number>;
      area_m2: Nullable<number>;
      bedrooms: Nullable<number>;
      bathrooms: Nullable<number>;
      address: components["schemas"]["Address"];
      project: Nullable<string>;
      published_month: Nullable<string>;
      description_excerpt: string;
      amenities: string[];
      score: Nullable<number>;
      cited: boolean;
    };
    ListingDetail: components["schemas"]["ListingCard"] & {
      description: Nullable<string>;
      floors: Nullable<number>;
      frontage_m: Nullable<number>;
      road_width_m: Nullable<number>;
      house_direction: Nullable<string>;
      balcony_direction: Nullable<string>;
      published_at: Nullable<string>;
      n_raw_rows: number;
      similar_in_cluster: number;
    };
    CypherResult: {
      cypher: string;
      columns: string[];
      rows: Record<string, Scalar>[];
      error: Nullable<string>;
      attempts: number;
    };
    ShapItem: { feature: string; value: string | number | null; contribution_log: number };
    ErrorBody: {
      code: components["schemas"]["ErrorCode"];
      message: string;
      request_id: string;
      details: Nullable<Record<string, unknown>>;
    };
    ErrorEnvelope: { error: components["schemas"]["ErrorBody"] };

    // ---- Request ----
    ChatRequest: {
      question: string;
      system?: components["schemas"]["SystemId"];
      filters?: Nullable<components["schemas"]["QueryFilters"]>;
      top_k?: number;
      use_rerank?: boolean;
      stream_tokens?: boolean;
    };
    CompareRequest: {
      question: string;
      systems?: components["schemas"]["SystemId"][];
      filters?: Nullable<components["schemas"]["QueryFilters"]>;
      top_k?: number;
    };
    ValuationInput: {
      property_type: components["schemas"]["PropertyType"];
      deal_type?: components["schemas"]["DealType"];
      province: string;
      district: string;
      ward?: Nullable<string>;
      area_m2: number;
      bedrooms?: Nullable<number>;
      bathrooms?: Nullable<number>;
      floors?: Nullable<number>;
      frontage_m?: Nullable<number>;
      road_width_m?: Nullable<number>;
      house_direction?: Nullable<string>;
      month?: Nullable<string>;
      amenities?: string[];
      asking_price_vnd?: Nullable<number>;
    };

    // ---- Response REST §3 ----
    HealthResponse: {
      status: "ok" | "degraded";
      qdrant: boolean;
      neo4j: boolean;
      llm_configured: boolean;
      version: string;
    };
    SystemInfo: {
      id: components["schemas"]["SystemId"];
      name: string;
      available: boolean;
      unavailable_reason: Nullable<string>;
    };
    IndexInfo: { collection: string; model_slug: string; n_listings: number; n_points: number };
    DataPeriod: { min_month: string; max_month: string };
    Limits: { max_question_chars: number; rate_per_min: number };
    InfoResponse: {
      systems: components["schemas"]["SystemInfo"][];
      index: components["schemas"]["IndexInfo"];
      graph_ok: boolean;
      data_period: components["schemas"]["DataPeriod"];
      limits: components["schemas"]["Limits"];
    };
    FilterOptions: {
      provinces: string[];
      districts: Record<string, string[]>;
      property_types: components["schemas"]["PropertyType"][];
      months: string[];
    };
    PriceInterval: { low: number; high: number };
    ModelInfo: { variant: "full" | "tabular"; path: string; trained_at: string; test_mape: number };
    ValuationResult: {
      predicted_price_m2_mil: number;
      predicted_price_vnd: number;
      interval_price_vnd: components["schemas"]["PriceInterval"];
      baseline_ward_median_price_m2_mil: Nullable<number>;
      comparables_n: number;
      verdict: Nullable<"below_market" | "fair" | "above_market">;
      shap: components["schemas"]["ShapItem"][];
      model: components["schemas"]["ModelInfo"];
    };
    TypeStat: {
      property_type: components["schemas"]["PropertyType"];
      n: number;
      median_price_m2_mil: Nullable<number>;
    };
    ProvinceStat: { province: string; n: number; median_price_m2_mil: Nullable<number> };
    StatsOverview: {
      n_listings: number;
      n_raw_rows: number;
      n_duplicates_removed: number;
      by_property_type: components["schemas"]["TypeStat"][];
      by_province_top: components["schemas"]["ProvinceStat"][];
    };
    MarketScope: {
      province: string;
      district: Nullable<string>;
      property_type: Nullable<components["schemas"]["PropertyType"]>;
      deal_type: components["schemas"]["DealType"];
    };
    MonthStat: { month: string; n: number; median_price_m2_mil: Nullable<number> };
    AreaStat: { key: string; name: string; n: number; median_price_m2_mil: Nullable<number> };
    MarketStats: {
      scope: components["schemas"]["MarketScope"];
      monthly: components["schemas"]["MonthStat"][];
      by_area: components["schemas"]["AreaStat"][];
    };
    GeoItem: {
      key: string;
      name: string;
      lat: Nullable<number>;
      lon: Nullable<number>;
      n_listings: number;
      median_price_m2_mil: Nullable<number>;
    };
    GeoAreas: {
      level: components["schemas"]["GeoLevel"];
      items: components["schemas"]["GeoItem"][];
      coverage: number;
    };
    GraphNode: {
      id: string;
      label: components["schemas"]["NodeLabel"];
      name: string;
      props: Record<string, Scalar>;
    };
    GraphEdge: { source: string; target: string; type: components["schemas"]["RelType"] };
    Subgraph: {
      nodes: components["schemas"]["GraphNode"][];
      edges: components["schemas"]["GraphEdge"][];
      truncated: boolean;
    };
    BenchmarkResponse: { [key: string]: unknown };

    // ---- Payload SSE §4 (đã gồm request_id, system) ----
    EventMeta: {
      request_id: string;
      system: components["schemas"]["SystemId"];
      question: string;
      started_at: string;
    };
    EventRoute: {
      request_id: string;
      system: components["schemas"]["SystemId"];
      route: components["schemas"]["Route"];
      reason: string;
    };
    EventFilters: {
      request_id: string;
      system: components["schemas"]["SystemId"];
      filters: components["schemas"]["QueryFilters"];
      source: "llm" | "manual" | "merged";
      note: Nullable<string>;
    };
    EventCypher: components["schemas"]["CypherResult"] & {
      request_id: string;
      system: components["schemas"]["SystemId"];
    };
    EventValuation: components["schemas"]["ValuationResult"] & {
      request_id: string;
      system: components["schemas"]["SystemId"];
    };
    EventSources: {
      request_id: string;
      system: components["schemas"]["SystemId"];
      sources: components["schemas"]["ListingCard"][];
    };
    EventToken: { request_id: string; system: components["schemas"]["SystemId"]; text: string };
    EventAnswer: {
      request_id: string;
      system: components["schemas"]["SystemId"];
      answer: string;
      cited: number[];
      unsupported_citations: number[];
      fallback: boolean;
      route: Nullable<components["schemas"]["Route"]>;
    };
    EventUsage: {
      request_id: string;
      system: components["schemas"]["SystemId"];
      latency_ms: number;
      input_tokens: number;
      output_tokens: number;
      usd: number;
      cached: boolean;
    };
    EventError: {
      request_id: string;
      system: components["schemas"]["SystemId"];
      code: components["schemas"]["ErrorCode"];
      message: string;
    };
    EventDone: {
      request_id: string;
      system: components["schemas"]["SystemId"];
      status: "ok" | "error";
    };
    EventCompareDone: {
      request_id: string;
      systems_ok: components["schemas"]["SystemId"][];
      systems_failed: components["schemas"]["SystemId"][];
    };
  };
  responses: never;
  parameters: never;
  requestBodies: never;
  headers: never;
  pathItems: never;
}
