import type { components } from "./types";

type S = components["schemas"];
export type SystemId = S["SystemId"];
export type Route = S["Route"];
export type PropertyType = S["PropertyType"];
export type DealType = S["DealType"];
export type ErrorCode = S["ErrorCode"];
export type QueryFilters = S["QueryFilters"];
export type ListingCard = S["ListingCard"];
export type ListingDetail = S["ListingDetail"];
export type CypherResult = S["CypherResult"];
export type ShapItem = S["ShapItem"];
export type ErrorEnvelope = S["ErrorEnvelope"];
export type ChatRequest = S["ChatRequest"];
export type CompareRequest = S["CompareRequest"];
export type ValuationInput = S["ValuationInput"];
export type ValuationResult = S["ValuationResult"];
export type HealthResponse = S["HealthResponse"];
export type InfoResponse = S["InfoResponse"];
export type FilterOptions = S["FilterOptions"];
export type StatsOverview = S["StatsOverview"];
export type MarketStats = S["MarketStats"];
export type GeoAreas = S["GeoAreas"];
export type GeoItem = S["GeoItem"];
export type GeoLevel = S["GeoLevel"];
export type Subgraph = S["Subgraph"];
export type GraphNode = S["GraphNode"];
export type NodeLabel = S["NodeLabel"];
export type BenchmarkResponse = S["BenchmarkResponse"];
export type BenchmarkCategory = S["BenchmarkCategory"];

export type EventMeta = S["EventMeta"];
export type EventRoute = S["EventRoute"];
export type EventFilters = S["EventFilters"];
export type EventCypher = S["EventCypher"];
export type EventValuation = S["EventValuation"];
export type EventSources = S["EventSources"];
export type EventToken = S["EventToken"];
export type EventAnswer = S["EventAnswer"];
export type EventUsage = S["EventUsage"];
export type EventError = S["EventError"];
export type EventDone = S["EventDone"];
export type EventCompareDone = S["EventCompareDone"];

/** Sự kiện SSE §4 dạng tagged union (tên sự kiện → payload). */
export type StreamEvent =
  | { event: "meta"; data: EventMeta }
  | { event: "route"; data: EventRoute }
  | { event: "filters"; data: EventFilters }
  | { event: "cypher"; data: EventCypher }
  | { event: "valuation"; data: EventValuation }
  | { event: "sources"; data: EventSources }
  | { event: "token"; data: EventToken }
  | { event: "answer"; data: EventAnswer }
  | { event: "usage"; data: EventUsage }
  | { event: "error"; data: EventError }
  | { event: "done"; data: EventDone }
  | { event: "compare_done"; data: EventCompareDone };

export type StreamEventName = StreamEvent["event"];

export const SYSTEM_IDS: readonly SystemId[] = ["llm_only", "basic_rag", "graph", "hybrid"];
export const PROPERTY_TYPES: readonly PropertyType[] = [
  "Nhà",
  "Đất",
  "Căn hộ chung cư",
  "Biệt thự/Nhà liền kề",
  "Shophouse",
];
export const BENCHMARK_CATEGORIES: readonly BenchmarkCategory[] = [
  "lookup",
  "constraint",
  "aggregate",
  "multihop",
  "temporal",
  "comparison",
  "project_street",
  "valuation",
  "unanswerable",
];

export const EMPTY_FILTERS: QueryFilters = {
  province: null,
  district: null,
  ward: null,
  property_type: null,
  deal_type: null,
  bedrooms: null,
  min_price_mil: null,
  max_price_mil: null,
  min_area_m2: null,
  max_area_m2: null,
  project: null,
  street: null,
  month_from: null,
  month_to: null,
};
