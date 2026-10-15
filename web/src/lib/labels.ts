import type { BenchmarkCategory, ErrorCode, Route, SystemId } from "@/api/schema";

export const SYSTEM_LABEL: Record<SystemId, string> = {
  llm_only: "Chỉ LLM",
  basic_rag: "RAG cơ bản",
  graph: "GraphRAG (Cypher)",
  hybrid: "Hybrid",
};

export const ROUTE_LABEL: Record<Route, string> = {
  basic_rag: "RAG",
  graph: "Đồ thị",
  hybrid: "RAG + đồ thị",
  valuation: "Định giá",
};

export const CATEGORY_LABEL: Record<BenchmarkCategory, string> = {
  lookup: "Tra cứu",
  constraint: "Ràng buộc",
  aggregate: "Tổng hợp",
  multihop: "Nhiều chặng",
  temporal: "Thời gian",
  comparison: "So sánh",
  project_street: "Dự án/đường",
  valuation: "Định giá",
  unanswerable: "Không trả lời được",
};

export const ERROR_LABEL: Partial<Record<ErrorCode, string>> = {
  busy: "Máy chủ đang bận",
  timeout: "Quá thời gian",
  system_unavailable: "Hệ thống chưa sẵn sàng",
  upstream_llm_error: "Lỗi LLM",
  rate_limited: "Quá giới hạn tần suất",
};

export const VERDICT_LABEL = {
  below_market: "Thấp hơn thị trường",
  fair: "Hợp lý (trong khoảng dự báo)",
  above_market: "Cao hơn thị trường",
} as const;

export const FILTER_SOURCE_LABEL = {
  llm: "LLM trích",
  manual: "chỉnh tay",
  merged: "LLM + chỉnh tay",
} as const;
