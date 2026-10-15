import type {
  CypherResult,
  ErrorCode,
  EventAnswer,
  EventCompareDone,
  EventFilters,
  EventMeta,
  EventRoute,
  EventUsage,
  ListingCard,
  QueryFilters,
  StreamEvent,
  SystemId,
  ValuationResult,
} from "@/api/schema";

export type RunStatus = "idle" | "streaming" | "done" | "error";

/** Trạng thái hiển thị của MỘT hệ thống trong một lượt hỏi (chat: 1 hệ; so sánh: tối đa 4). */
export interface SystemRun {
  system: SystemId;
  status: RunStatus;
  meta: EventMeta | null;
  route: EventRoute | null;
  filters: EventFilters | null;
  cypher: CypherResult | null;
  valuation: ValuationResult | null;
  sources: ListingCard[];
  /** Văn bản đang stream (ghép các `token`); bị thay bằng `answer.answer` khi có sự kiện answer. */
  text: string;
  answer: EventAnswer | null;
  usage: EventUsage | null;
  error: { code: ErrorCode | "network_error" | "protocol_error"; message: string } | null;
}

export interface ChatState {
  requestId: string | null;
  question: string;
  runs: Partial<Record<SystemId, SystemRun>>;
  compareDone: EventCompareDone | null;
  /** Lỗi trước khi stream mở (HTTP) hoặc lỗi mạng — không gắn với hệ cụ thể. */
  requestError: string | null;
  active: boolean;
}

export type ChatAction =
  | { type: "start"; question: string; systems: SystemId[] }
  | { type: "event"; event: StreamEvent }
  | { type: "request_error"; message: string }
  | { type: "stream_closed" }
  | { type: "reset" };

export const initialChatState: ChatState = {
  requestId: null,
  question: "",
  runs: {},
  compareDone: null,
  requestError: null,
  active: false,
};

export function emptyRun(system: SystemId): SystemRun {
  return {
    system,
    status: "streaming",
    meta: null,
    route: null,
    filters: null,
    cypher: null,
    valuation: null,
    sources: [],
    text: "",
    answer: null,
    usage: null,
    error: null,
  };
}

function stripEnvelope<T extends { request_id: string; system: SystemId }>(
  payload: T,
): Omit<T, "request_id" | "system"> {
  const { request_id: _r, system: _s, ...rest } = payload;
  return rest;
}

function updateRun(state: ChatState, system: SystemId, patch: (run: SystemRun) => SystemRun): ChatState {
  const current = state.runs[system] ?? emptyRun(system);
  return { ...state, runs: { ...state.runs, [system]: patch(current) } };
}

function applyEvent(state: ChatState, event: StreamEvent): ChatState {
  if (event.event === "compare_done") {
    return { ...state, compareDone: event.data, requestId: state.requestId ?? event.data.request_id };
  }
  const system = event.data.system;
  const withId = state.requestId ? state : { ...state, requestId: event.data.request_id };
  switch (event.event) {
    case "meta":
      return updateRun(withId, system, (r) => ({ ...r, meta: event.data, status: "streaming" }));
    case "route":
      return updateRun(withId, system, (r) => ({ ...r, route: event.data }));
    case "filters":
      return updateRun(withId, system, (r) => ({ ...r, filters: event.data }));
    case "cypher":
      return updateRun(withId, system, (r) => ({ ...r, cypher: stripEnvelope(event.data) }));
    case "valuation":
      return updateRun(withId, system, (r) => ({ ...r, valuation: stripEnvelope(event.data) }));
    case "sources":
      return updateRun(withId, system, (r) => ({ ...r, sources: event.data.sources }));
    case "token":
      return updateRun(withId, system, (r) => ({ ...r, text: r.text + event.data.text }));
    case "answer": {
      const cited = new Set(event.data.cited);
      return updateRun(withId, system, (r) => ({
        ...r,
        answer: event.data,
        text: event.data.answer,
        // §4: `cited` của thẻ nguồn được cập nhật lại ở `answer`.
        sources: r.sources.map((s) => ({ ...s, cited: cited.has(s.listing_id) })),
      }));
    }
    case "usage":
      return updateRun(withId, system, (r) => ({ ...r, usage: event.data }));
    case "error":
      return updateRun(withId, system, (r) => ({
        ...r,
        error: { code: event.data.code, message: event.data.message },
        status: "error",
      }));
    case "done":
      return updateRun(withId, system, (r) => ({
        ...r,
        status: event.data.status === "error" || r.error ? "error" : "done",
      }));
  }
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "start":
      return {
        ...initialChatState,
        question: action.question,
        active: true,
        runs: Object.fromEntries(action.systems.map((s) => [s, emptyRun(s)])),
      };
    case "event":
      return applyEvent(state, action.event);
    case "request_error":
      return {
        ...state,
        active: false,
        requestError: action.message,
        runs: Object.fromEntries(
          Object.entries(state.runs).map(([k, r]) => [k, { ...r, status: r.status === "streaming" ? "error" : r.status }]),
        ),
      };
    case "stream_closed": {
      // Stream đóng mà hệ chưa nhận `done` ⇒ lỗi giao thức, hiển thị rõ thay vì treo "đang trả lời".
      const runs = Object.fromEntries(
        Object.entries(state.runs).map(([k, r]) => [
          k,
          r.status === "streaming"
            ? { ...r, status: "error" as const, error: r.error ?? { code: "protocol_error" as const, message: "Stream kết thúc trước sự kiện done" } }
            : r,
        ]),
      );
      return { ...state, runs, active: false };
    }
    case "reset":
      return initialChatState;
  }
}

/** Ví dụ: câu trả lời → các id [Tin#id] xuất hiện (đã chuẩn hoá ở server; dùng cho liên kết cuộn tới thẻ). */
export const CITATION_RE = /\[Tin#(\d{1,16})\]/g;

/** Thay `[Tin#id]` bằng liên kết markdown tới thẻ nguồn `#src-<id>` (chỉ khi id có trong nguồn). */
export function linkCitations(answer: string, sourceIds: ReadonlySet<number>): string {
  return answer.replace(CITATION_RE, (whole, id: string) =>
    sourceIds.has(Number(id)) ? `[Tin#${id}](#src-${id})` : whole,
  );
}

/** Bộ lọc gửi lên server: null nếu không có trường nào được điền. */
export function filtersOrNull(f: QueryFilters): QueryFilters | null {
  return Object.values(f).some((v) => v !== null) ? f : null;
}
