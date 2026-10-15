import type { ErrorCode, ErrorEnvelope, StreamEvent } from "./schema";
import { readSse } from "./sse";

/** Khoá sessionStorage cho API key nhập tay (api-contract §0). KHÔNG bao giờ đọc key từ URL. */
export const API_KEY_STORAGE = "vre_api_key";

export function getApiKey(): string | null {
  try {
    return sessionStorage.getItem(API_KEY_STORAGE);
  } catch {
    return null;
  }
}

export function setApiKey(key: string | null): void {
  if (key) sessionStorage.setItem(API_KEY_STORAGE, key);
  else sessionStorage.removeItem(API_KEY_STORAGE);
}

/** Lỗi HTTP mang envelope §5 (code + request_id) để UI hiện đúng thông điệp. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: ErrorCode | "network_error";
  readonly requestId: string | null;
  readonly retryAfter: number | null;

  constructor(
    status: number,
    code: ErrorCode | "network_error",
    message: string,
    requestId: string | null,
    retryAfter: number | null,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
    this.retryAfter = retryAfter;
  }
}

function buildHeaders(accept: string, json: boolean): Headers {
  const h = new Headers({ Accept: accept });
  if (json) h.set("Content-Type", "application/json");
  const key = getApiKey();
  if (key) h.set("X-API-Key", key);
  return h;
}

async function toApiError(res: Response): Promise<ApiError> {
  const retry = res.headers.get("Retry-After");
  const retryAfter = retry !== null && retry !== "" ? Number(retry) : null;
  try {
    const body = (await res.json()) as Partial<ErrorEnvelope>;
    if (body?.error) {
      const e = body.error;
      return new ApiError(res.status, e.code, e.message, e.request_id, retryAfter);
    }
  } catch {
    // thân không phải JSON (vd lỗi proxy) — dùng thông điệp chung bên dưới
  }
  return new ApiError(
    res.status,
    "internal_error",
    `Lỗi máy chủ (HTTP ${res.status})`,
    res.headers.get("X-Request-Id"),
    retryAfter,
  );
}

export type QueryParams = Record<string, string | number | null | undefined>;

export function buildUrl(path: string, params?: QueryParams): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v !== null && v !== undefined && v !== "") qs.set(k, String(v));
  }
  const s = qs.toString();
  return s ? `${path}?${s}` : path;
}

async function send(url: string, init: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(0, "network_error", "Không kết nối được tới API", null, null);
  }
  if (!res.ok) throw await toApiError(res);
  return res;
}

export async function apiGet<T>(path: string, params?: QueryParams, signal?: AbortSignal): Promise<T> {
  const res = await send(buildUrl(path, params), { headers: buildHeaders("application/json", false), signal });
  return (await res.json()) as T;
}

export async function apiPost<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await send(path, {
    method: "POST",
    headers: buildHeaders("application/json", true),
    body: JSON.stringify(body),
    signal,
  });
  return (await res.json()) as T;
}

/** POST trả SSE (§4). Lỗi trước khi mở stream ⇒ ApiError; sau đó mọi lỗi đến dưới dạng sự kiện `error`. */
export async function* apiStream(path: string, body: unknown, signal?: AbortSignal): AsyncGenerator<StreamEvent> {
  const res = await send(path, {
    method: "POST",
    headers: buildHeaders("text/event-stream", true),
    body: JSON.stringify(body),
    signal,
  });
  if (!res.body) throw new ApiError(res.status, "internal_error", "Phản hồi stream rỗng", null, null);
  yield* readSse(res.body);
}

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    const suffix = err.requestId ? ` (mã ${err.requestId})` : "";
    if (err.code === "unauthorized") return `Cần API key hợp lệ — mở Cài đặt để nhập.${suffix}`;
    if (err.code === "rate_limited") return `Gửi quá nhanh, thử lại sau ${err.retryAfter ?? "vài"} giây.${suffix}`;
    return `${err.message}${suffix}`;
  }
  if (err instanceof Error) return err.message;
  return "Lỗi không xác định";
}
