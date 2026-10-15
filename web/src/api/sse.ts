import type { StreamEvent, StreamEventName } from "./schema";

const KNOWN_EVENTS: ReadonlySet<string> = new Set<StreamEventName>([
  "meta",
  "route",
  "filters",
  "cypher",
  "valuation",
  "sources",
  "token",
  "answer",
  "usage",
  "error",
  "done",
  "compare_done",
]);

export interface RawSseEvent {
  event: string;
  data: string;
}

/**
 * Bộ phân tích SSE tăng dần (theo đặc tả WHATWG): nhận từng mảnh văn bản bất kỳ (có thể cắt giữa dòng,
 * giữa CRLF), trả các sự kiện đã hoàn chỉnh. Dòng bắt đầu bằng `:` là comment (heartbeat `: ping`), bỏ qua.
 */
export class SseParser {
  private buffer = "";
  private eventName = "";
  private dataLines: string[] = [];
  private pendingCr = false;

  push(chunk: string): RawSseEvent[] {
    let text = chunk;
    if (this.pendingCr && text.length > 0) {
      if (text.startsWith("\n")) text = text.slice(1);
      this.pendingCr = false;
    }
    this.buffer += text;
    const out: RawSseEvent[] = [];
    for (;;) {
      const match = /\r\n|\r|\n/.exec(this.buffer);
      if (!match) break;
      const line = this.buffer.slice(0, match.index);
      this.buffer = this.buffer.slice(match.index + match[0].length);
      // CR cuối bộ đệm có thể là nửa đầu của CRLF: nhớ để bỏ LF ở mảnh sau.
      if (match[0] === "\r" && this.buffer === "") this.pendingCr = true;
      this.handleLine(line, out);
    }
    return out;
  }

  /** Kết thúc stream: sự kiện dở dang (thiếu dòng trống cuối) bị bỏ theo đặc tả. */
  end(): void {
    this.buffer = "";
    this.eventName = "";
    this.dataLines = [];
    this.pendingCr = false;
  }

  private handleLine(line: string, out: RawSseEvent[]): void {
    if (line === "") {
      if (this.dataLines.length > 0) {
        out.push({ event: this.eventName || "message", data: this.dataLines.join("\n") });
      }
      this.eventName = "";
      this.dataLines = [];
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") this.eventName = value;
    else if (field === "data") this.dataLines.push(value);
  }
}

/** Sự kiện thô → sự kiện có kiểu; tên ngoài hợp đồng hoặc JSON hỏng ⇒ ném lỗi (không nuốt im lặng). */
export function decodeEvent(raw: RawSseEvent): StreamEvent {
  if (!KNOWN_EVENTS.has(raw.event)) throw new Error(`Sự kiện SSE không có trong hợp đồng: ${raw.event}`);
  let data: unknown;
  try {
    data = JSON.parse(raw.data);
  } catch {
    throw new Error(`Payload SSE không phải JSON (sự kiện ${raw.event})`);
  }
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new Error(`Payload SSE không phải object (sự kiện ${raw.event})`);
  }
  return { event: raw.event, data } as StreamEvent;
}

/** Đọc ReadableStream byte (thân fetch) thành chuỗi sự kiện có kiểu. */
export async function* readSse(body: ReadableStream<Uint8Array>): AsyncGenerator<StreamEvent> {
  const parser = new SseParser();
  const decoder = new TextDecoder("utf-8");
  const reader = body.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const raw of parser.push(decoder.decode(value, { stream: true }))) yield decodeEvent(raw);
    }
    for (const raw of parser.push(decoder.decode())) yield decodeEvent(raw);
    parser.end();
  } finally {
    reader.releaseLock();
  }
}
