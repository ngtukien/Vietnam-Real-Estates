import { describe, expect, it } from "vitest";
import { SseParser, decodeEvent, readSse } from "./sse";

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
}

describe("SseParser", () => {
  it("phân tích sự kiện đầy đủ và bỏ heartbeat", () => {
    const p = new SseParser();
    const out = p.push(': ping\n\nevent: meta\ndata: {"a":1}\n\n');
    expect(out).toEqual([{ event: "meta", data: '{"a":1}' }]);
  });

  it("ghép sự kiện bị cắt ở giữa dòng và giữa các mảnh", () => {
    const p = new SseParser();
    expect(p.push("event: tok")).toEqual([]);
    expect(p.push('en\ndata: {"text":"Xin ')).toEqual([]);
    expect(p.push('chào"}\n')).toEqual([]);
    expect(p.push("\n")).toEqual([{ event: "token", data: '{"text":"Xin chào"}' }]);
  });

  it("xử lý CRLF kể cả khi CR và LF nằm ở hai mảnh khác nhau", () => {
    const p = new SseParser();
    expect(p.push("event: done\r")).toEqual([]);
    expect(p.push('\ndata: {"status":"ok"}\r\n\r')).toEqual([{ event: "done", data: '{"status":"ok"}' }]);
    expect(p.push("\n")).toEqual([]);
  });

  it("nối nhiều dòng data bằng \\n và bỏ một dấu cách đầu", () => {
    const p = new SseParser();
    expect(p.push("data: a\ndata:b\n\n")).toEqual([{ event: "message", data: "a\nb" }]);
  });

  it("bỏ sự kiện không có data và sự kiện dở dang khi kết thúc", () => {
    const p = new SseParser();
    expect(p.push("event: meta\n\n")).toEqual([]);
    expect(p.push("event: meta\ndata: {}")).toEqual([]);
    p.end();
    expect(p.push("\n\n")).toEqual([]);
  });
});

describe("decodeEvent", () => {
  it("trả sự kiện có kiểu", () => {
    const e = decodeEvent({ event: "done", data: '{"request_id":"abcd1234","system":"hybrid","status":"ok"}' });
    expect(e.event).toBe("done");
    if (e.event === "done") expect(e.data.status).toBe("ok");
  });

  it("ném lỗi với tên sự kiện ngoài hợp đồng hoặc JSON hỏng", () => {
    expect(() => decodeEvent({ event: "weird", data: "{}" })).toThrow(/không có trong hợp đồng/);
    expect(() => decodeEvent({ event: "token", data: "{oops" })).toThrow(/không phải JSON/);
    expect(() => decodeEvent({ event: "token", data: "[1]" })).toThrow(/không phải object/);
  });
});

describe("readSse", () => {
  it("đọc stream byte với ký tự UTF-8 nhiều byte bị cắt đôi", async () => {
    const bytes = new TextEncoder().encode('event: token\ndata: {"request_id":"r","system":"hybrid","text":"Đà Nẵng"}\n\n');
    const cut = bytes.indexOf(0xc4) + 1; // cắt giữa ký tự "Đ" (2 byte)
    const body = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(bytes.slice(0, cut));
        c.enqueue(bytes.slice(cut));
        c.close();
      },
    });
    const events = [];
    for await (const e of readSse(body)) events.push(e);
    expect(events).toHaveLength(1);
    const first = events[0];
    expect(first?.event).toBe("token");
    if (first?.event === "token") expect(first.data.text).toBe("Đà Nẵng");
  });

  it("giữ đúng thứ tự nhiều sự kiện", async () => {
    const names: string[] = [];
    const body = streamOf([
      'event: meta\ndata: {"request_id":"r","system":"graph","question":"q","started_at":"t"}\n\n',
      ': ping\n\n',
      'event: done\ndata: {"request_id":"r","system":"graph","status":"ok"}\n\n',
    ]);
    for await (const e of readSse(body)) names.push(e.event);
    expect(names).toEqual(["meta", "done"]);
  });
});
