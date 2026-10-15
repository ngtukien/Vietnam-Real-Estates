import { describe, expect, it } from "vitest";
import type { ListingCard, StreamEvent, SystemId } from "@/api/schema";
import { EMPTY_FILTERS } from "@/api/schema";
import { chatReducer, initialChatState, linkCitations, type ChatState } from "./reducer";

const RID = "a1b2c3d4";

function card(id: number): ListingCard {
  return {
    listing_id: id,
    dup_cluster_id: id,
    title: `Tin ${id}`,
    property_type: "Căn hộ chung cư",
    deal_type: "sale",
    price_vnd: 3_900_000_000,
    price_m2_mil: 55.7,
    area_m2: 70,
    bedrooms: 2,
    bathrooms: 2,
    address: { street: null, ward: null, district: "Quận 7", province: "Hồ Chí Minh" },
    project: null,
    published_month: "2026-03",
    description_excerpt: "",
    amenities: [],
    score: 0.5,
    cited: false,
  };
}

function run(events: StreamEvent[], systems: SystemId[] = ["hybrid"]): ChatState {
  let s = chatReducer(initialChatState, { type: "start", question: "q", systems });
  for (const e of events) s = chatReducer(s, { type: "event", event: e });
  return s;
}

const env = (system: SystemId) => ({ request_id: RID, system });

describe("chatReducer", () => {
  it("ghép token rồi thay bằng answer, cập nhật cited cho thẻ nguồn", () => {
    const s = run([
      { event: "meta", data: { ...env("hybrid"), question: "q", started_at: "2026-10-15T07:00:00Z" } },
      { event: "route", data: { ...env("hybrid"), route: "hybrid", reason: "cần cả hai" } },
      { event: "filters", data: { ...env("hybrid"), filters: EMPTY_FILTERS, source: "llm", note: null } },
      { event: "sources", data: { ...env("hybrid"), sources: [card(11), card(22)] } },
      { event: "token", data: { ...env("hybrid"), text: "Có " } },
      { event: "token", data: { ...env("hybrid"), text: "2 tin" } },
    ]);
    expect(s.requestId).toBe(RID);
    expect(s.runs.hybrid?.text).toBe("Có 2 tin");
    expect(s.runs.hybrid?.route?.route).toBe("hybrid");

    const s2 = chatReducer(s, {
      type: "event",
      event: {
        event: "answer",
        data: { ...env("hybrid"), answer: "Có 2 tin [Tin#22]", cited: [22], unsupported_citations: [], fallback: false, route: "hybrid" },
      },
    });
    expect(s2.runs.hybrid?.text).toBe("Có 2 tin [Tin#22]");
    expect(s2.runs.hybrid?.sources.map((c) => c.cited)).toEqual([false, true]);

    const s3 = chatReducer(s2, { type: "event", event: { event: "done", data: { ...env("hybrid"), status: "ok" } } });
    expect(s3.runs.hybrid?.status).toBe("done");
  });

  it("cypher bỏ request_id/system và giữ lỗi để hiển thị", () => {
    const s = run([
      {
        event: "cypher",
        data: { ...env("graph"), cypher: "MATCH (n) RETURN n LIMIT 50", columns: [], rows: [], error: "SyntaxError", attempts: 3 },
      },
    ], ["graph"]);
    expect(s.runs.graph?.cypher).toEqual({ cypher: "MATCH (n) RETURN n LIMIT 50", columns: [], rows: [], error: "SyntaxError", attempts: 3 });
  });

  it("error → done giữ trạng thái lỗi", () => {
    const s = run([
      { event: "meta", data: { ...env("graph"), question: "q", started_at: "t" } },
      { event: "error", data: { ...env("graph"), code: "system_unavailable", message: "Neo4j chưa sẵn sàng" } },
      { event: "done", data: { ...env("graph"), status: "error" } },
    ], ["graph"]);
    expect(s.runs.graph?.status).toBe("error");
    expect(s.runs.graph?.error?.message).toBe("Neo4j chưa sẵn sàng");
  });

  it("so sánh: sự kiện xen kẽ, cột lỗi không ảnh hưởng cột khác", () => {
    const s = run(
      [
        { event: "meta", data: { ...env("graph"), question: "q", started_at: "t" } },
        { event: "meta", data: { ...env("llm_only"), question: "q", started_at: "t" } },
        { event: "token", data: { ...env("llm_only"), text: "A" } },
        { event: "error", data: { ...env("graph"), code: "timeout", message: "quá hạn" } },
        { event: "done", data: { ...env("graph"), status: "error" } },
        { event: "token", data: { ...env("llm_only"), text: "B" } },
        { event: "done", data: { ...env("llm_only"), status: "ok" } },
        { event: "compare_done", data: { request_id: RID, systems_ok: ["llm_only"], systems_failed: ["graph"] } },
      ],
      ["llm_only", "graph"],
    );
    expect(s.runs.llm_only?.text).toBe("AB");
    expect(s.runs.llm_only?.status).toBe("done");
    expect(s.runs.graph?.status).toBe("error");
    expect(s.compareDone?.systems_failed).toEqual(["graph"]);
  });

  it("stream đóng sớm ⇒ hệ đang chạy chuyển lỗi giao thức", () => {
    const s = run([{ event: "meta", data: { ...env("hybrid"), question: "q", started_at: "t" } }]);
    const closed = chatReducer(s, { type: "stream_closed" });
    expect(closed.active).toBe(false);
    expect(closed.runs.hybrid?.status).toBe("error");
    expect(closed.runs.hybrid?.error?.code).toBe("protocol_error");
  });

  it("lỗi HTTP trước stream được lưu ở requestError", () => {
    const s = chatReducer(run([]), { type: "request_error", message: "Gửi quá nhanh" });
    expect(s.requestError).toBe("Gửi quá nhanh");
    expect(s.runs.hybrid?.status).toBe("error");
  });
});

describe("linkCitations", () => {
  it("chỉ biến [Tin#id] có trong nguồn thành liên kết", () => {
    const out = linkCitations("A [Tin#1234567890123456] B [Tin#9]", new Set([1234567890123456]));
    expect(out).toBe("A [Tin#1234567890123456](#src-1234567890123456) B [Tin#9]");
  });
});
