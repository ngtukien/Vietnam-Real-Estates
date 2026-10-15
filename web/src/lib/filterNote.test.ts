import { describe, expect, it } from "vitest";
import { describeFilterNote } from "./filterNote";

describe("describeFilterNote", () => {
  it("không hiển thị gì khi note rỗng hoặc lạ", () => {
    expect(describeFilterNote(null)).toBeNull();
    expect(describeFilterNote(undefined)).toBeNull();
    expect(describeFilterNote("something_else")).toBeNull();
    expect(describeFilterNote("filter_relaxed:")).toBeNull();
  });

  it("bỏ hết lọc", () => {
    expect(describeFilterNote("filter_dropped")).toEqual({ kind: "dropped", label: "Đã bỏ lọc vì 0 kết quả" });
  });

  it("nới một trường", () => {
    expect(describeFilterNote("filter_relaxed:project")).toEqual({
      kind: "relaxed",
      fields: ["project"],
      label: "Đã nới lọc: dự án",
    });
  });

  it("nới nhiều trường, trường lạ giữ tên gốc", () => {
    const v = describeFilterNote("filter_relaxed:project,district,foo");
    expect(v?.kind).toBe("relaxed");
    expect(v?.label).toBe("Đã nới lọc: dự án, quận/huyện, foo");
  });
});
