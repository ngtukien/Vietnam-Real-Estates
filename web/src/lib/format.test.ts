import { describe, expect, it } from "vitest";
import {
  formatArea,
  formatInt,
  formatMillion,
  formatMonth,
  formatPercent,
  formatPriceM2,
  formatVnd,
  parseOptionalNumber,
} from "./format";

// Intl vi-VN dùng dấu chấm phân nghìn và dấu phẩy thập phân.
describe("formatVnd", () => {
  it("đổi sang tỷ / triệu / đồng", () => {
    expect(formatVnd(7_450_000_000)).toBe("7,45 tỷ");
    expect(formatVnd(1_000_000_000)).toBe("1 tỷ");
    expect(formatVnd(850_000_000)).toBe("850 triệu");
    expect(formatVnd(12_500_000)).toBe("12,5 triệu");
    expect(formatVnd(12_000)).toBe("12.000 đ");
  });

  it("giữ dấu âm và trả — cho null/NaN", () => {
    expect(formatVnd(-2_500_000_000)).toBe("-2,5 tỷ");
    expect(formatVnd(null)).toBe("—");
    expect(formatVnd(undefined)).toBe("—");
    expect(formatVnd(Number.NaN)).toBe("—");
  });
});

describe("các định dạng khác", () => {
  it("giá/m², diện tích, số nguyên, phần trăm", () => {
    expect(formatPriceM2(85.34)).toBe("85,3 tr/m²");
    expect(formatPriceM2(null)).toBe("—");
    expect(formatArea(72)).toBe("72 m²");
    expect(formatInt(2_712_345)).toBe("2.712.345");
    expect(formatPercent(0.1234)).toBe("12,3%");
  });

  it("triệu đồng của bộ lọc → tiền", () => {
    expect(formatMillion(4000)).toBe("4 tỷ");
    expect(formatMillion(null)).toBe("—");
  });

  it("tháng YYYY-MM → MM/YYYY", () => {
    expect(formatMonth("2026-03")).toBe("03/2026");
    expect(formatMonth(null)).toBe("—");
  });

  it("parseOptionalNumber: rỗng/không hợp lệ → null, chấp nhận dấu phẩy", () => {
    expect(parseOptionalNumber("")).toBeNull();
    expect(parseOptionalNumber("  ")).toBeNull();
    expect(parseOptionalNumber("abc")).toBeNull();
    expect(parseOptionalNumber("72,5")).toBe(72.5);
    expect(parseOptionalNumber("4000")).toBe(4000);
  });
});
