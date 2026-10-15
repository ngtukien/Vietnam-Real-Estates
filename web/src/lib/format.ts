const DASH = "—";

const nf = (digits: number) =>
  new Intl.NumberFormat("vi-VN", { maximumFractionDigits: digits, minimumFractionDigits: 0 });

/** Số nguyên kiểu Việt Nam: 1.234.567 */
export function formatInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return nf(0).format(Math.round(value));
}

/** Tiền VND gọn: 7,45 tỷ · 850 triệu · 12.000 đ. Giữ dấu âm. */
export function formatVnd(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1e9) return `${sign}${nf(2).format(abs / 1e9)} tỷ`;
  if (abs >= 1e6) return `${sign}${nf(abs >= 1e8 ? 0 : 1).format(abs / 1e6)} triệu`;
  return `${sign}${nf(0).format(abs)} đ`;
}

/** Triệu đồng (giá tổng nhập ở bộ lọc) → chuỗi tiền. */
export function formatMillion(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return formatVnd(value * 1e6);
}

/** Giá/m² (triệu đồng/m²): 85,3 tr/m² */
export function formatPriceM2(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return `${nf(1).format(value)} tr/m²`;
}

export function formatArea(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return `${nf(1).format(value)} m²`;
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return `${nf(digits).format(value * 100)}%`;
}

export function formatUsd(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return `$${value < 0.01 ? value.toFixed(5) : value.toFixed(4)}`;
}

export function formatMs(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return value >= 1000 ? `${nf(2).format(value / 1000)} s` : `${formatInt(value)} ms`;
}

export function formatDecimal(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return nf(digits).format(value);
}

/** "2026-03" → "03/2026" */
export function formatMonth(month: string | null | undefined): string {
  if (!month) return DASH;
  const m = /^(\d{4})-(\d{2})$/.exec(month);
  return m ? `${m[2]}/${m[1]}` : month;
}

/** Ô nhập số tuỳ chọn: "" → null; giá trị không hợp lệ → null. */
export function parseOptionalNumber(text: string): number | null {
  const trimmed = text.trim().replace(",", ".");
  if (trimmed === "") return null;
  const n = Number(trimmed);
  return Number.isFinite(n) ? n : null;
}
