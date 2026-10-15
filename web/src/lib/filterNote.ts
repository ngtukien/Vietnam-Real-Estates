/** Ý nghĩa của `filters.note` (api §4, `str | null`): bỏ hết lọc, hoặc nới một số trường. */
export const FILTER_DROPPED = "filter_dropped";
export const FILTER_RELAXED_PREFIX = "filter_relaxed:";

export const RELAXED_FIELD_LABEL: Record<string, string> = {
  project: "dự án",
  street: "đường",
  bedrooms: "số phòng ngủ",
  price: "giá",
  area: "diện tích",
  province: "tỉnh/thành",
  district: "quận/huyện",
  ward: "phường/xã",
};

export type FilterNoteView =
  | { kind: "dropped"; label: string }
  | { kind: "relaxed"; fields: string[]; label: string };

/** Chuyển `note` thành nhãn tiếng Việt; trường lạ giữ nguyên tên gốc. `null` khi không có gì để hiển thị. */
export function describeFilterNote(note: string | null | undefined): FilterNoteView | null {
  if (note === FILTER_DROPPED) return { kind: "dropped", label: "Đã bỏ lọc vì 0 kết quả" };
  if (!note?.startsWith(FILTER_RELAXED_PREFIX)) return null;
  const fields = note
    .slice(FILTER_RELAXED_PREFIX.length)
    .split(",")
    .map((f) => f.trim())
    .filter(Boolean);
  if (fields.length === 0) return null;
  const names = [...new Set(fields.map((f) => RELAXED_FIELD_LABEL[f] ?? f))];
  return { kind: "relaxed", fields, label: `Đã nới lọc: ${names.join(", ")}` };
}
