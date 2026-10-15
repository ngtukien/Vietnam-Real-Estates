import type { SystemId } from "@/api/schema";

export const HISTORY_KEY = "vre_history_v1";
export const HISTORY_MAX = 50;

export interface HistoryEntry {
  id: string;
  at: string;
  question: string;
  system: SystemId;
  answer: string;
}

function storage(): Storage | null {
  try {
    return localStorage;
  } catch {
    return null;
  }
}

function isEntry(value: unknown): value is HistoryEntry {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return typeof v.id === "string" && typeof v.question === "string" && typeof v.answer === "string";
}

export function loadHistory(): HistoryEntry[] {
  const raw = storage()?.getItem(HISTORY_KEY);
  if (!raw) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isEntry).slice(0, HISTORY_MAX) : [];
  } catch {
    console.warn(`[vre] ${HISTORY_KEY} hỏng, bỏ qua`);
    return [];
  }
}

/** Thêm mục mới lên đầu, giữ tối đa HISTORY_MAX mục. Trả danh sách mới. */
export function pushHistory(entries: HistoryEntry[], entry: HistoryEntry): HistoryEntry[] {
  const next = [entry, ...entries.filter((e) => e.id !== entry.id)].slice(0, HISTORY_MAX);
  storage()?.setItem(HISTORY_KEY, JSON.stringify(next));
  return next;
}

export function clearHistory(): void {
  storage()?.removeItem(HISTORY_KEY);
}
