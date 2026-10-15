import type { BenchmarkResponse, SystemId } from "@/api/schema";

/** Khung `results/eval_summary.json` (data-contract §7.7) — `/api/benchmark` trả object tự do nên đọc phòng thủ. */
export interface Score {
  n?: number;
  accuracy: number;
  mean_points?: number;
  ci95: [number, number] | null;
}

export interface Pairwise {
  a: string;
  b: string;
  scope: string;
  metric: string;
  diff: number;
  ci95: [number, number] | null;
  p_value: number;
  n_boot?: number;
}

export interface AblationRun {
  name: string;
  embedding: string;
  chunk_size: number;
  retrieval: string;
  prefilter: boolean;
  rerank: boolean;
  recall_at_5: number;
  recall_at_10: number;
  mrr: number;
  ndcg_at_10: number;
  n_questions: number;
  latency_ms_p50: number;
}

export interface EmbeddingRow {
  model: string;
  params_m: number;
  dim: number;
  docs_per_s: number;
  recall_at_1: number;
  recall_at_10: number;
  mrr_at_10: number;
}

export interface Benchmark {
  generatedAt: string | null;
  nQuestions: number | null;
  systems: SystemId[];
  llm: { generator?: string; judge?: string };
  overall: Partial<Record<SystemId, Score>>;
  byCategory: Record<string, Partial<Record<SystemId, Score>>>;
  retrieval: Partial<Record<SystemId, Record<string, number>>>;
  cost: Partial<Record<SystemId, Record<string, number>>>;
  pairwise: Pairwise[];
  routerAccuracy: number | null;
  ablation: AblationRun[];
  embedding: EmbeddingRow[];
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

function ci(v: unknown): [number, number] | null {
  return Array.isArray(v) && v.length === 2 && num(v[0]) !== null && num(v[1]) !== null ? [v[0] as number, v[1] as number] : null;
}

function score(v: unknown): Score | null {
  if (!isObj(v) || num(v.accuracy) === null) return null;
  return {
    n: num(v.n) ?? undefined,
    accuracy: v.accuracy as number,
    mean_points: num(v.mean_points) ?? undefined,
    ci95: ci(v.ci95),
  };
}

function scores(v: unknown): Partial<Record<SystemId, Score>> {
  const out: Partial<Record<SystemId, Score>> = {};
  if (!isObj(v)) return out;
  for (const [k, s] of Object.entries(v)) {
    const parsed = score(s);
    if (parsed) out[k as SystemId] = parsed;
  }
  return out;
}

function numbers(v: unknown): Partial<Record<SystemId, Record<string, number>>> {
  const out: Partial<Record<SystemId, Record<string, number>>> = {};
  if (!isObj(v)) return out;
  for (const [k, row] of Object.entries(v)) {
    if (!isObj(row)) continue;
    out[k as SystemId] = Object.fromEntries(Object.entries(row).filter(([, x]) => num(x) !== null)) as Record<string, number>;
  }
  return out;
}

export function parseBenchmark(raw: BenchmarkResponse): Benchmark {
  const byCategory: Benchmark["byCategory"] = {};
  if (isObj(raw.by_category)) for (const [cat, v] of Object.entries(raw.by_category)) byCategory[cat] = scores(v);
  const pairwise = Array.isArray(raw.pairwise)
    ? raw.pairwise.filter(isObj).map((p) => ({
        a: String(p.a),
        b: String(p.b),
        scope: String(p.scope),
        metric: String(p.metric),
        diff: num(p.diff) ?? Number.NaN,
        ci95: ci(p.ci95),
        p_value: num(p.p_value) ?? Number.NaN,
        n_boot: num(p.n_boot) ?? undefined,
      }))
    : [];
  const ablationRuns = isObj(raw.ablation) && Array.isArray(raw.ablation.runs) ? raw.ablation.runs.filter(isObj) : [];
  const embedding = Array.isArray(raw.embedding_benchmark) ? raw.embedding_benchmark.filter(isObj) : [];
  const router = isObj(raw.router) ? num(raw.router.accuracy) : null;
  return {
    generatedAt: typeof raw.generated_at === "string" ? raw.generated_at : null,
    nQuestions: num(raw.n_questions),
    systems: Array.isArray(raw.systems) ? (raw.systems.filter((s) => typeof s === "string") as SystemId[]) : [],
    llm: isObj(raw.llm) ? { generator: String(raw.llm.generator ?? ""), judge: String(raw.llm.judge ?? "") } : {},
    overall: scores(raw.overall),
    byCategory,
    retrieval: numbers(raw.retrieval),
    cost: numbers(raw.cost),
    pairwise,
    routerAccuracy: router,
    ablation: ablationRuns as unknown as AblationRun[],
    embedding: embedding as unknown as EmbeddingRow[],
  };
}
