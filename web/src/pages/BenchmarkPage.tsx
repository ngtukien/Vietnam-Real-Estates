import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, errorMessage } from "@/api/client";
import { useBenchmark } from "@/api/queries";
import { BENCHMARK_CATEGORIES, SYSTEM_IDS, type SystemId } from "@/api/schema";
import { Alert, Card, CardContent, CardHeader, CardTitle, DataTable, Spinner } from "@/components/ui/primitives";
import { parseBenchmark, type Score } from "@/features/benchmark/parse";
import { formatDecimal, formatInt, formatMs, formatPercent, formatUsd } from "@/lib/format";
import { CATEGORY_LABEL, SYSTEM_LABEL } from "@/lib/labels";

const COLORS: Record<SystemId, string> = {
  llm_only: "var(--chart-4)",
  basic_rag: "var(--chart-2)",
  graph: "var(--chart-3)",
  hybrid: "var(--chart-1)",
};
const AXIS = { fontSize: 12, fill: "var(--muted-foreground)" };

function scoreText(s: Score | undefined): string {
  if (!s) return "—";
  const interval = s.ci95 ? ` [${formatPercent(s.ci95[0], 0)}–${formatPercent(s.ci95[1], 0)}]` : "";
  return `${formatPercent(s.accuracy, 0)}${interval}`;
}

export default function BenchmarkPage() {
  const { data: raw, isLoading, error } = useBenchmark();
  const bench = useMemo(() => (raw ? parseBenchmark(raw) : null), [raw]);
  const systems = useMemo<SystemId[]>(() => (bench?.systems.length ? bench.systems : [...SYSTEM_IDS]), [bench]);

  const chartData = useMemo(() => {
    if (!bench) return [];
    return BENCHMARK_CATEGORIES.filter((c) => bench.byCategory[c]).map((c) => {
      const row: Record<string, string | number | [number, number]> = { category: CATEGORY_LABEL[c] };
      for (const s of systems) {
        const sc = bench.byCategory[c]?.[s];
        if (!sc) continue;
        row[s] = sc.accuracy;
        if (sc.ci95) row[`${s}_err`] = [sc.accuracy - sc.ci95[0], sc.ci95[1] - sc.accuracy];
      }
      return row;
    });
  }, [bench, systems]);

  if (isLoading) return <Spinner label="Đang tải kết quả benchmark" />;
  if (error) {
    const notRun = error instanceof ApiError && error.code === "not_found";
    return (
      <div className="grid gap-3">
        <h1 className="text-lg font-semibold">Benchmark</h1>
        <Alert tone={notRun ? "info" : "error"}>{notRun ? "Chưa có kết quả eval (results/eval_summary.json)." : errorMessage(error)}</Alert>
      </div>
    );
  }
  if (!bench) return null;

  return (
    <div className="grid gap-4">
      <div>
        <h1 className="text-lg font-semibold">Benchmark {bench.nQuestions ?? ""} câu · 9 nhóm × {systems.length} hệ</h1>
        <p className="text-xs text-muted-foreground">
          Generator {bench.llm.generator || "—"} · judge {bench.llm.judge || "—"} · sinh lúc {bench.generatedAt ?? "—"}
          {bench.routerAccuracy !== null ? ` · router đúng ${formatPercent(bench.routerAccuracy)}` : ""}. Khoảng tin cậy 95%
          bằng bootstrap theo câu hỏi.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Độ chính xác theo nhóm câu hỏi (thanh lỗi = CI 95%)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96" aria-hidden data-testid="benchmark-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart accessibilityLayer={false} data={chartData} margin={{ left: 0, right: 8 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="category" tick={AXIS} interval={0} angle={-20} textAnchor="end" height={60} />
                <YAxis tick={AXIS} domain={[0, 1]} tickFormatter={(v: number) => formatPercent(v, 0)} />
                <Tooltip formatter={(v) => (typeof v === "number" ? formatPercent(v) : String(v))} />
                <Legend />
                {systems.map((s) => (
                  <Bar key={s} dataKey={s} name={SYSTEM_LABEL[s]} fill={COLORS[s]}>
                    <ErrorBar dataKey={`${s}_err`} width={3} strokeWidth={1} stroke="var(--foreground)" />
                  </Bar>
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <DataTable
            caption="Độ chính xác theo nhóm câu hỏi và hệ thống, kèm CI 95%"
            columns={["Nhóm", ...systems.map((s) => SYSTEM_LABEL[s])]}
            rows={[
              ...BENCHMARK_CATEGORIES.filter((c) => bench.byCategory[c]).map((c) => [
                `${CATEGORY_LABEL[c]}${bench.byCategory[c]?.[systems[0] as SystemId]?.n ? ` (n=${bench.byCategory[c]?.[systems[0] as SystemId]?.n})` : ""}`,
                ...systems.map((s) => scoreText(bench.byCategory[c]?.[s])),
              ]),
              ["Tổng", ...systems.map((s) => scoreText(bench.overall[s]))],
            ]}
            className="mt-3"
          />
        </CardContent>
      </Card>

      {bench.pairwise.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>So sánh cặp (bootstrap ghép cặp)</CardTitle>
          </CardHeader>
          <CardContent>
            <DataTable
              caption="So sánh cặp"
              columns={["A − B", "Phạm vi", "Chỉ số", "Chênh lệch", "CI 95%", "p-value"]}
              rows={bench.pairwise.map((p) => [
                `${SYSTEM_LABEL[p.a as SystemId] ?? p.a} − ${SYSTEM_LABEL[p.b as SystemId] ?? p.b}`,
                CATEGORY_LABEL[p.scope as keyof typeof CATEGORY_LABEL] ?? p.scope,
                p.metric,
                formatPercent(p.diff),
                p.ci95 ? `[${formatPercent(p.ci95[0])}, ${formatPercent(p.ci95[1])}]` : "—",
                formatDecimal(p.p_value, 4),
              ])}
            />
          </CardContent>
        </Card>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-2">
        {Object.keys(bench.retrieval).length > 0 ? (
          <Card>
            <CardHeader>
              <CardTitle>Truy hồi</CardTitle>
            </CardHeader>
            <CardContent>
              <DataTable
                caption="Chỉ số truy hồi"
                columns={["Hệ", "Recall@5", "Recall@10", "MRR", "nDCG@10"]}
                rows={(Object.entries(bench.retrieval) as [SystemId, Record<string, number>][]).map(([s, r]) => [
                  SYSTEM_LABEL[s] ?? s,
                  formatDecimal(r.recall_at_5),
                  formatDecimal(r.recall_at_10),
                  formatDecimal(r.mrr),
                  formatDecimal(r.ndcg_at_10),
                ])}
              />
            </CardContent>
          </Card>
        ) : null}
        {Object.keys(bench.cost).length > 0 ? (
          <Card>
            <CardHeader>
              <CardTitle>Chi phí và độ trễ</CardTitle>
            </CardHeader>
            <CardContent>
              <DataTable
                caption="Chi phí và độ trễ"
                columns={["Hệ", "p50", "p95", "USD/câu", "Trích dẫn sai"]}
                rows={(Object.entries(bench.cost) as [SystemId, Record<string, number>][]).map(([s, c]) => [
                  SYSTEM_LABEL[s] ?? s,
                  formatMs(c.latency_ms_p50),
                  formatMs(c.latency_ms_p95),
                  formatUsd(c.usd_per_question),
                  formatPercent(c.bad_citation_rate),
                ])}
              />
            </CardContent>
          </Card>
        ) : null}
      </div>

      {bench.ablation.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>Ablation truy hồi (embedding e5-large)</CardTitle>
          </CardHeader>
          <CardContent>
            <DataTable
              caption="Ablation"
              columns={["Cấu hình", "Chunk", "Truy hồi", "Lọc trước", "Rerank", "Recall@5", "Recall@10", "MRR", "nDCG@10", "p50"]}
              rows={bench.ablation.map((r) => [
                r.name,
                formatInt(r.chunk_size),
                r.retrieval,
                r.prefilter ? "có" : "không",
                r.rerank ? "có" : "không",
                formatDecimal(r.recall_at_5),
                formatDecimal(r.recall_at_10),
                formatDecimal(r.mrr),
                formatDecimal(r.ndcg_at_10),
                formatMs(r.latency_ms_p50),
              ])}
            />
          </CardContent>
        </Card>
      ) : null}

      {bench.embedding.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>So sánh model embedding (đo 2026-10-15)</CardTitle>
          </CardHeader>
          <CardContent>
            <DataTable
              caption="So sánh model embedding"
              columns={["Model", "Tham số (M)", "Chiều", "Docs/s", "Recall@1", "Recall@10", "MRR@10"]}
              rows={bench.embedding.map((e) => [
                e.model,
                formatInt(e.params_m),
                formatInt(e.dim),
                formatDecimal(e.docs_per_s, 1),
                formatDecimal(e.recall_at_1),
                formatDecimal(e.recall_at_10),
                formatDecimal(e.mrr_at_10),
              ])}
            />
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
