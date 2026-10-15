import { useMemo } from "react";
import { Alert, Badge, Spinner } from "@/components/ui/primitives";
import { ValuationResultView } from "@/features/valuation/ValuationResultView";
import { formatInt, formatMs, formatUsd } from "@/lib/format";
import { ERROR_LABEL, FILTER_SOURCE_LABEL, ROUTE_LABEL } from "@/lib/labels";
import { Answer } from "./Answer";
import { CypherPanel } from "./CypherPanel";
import type { SystemRun } from "./reducer";
import { SourceCard } from "./SourceCard";

function FiltersLine({ run }: { run: SystemRun }) {
  const f = run.filters;
  if (!f) return null;
  const entries = Object.entries(f.filters).filter(([, v]) => v !== null);
  return (
    <p className="text-xs text-muted-foreground" data-testid="filters-line">
      Bộ lọc ({FILTER_SOURCE_LABEL[f.source]}):{" "}
      {entries.length ? entries.map(([k, v]) => `${k}=${String(v)}`).join(" · ") : "không có"}
      {f.note === "filter_dropped" ? (
        <Badge tone="warning" className="ml-2">
          đã bỏ lọc vì 0 kết quả
        </Badge>
      ) : null}
    </p>
  );
}

/** Hiển thị đầy đủ một lượt của một hệ: route, bộ lọc, Cypher, định giá, câu trả lời stream, nguồn, usage, lỗi. */
export function RunView({
  run,
  scope = "",
  compact = false,
  onShowGraph,
}: {
  run: SystemRun;
  scope?: string;
  compact?: boolean;
  onShowGraph?: (listingId: number) => void;
}) {
  const sourceIds = useMemo(() => new Set(run.sources.map((s) => s.listing_id)), [run.sources]);
  const streaming = run.status === "streaming";
  const route = run.answer?.route ?? run.route?.route ?? null;

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        {route ? (
          <Badge tone="outline" data-testid="route-badge" title={run.route?.reason}>
            Route: {ROUTE_LABEL[route]}
          </Badge>
        ) : null}
        {run.answer?.fallback ? <Badge tone="warning">Fallback sang RAG</Badge> : null}
        {run.answer && run.answer.unsupported_citations.length > 0 ? (
          <Badge tone="warning">Trích dẫn không có trong nguồn: {run.answer.unsupported_citations.join(", ")}</Badge>
        ) : null}
        {streaming ? <Spinner label="Đang trả lời…" /> : null}
      </div>
      {run.route?.reason && !compact ? <p className="text-xs text-muted-foreground">Lý do route: {run.route.reason}</p> : null}
      <FiltersLine run={run} />
      {run.cypher ? <CypherPanel result={run.cypher} /> : null}
      {run.valuation ? <ValuationResultView result={run.valuation} compact={compact} /> : null}

      <section aria-label="Câu trả lời" aria-live="polite" aria-busy={streaming} data-testid="answer">
        {run.text ? <Answer text={run.text} sourceIds={sourceIds} scope={scope} /> : null}
      </section>

      {run.error ? (
        <Alert tone="error" data-testid="run-error">
          <strong>{ERROR_LABEL[run.error.code as keyof typeof ERROR_LABEL] ?? "Lỗi"}:</strong> {run.error.message}
        </Alert>
      ) : null}

      {run.usage ? (
        <p className="text-xs text-muted-foreground" data-testid="usage">
          {formatMs(run.usage.latency_ms)} · {formatInt(run.usage.input_tokens)} token vào ·{" "}
          {formatInt(run.usage.output_tokens)} token ra · {formatUsd(run.usage.usd)}
          {run.usage.cached ? " · từ cache" : ""}
        </p>
      ) : null}

      {run.sources.length > 0 ? (
        <section aria-label="Nguồn" className="grid gap-2">
          <h3 className="text-sm font-medium">Nguồn ({run.sources.length})</h3>
          <div className={compact ? "grid gap-2" : "grid gap-2 md:grid-cols-2"}>
            {run.sources.map((s) => (
              <SourceCard key={s.listing_id} card={s} scope={scope} onShowGraph={onShowGraph} />
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
