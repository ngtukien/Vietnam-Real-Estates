import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { errorMessage } from "@/api/client";
import { useSubgraph } from "@/api/queries";
import type { NodeLabel } from "@/api/schema";
import { Alert, Spinner } from "@/components/ui/primitives";

const ForceGraph2D = lazy(() => import("react-force-graph-2d"));

const LABEL_COLOR: Record<NodeLabel, string> = {
  Listing: "#0f766e",
  PropertyType: "#7c3aed",
  Province: "#b45309",
  District: "#c2410c",
  Ward: "#ea580c",
  Street: "#4d7c0f",
  Project: "#be185d",
  Month: "#475569",
  Amenity: "#0369a1",
  Cluster: "#6d28d9",
  MarketStat: "#a16207",
};

interface FgNode {
  id: string;
  label: NodeLabel;
  name: string;
}

/** Đường đi trong KG quanh một tin (`/api/graph/subgraph?listing_id=`) + danh sách văn bản cho trình đọc màn hình. */
export function GraphPanel({ listingId }: { listingId: number }) {
  const { data, isLoading, error } = useSubgraph(listingId);
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(480);
  const hasData = data !== undefined;

  // Khung chỉ render khi đã có `data` ⇒ gắn observer lại lúc khung xuất hiện (deps rỗng thì box.current còn null).
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => entry && setWidth(Math.max(240, entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, [hasData]);

  const graph = useMemo(
    () => ({
      nodes: (data?.nodes ?? []).map((n) => ({ id: n.id, label: n.label, name: n.name })),
      links: (data?.edges ?? []).map((e) => ({ source: e.source, target: e.target, type: e.type })),
    }),
    [data],
  );

  return (
    <section aria-label={`Đồ thị tri thức quanh Tin#${listingId}`} className="grid gap-2" data-testid="graph-panel">
      <h3 className="text-sm font-medium">Đồ thị quanh Tin#{listingId}</h3>
      {isLoading ? <Spinner label="Đang tải đồ thị" /> : null}
      {error ? <Alert tone="error">{errorMessage(error)}</Alert> : null}
      {data ? (
        <>
          <div ref={box} className="h-72 overflow-hidden rounded-md border bg-card" aria-hidden>
            <Suspense fallback={<Spinner label="Đang tải thư viện đồ thị" />}>
              <ForceGraph2D
                graphData={graph}
                width={width}
                height={288}
                nodeLabel={(n) => `${(n as FgNode).label}: ${(n as FgNode).name}`}
                nodeColor={(n) => LABEL_COLOR[(n as FgNode).label] ?? "#64748b"}
                nodeRelSize={5}
                linkDirectionalArrowLength={3}
                linkLabel={(l) => String((l as { type: string }).type)}
                cooldownTicks={80}
              />
            </Suspense>
          </div>
          <details className="text-xs">
            <summary className="cursor-pointer">
              {data.nodes.length} nút · {data.edges.length} cạnh{data.truncated ? " (đã cắt bớt)" : ""} — xem dạng danh sách
            </summary>
            <ul className="mt-1 grid gap-0.5">
              {data.edges.map((e, i) => (
                <li key={i}>
                  <code>{e.source}</code> —{e.type}→ <code>{e.target}</code>
                </li>
              ))}
            </ul>
          </details>
        </>
      ) : null}
    </section>
  );
}
