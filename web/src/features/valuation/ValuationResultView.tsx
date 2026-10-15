import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ValuationResult } from "@/api/schema";
import { Badge, DataTable } from "@/components/ui/primitives";
import { formatDecimal, formatPercent, formatPriceM2, formatVnd } from "@/lib/format";
import { VERDICT_LABEL } from "@/lib/labels";

export function ValuationResultView({ result, compact = false }: { result: ValuationResult; compact?: boolean }) {
  const shap = [...result.shap].sort((a, b) => Math.abs(b.contribution_log) - Math.abs(a.contribution_log));
  return (
    <section aria-label="Kết quả định giá" className="grid gap-3" data-testid="valuation-result">
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-md border p-3">
          <p className="text-xs text-muted-foreground">Giá dự báo</p>
          <p className="text-xl font-semibold">{formatVnd(result.predicted_price_vnd)}</p>
          <p className="text-xs text-muted-foreground">{formatPriceM2(result.predicted_price_m2_mil)}</p>
        </div>
        <div className="rounded-md border p-3">
          <p className="text-xs text-muted-foreground">Khoảng dự báo (p10–p90)</p>
          <p className="text-base font-medium">
            {formatVnd(result.interval_price_vnd.low)} – {formatVnd(result.interval_price_vnd.high)}
          </p>
        </div>
        <div className="rounded-md border p-3">
          <p className="text-xs text-muted-foreground">Trung vị phường (baseline)</p>
          <p className="text-base font-medium">{formatPriceM2(result.baseline_ward_median_price_m2_mil)}</p>
          <p className="text-xs text-muted-foreground">{result.comparables_n} tin so sánh</p>
        </div>
      </div>
      {result.verdict ? (
        <p className="text-sm">
          Nhận định giá chào:{" "}
          <Badge tone={result.verdict === "fair" ? "default" : "warning"} data-testid="verdict">
            {VERDICT_LABEL[result.verdict]}
          </Badge>
        </p>
      ) : null}
      <p className="text-xs text-muted-foreground">
        Model <code>{result.model.variant}</code> · MAPE test {formatPercent(result.model.test_mape)} · huấn luyện{" "}
        {result.model.trained_at}
      </p>
      {!compact && shap.length > 0 ? (
        <div className="grid gap-2">
          <h3 className="text-sm font-medium">Đóng góp SHAP vào log(giá/m²)</h3>
          <div className="h-64" aria-hidden>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart accessibilityLayer={false} data={shap} layout="vertical" margin={{ left: 24, right: 16 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tickFormatter={(v: number) => formatDecimal(v, 2)} />
                <YAxis type="category" dataKey="feature" width={130} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v) => formatDecimal(Number(v), 4)} />
                <Bar dataKey="contribution_log" name="Đóng góp">
                  {shap.map((s) => (
                    <Cell key={s.feature} fill={s.contribution_log >= 0 ? "var(--chart-1)" : "var(--chart-2)"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <DataTable
            caption="Đóng góp SHAP"
            columns={["Đặc trưng", "Giá trị", "Đóng góp log"]}
            rows={shap.map((s) => [s.feature, s.value === null ? "—" : String(s.value), formatDecimal(s.contribution_log, 4)])}
          />
        </div>
      ) : null}
    </section>
  );
}
