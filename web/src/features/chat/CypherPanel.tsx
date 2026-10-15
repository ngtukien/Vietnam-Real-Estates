import { AlertTriangle } from "lucide-react";
import type { CypherResult } from "@/api/schema";
import { Alert, DataTable } from "@/components/ui/primitives";

function cell(v: string | number | boolean | null): string {
  if (v === null) return "null";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
}

/** Câu Cypher cuối + bảng kết quả + LỖI hiển thị rõ (sửa bug cũ "ẩn lỗi Cypher"). */
export function CypherPanel({ result }: { result: CypherResult }) {
  return (
    <section aria-label="Truy vấn Cypher" className="grid gap-2" data-testid="cypher-panel">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <h3 className="text-sm font-medium text-foreground">Cypher</h3>
        <span>
          {result.attempts} lần thử · {result.rows.length} dòng
        </span>
      </div>
      <pre className="overflow-x-auto rounded-md bg-muted p-3 text-xs" tabIndex={0} aria-label="Câu lệnh Cypher">
        <code>{result.cypher}</code>
      </pre>
      {result.error ? (
        <Alert tone="error" data-testid="cypher-error">
          <span className="flex items-start gap-2">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
            <span>
              <strong>Cypher lỗi:</strong> {result.error}
            </span>
          </span>
        </Alert>
      ) : null}
      {result.columns.length > 0 && result.rows.length > 0 ? (
        <DataTable
          caption="Kết quả Cypher"
          columns={result.columns}
          rows={result.rows.map((r) => result.columns.map((c) => cell(r[c] ?? null)))}
          className="max-h-72 overflow-y-auto"
        />
      ) : !result.error ? (
        <p className="text-xs text-muted-foreground">Truy vấn không trả dòng nào.</p>
      ) : null}
    </section>
  );
}
