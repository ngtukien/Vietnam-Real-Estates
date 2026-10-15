import { Send } from "lucide-react";
import { useId, useState } from "react";
import { SYSTEM_IDS, type SystemId } from "@/api/schema";
import { Button } from "@/components/ui/button";
import { Alert, Badge, Card, CardContent, CardHeader, CardTitle, Label, Textarea } from "@/components/ui/primitives";
import { emptyRun } from "@/features/chat/reducer";
import { RunView } from "@/features/chat/RunView";
import { useChatStream } from "@/features/chat/useChatStream";
import { SYSTEM_LABEL } from "@/lib/labels";

export default function ComparePage() {
  const id = useId();
  const [question, setQuestion] = useState("");
  const [systems, setSystems] = useState<SystemId[]>([...SYSTEM_IDS]);
  const { state, start } = useChatStream();
  const trimmed = question.trim();
  const shown = state.question ? (Object.keys(state.runs) as SystemId[]) : [];

  return (
    <div className="grid gap-4">
      <h1 className="text-lg font-semibold">So sánh 4 hệ thống trên cùng câu hỏi</h1>
      <Card>
        <CardContent className="pt-4">
          <form
            className="grid gap-3"
            onSubmit={(e) => {
              e.preventDefault();
              if (!trimmed || state.active || systems.length === 0) return;
              void start({ path: "/api/compare", systems, body: { question: trimmed, systems } });
            }}
          >
            <div className="grid gap-1.5">
              <Label htmlFor={`${id}-q`}>Câu hỏi</Label>
              <Textarea id={`${id}-q`} value={question} maxLength={500} onChange={(e) => setQuestion(e.target.value)} />
            </div>
            <fieldset className="flex flex-wrap gap-4">
              <legend className="mb-1 text-sm font-medium">Hệ thống so sánh</legend>
              {SYSTEM_IDS.map((s) => (
                <label key={s} className="flex items-center gap-1.5 text-sm">
                  <input
                    type="checkbox"
                    className="accent-primary"
                    checked={systems.includes(s)}
                    onChange={(e) =>
                      setSystems((prev) =>
                        e.target.checked ? SYSTEM_IDS.filter((x) => prev.includes(x) || x === s) : prev.filter((x) => x !== s),
                      )
                    }
                  />
                  {SYSTEM_LABEL[s]}
                </label>
              ))}
            </fieldset>
            <div>
              <Button type="submit" disabled={!trimmed || state.active || systems.length === 0}>
                <Send aria-hidden /> So sánh
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {state.requestError ? <Alert tone="error">{state.requestError}</Alert> : null}
      {state.compareDone ? (
        <p className="text-sm" role="status" data-testid="compare-done">
          Xong: {state.compareDone.systems_ok.length} hệ thành công
          {state.compareDone.systems_failed.length
            ? `, lỗi: ${state.compareDone.systems_failed.map((s) => SYSTEM_LABEL[s]).join(", ")}`
            : ""}
        </p>
      ) : null}

      {shown.length > 0 ? (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {shown.map((s) => {
            const run = state.runs[s] ?? emptyRun(s);
            return (
              <Card key={s} data-testid={`compare-col-${s}`}>
                <CardHeader>
                  <div className="flex items-center justify-between gap-2">
                    <CardTitle as="h2">{SYSTEM_LABEL[s]}</CardTitle>
                    <Badge tone={run.status === "error" ? "destructive" : run.status === "done" ? "outline" : "secondary"}>
                      {run.status === "streaming" ? "đang chạy" : run.status === "error" ? "lỗi" : "xong"}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <RunView run={run} scope={`${s}-`} compact />
                </CardContent>
              </Card>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
