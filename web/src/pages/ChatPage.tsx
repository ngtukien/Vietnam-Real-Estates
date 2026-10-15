import { History, Send, Square, Trash2 } from "lucide-react";
import { useId, useState } from "react";
import { useSearchParams } from "react-router";
import { useInfo } from "@/api/queries";
import { EMPTY_FILTERS, SYSTEM_IDS, type QueryFilters, type SystemId } from "@/api/schema";
import { Button } from "@/components/ui/button";
import { Alert, Card, CardContent, CardHeader, CardTitle, Input, Label, Textarea } from "@/components/ui/primitives";
import { FilterPanel } from "@/features/chat/FilterPanel";
import { GraphPanel } from "@/features/chat/GraphPanel";
import { clearHistory, loadHistory, pushHistory, type HistoryEntry } from "@/features/chat/history";
import { filtersOrNull } from "@/features/chat/reducer";
import { RunView } from "@/features/chat/RunView";
import { useChatStream } from "@/features/chat/useChatStream";
import { SYSTEM_LABEL } from "@/lib/labels";

const MAX_CHARS = 500;

export default function ChatPage() {
  const id = useId();
  const [params] = useSearchParams();
  // B8: `?q=` CHỈ điền sẵn ô nhập; không bao giờ tự gửi.
  const [question, setQuestion] = useState(() => (params.get("q") ?? "").slice(0, MAX_CHARS));
  const [system, setSystem] = useState<SystemId>("hybrid");
  const [filters, setFilters] = useState<QueryFilters>(EMPTY_FILTERS);
  const [topK, setTopK] = useState(5);
  const [useRerank, setUseRerank] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory);
  const [graphListing, setGraphListing] = useState<number | null>(null);
  const { data: info } = useInfo();
  const { state, start, stop } = useChatStream();
  const run = state.runs[system] ?? Object.values(state.runs)[0];
  const maxChars = info?.limits.max_question_chars ?? MAX_CHARS;
  const trimmed = question.trim();
  const availability = new Map(info?.systems.map((s) => [s.id, s]) ?? []);

  async function submit() {
    if (!trimmed || state.active) return;
    setGraphListing(null);
    const final = await start({
      path: "/api/chat",
      systems: [system],
      body: { question: trimmed, system, filters: filtersOrNull(filters), top_k: topK, use_rerank: useRerank, stream_tokens: true },
    });
    const finalRun = final?.runs[system];
    if (finalRun?.answer) {
      setHistory((h) =>
        pushHistory(h, {
          id: final?.requestId ?? `${Date.now()}`,
          at: new Date().toISOString(),
          question: trimmed,
          system,
          answer: finalRun.answer?.answer ?? "",
        }),
      );
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[18rem_1fr]">
      <h1 className="sr-only">Hỏi đáp bất động sản</h1>
      <aside className="grid content-start gap-4" aria-label="Tuỳ chọn hỏi đáp">
        <FilterPanel value={filters} onChange={setFilters} />
        <section aria-labelledby={`${id}-hist`} className="grid gap-2 rounded-md border p-3">
          <div className="flex items-center justify-between">
            <h2 id={`${id}-hist`} className="flex items-center gap-1 text-sm font-medium">
              <History className="size-4" aria-hidden /> Lịch sử
            </h2>
            <Button
              size="sm"
              variant="ghost"
              disabled={!history.length}
              onClick={() => {
                clearHistory();
                setHistory([]);
              }}
            >
              <Trash2 aria-hidden /> Xoá
            </Button>
          </div>
          {history.length === 0 ? (
            <p className="text-xs text-muted-foreground">Chưa có câu hỏi nào (lưu tối đa 50 câu trên máy này).</p>
          ) : (
            <ul className="grid max-h-80 gap-1 overflow-y-auto">
              {history.map((h) => (
                <li key={h.id}>
                  <button
                    type="button"
                    className="w-full rounded px-2 py-1 text-left text-xs hover:bg-muted"
                    onClick={() => {
                      setQuestion(h.question);
                      setSystem(h.system);
                    }}
                    title={h.answer.slice(0, 300)}
                  >
                    <span className="line-clamp-2">{h.question}</span>
                    <span className="text-muted-foreground">{SYSTEM_LABEL[h.system]}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      </aside>

      <div className="grid content-start gap-4">
        <Card>
          <CardContent className="pt-4">
            <form
              className="grid gap-3"
              onSubmit={(e) => {
                e.preventDefault();
                void submit();
              }}
            >
              <div className="grid gap-1.5">
                <Label htmlFor={`${id}-q`}>Câu hỏi</Label>
                <Textarea
                  id={`${id}-q`}
                  value={question}
                  maxLength={maxChars}
                  placeholder="Ví dụ: căn hộ 2PN ở Quận 7 dưới 4 tỷ?"
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                      e.preventDefault();
                      void submit();
                    }
                  }}
                  aria-describedby={`${id}-q-hint`}
                />
                <p id={`${id}-q-hint`} className="text-xs text-muted-foreground">
                  Enter để gửi, Shift+Enter xuống dòng · {question.length}/{maxChars} ký tự
                </p>
              </div>
              <fieldset className="flex flex-wrap items-center gap-x-4 gap-y-2">
                <legend className="mb-1 text-sm font-medium">Hệ thống</legend>
                {SYSTEM_IDS.map((s) => {
                  const avail = availability.get(s);
                  return (
                    <label key={s} className="flex items-center gap-1.5 text-sm" title={avail?.unavailable_reason ?? undefined}>
                      <input
                        type="radio"
                        name={`${id}-system`}
                        value={s}
                        checked={system === s}
                        onChange={() => setSystem(s)}
                        className="accent-primary"
                      />
                      {SYSTEM_LABEL[s]}
                      {avail && !avail.available ? <span className="text-xs text-destructive">(chưa sẵn sàng)</span> : null}
                    </label>
                  );
                })}
              </fieldset>
              <div className="flex flex-wrap items-end gap-4">
                <div className="grid gap-1.5">
                  <Label htmlFor={`${id}-topk`}>Số tin (top-k)</Label>
                  <Input
                    id={`${id}-topk`}
                    type="number"
                    min={1}
                    max={20}
                    value={topK}
                    className="w-24"
                    onChange={(e) => setTopK(Math.min(20, Math.max(1, Number(e.target.value) || 5)))}
                  />
                </div>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={useRerank}
                    onChange={(e) => setUseRerank(e.target.checked)}
                    className="accent-primary"
                  />
                  Rerank
                </label>
                <div className="ml-auto flex gap-2">
                  {state.active ? (
                    <Button variant="outline" onClick={stop}>
                      <Square aria-hidden /> Dừng
                    </Button>
                  ) : null}
                  <Button type="submit" disabled={!trimmed || state.active}>
                    <Send aria-hidden /> Gửi
                  </Button>
                </div>
              </div>
            </form>
          </CardContent>
        </Card>

        {state.requestError ? <Alert tone="error">{state.requestError}</Alert> : null}

        {run && state.question ? (
          <Card>
            <CardHeader>
              <CardTitle>{state.question}</CardTitle>
              <p className="text-xs text-muted-foreground">
                {SYSTEM_LABEL[run.system]}
                {state.requestId ? ` · mã ${state.requestId}` : ""}
              </p>
            </CardHeader>
            <CardContent>
              <RunView run={run} onShowGraph={setGraphListing} />
            </CardContent>
          </Card>
        ) : null}

        {graphListing !== null ? (
          <Card>
            <CardContent className="pt-4">
              <GraphPanel listingId={graphListing} />
            </CardContent>
          </Card>
        ) : null}
      </div>
    </div>
  );
}
