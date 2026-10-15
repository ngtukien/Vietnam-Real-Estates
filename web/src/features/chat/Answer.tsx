import { useMemo } from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { linkCitations } from "./reducer";
import { sourceAnchorId } from "./anchors";

/** Câu trả lời markdown (bảng GFM). `[Tin#id]` → liên kết cuộn + focus tới thẻ nguồn tương ứng. HTML thô bị bỏ. */
export function Answer({ text, sourceIds, scope = "" }: { text: string; sourceIds: ReadonlySet<number>; scope?: string }) {
  const md = useMemo(() => linkCitations(text, sourceIds), [text, sourceIds]);
  const components: Components = useMemo(
    () => ({
      a: ({ href, children }) => {
        const m = href ? /^#src-(\d+)$/.exec(href) : null;
        if (m?.[1]) {
          const target = sourceAnchorId(scope, Number(m[1]));
          return (
            <a
              href={`#${target}`}
              className="font-mono text-xs"
              onClick={(e) => {
                e.preventDefault();
                const el = document.getElementById(target);
                el?.scrollIntoView({ behavior: "smooth", block: "center" });
                el?.focus({ preventScroll: true });
              }}
            >
              {children}
            </a>
          );
        }
        return (
          <a href={href} target="_blank" rel="noreferrer noopener">
            {children}
          </a>
        );
      },
    }),
    [scope],
  );
  return (
    <div className="prose-answer text-sm">
      <Markdown remarkPlugins={[remarkGfm]} components={components} skipHtml>
        {md}
      </Markdown>
    </div>
  );
}
