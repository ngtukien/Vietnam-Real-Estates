import { useCallback, useEffect, useReducer, useRef } from "react";
import { apiStream, errorMessage } from "@/api/client";
import type { ChatRequest, CompareRequest, SystemId } from "@/api/schema";
import { chatReducer, initialChatState, type ChatState } from "./reducer";

type Start =
  | { path: "/api/chat"; body: ChatRequest; systems: SystemId[] }
  | { path: "/api/compare"; body: CompareRequest; systems: SystemId[] };

/** Gửi câu hỏi tới endpoint SSE và đưa từng sự kiện vào reducer. Gửi câu mới ⇒ huỷ stream cũ. */
export function useChatStream(): {
  state: ChatState;
  start: (req: Start) => Promise<ChatState | null>;
  stop: () => void;
} {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    dispatch({ type: "stream_closed" });
  }, []);

  const start = useCallback(async (req: Start): Promise<ChatState | null> => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    dispatch({ type: "start", question: req.body.question, systems: req.systems });
    // Giữ bản sao trạng thái cuối để trả về cho người gọi (lưu lịch sử) mà không đợi render.
    let local: ChatState = chatReducer(initialChatState, {
      type: "start",
      question: req.body.question,
      systems: req.systems,
    });
    try {
      for await (const event of apiStream(req.path, req.body, controller.signal)) {
        dispatch({ type: "event", event });
        local = chatReducer(local, { type: "event", event });
      }
      dispatch({ type: "stream_closed" });
      return chatReducer(local, { type: "stream_closed" });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return null;
      dispatch({ type: "request_error", message: errorMessage(err) });
      return null;
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, []);

  return { state, start, stop };
}
