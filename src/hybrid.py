"""GraphRAG lai (Notebook 4): router chọn nguồn, ask_hybrid gộp ngữ cảnh và lui về Basic RAG khi cần."""

from time import perf_counter
from typing import Literal

from pydantic import BaseModel

from src.config import TOP_K
from src.graph import rows_context, text2cypher
from src.llm import ask_llm
from src.rag import format_context, generate, retrieve

ROUTES = ("basic_rag", "graph", "hybrid")


class Route(BaseModel):
    route: Literal["basic_rag", "graph", "hybrid"]
    reason: str


ROUTER_SYSTEM = """Phân loại câu hỏi về bất động sản vào đúng một nguồn trả lời:
- basic_rag: tìm tin theo mô tả tự do (tiện ích, vị trí 'gần...', đặc điểm trong bài đăng), không cần tính toán.
- graph: câu hỏi tổng hợp hoặc đa chặng trên dữ liệu có cấu trúc: đếm, trung bình, xếp hạng, so sánh
  quận/phường/tỉnh, hoặc lọc thuần theo cột (loại hình, quận, giá, diện tích, số phòng ngủ).
- hybrid: vừa có điều kiện cấu trúc (giá, quận, loại hình, số phòng) vừa có yêu cầu nằm trong mô tả tự do.

Ví dụ:
- "Căn hộ nào có view sông và hồ bơi?" -> basic_rag
- "Nhà gần trường học, hẻm xe hơi ở Gò Vấp" -> basic_rag
- "Quận nào ở Hà Nội có giá/m² trung bình cao nhất?" -> graph
- "Có bao nhiêu tin đất ở Đà Nẵng?" -> graph
- "Liệt kê căn hộ 2 phòng ngủ ở Quận 7 dưới 4 tỷ" -> graph
- "Căn hộ 2PN dưới 3 tỷ ở Quận 7 có hồ bơi" -> hybrid
- "Nhà ở Cầu Giấy dưới 10 tỷ có thang máy" -> hybrid"""


def router(question: str) -> dict:
    result = ask_llm(f"Câu hỏi: {question}", system=ROUTER_SYSTEM, output=Route)
    return {**result.data, "input_tokens": result.input_tokens, "output_tokens": result.output_tokens}


def ask_hybrid(question: str, k: int = TOP_K) -> dict:
    """router -> (basic_rag | graph | hybrid). Đồ thị lỗi hoặc rỗng thì lui về Basic RAG."""
    start = perf_counter()
    routed = router(question)
    route, tokens = routed["route"], [routed["input_tokens"], routed["output_tokens"]]
    graph = None
    if route in ("graph", "hybrid"):
        graph = text2cypher(question)
        tokens = [tokens[0] + graph["usage"][0], tokens[1] + graph["usage"][1]]
        if graph["error"] or graph["rows"].empty:
            graph = None
    fallback = route != "basic_rag" and graph is None

    contexts, sources = [], []
    if graph is not None:
        rows = graph["rows"]
        contexts.append(f"Cypher:\n{graph['cypher']}\n\nKẾT QUẢ TRUY VẤN ({len(rows)} dòng):\n{rows_context(rows)}")
        if "listing_id" in rows:
            sources += rows["listing_id"].dropna().astype(int).tolist()
    if route != "graph" or graph is None:
        found = retrieve(question, k)
        contexts += format_context(found["hits"])
        sources += found["hits"]["listing_id"].astype(int).tolist()
        tokens = [tokens[0] + sum(c.input_tokens for c in found["calls"]),
                  tokens[1] + sum(c.output_tokens for c in found["calls"])]
    answer = generate(question, contexts)
    return dict(system="hybrid", question=question, answer=answer.text,
                sources=list(dict.fromkeys(sources)), contexts=contexts, route=route,
                cypher=graph["cypher"] if graph else None, fallback=fallback,
                latency=perf_counter() - start,
                input_tokens=tokens[0] + answer.input_tokens, output_tokens=tokens[1] + answer.output_tokens)

