"""GraphRAG lai (Notebook 4): router chọn nguồn, ask_hybrid gộp ngữ cảnh và lui về Basic RAG khi cần."""

import logging
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Literal

from pydantic import BaseModel

from src.config import TOP_K
from src.graph import rows_context, text2cypher
from src.llm import ask_llm
from src.rag import ANSWER_SYSTEM, check_citations, format_context, generate, retrieve

ROUTES = ("basic_rag", "graph", "hybrid")
# Ngữ cảnh có kết quả Cypher: số tổng hợp không gắn với tin nào, nên không được đòi [Tin#ID] cho mọi ý
# (benchmark cho thấy mô hình khi đó tự đặt [Tin#1], [Tin#2] theo số dòng) và phải nói rõ đơn vị từng cột.
GRAPH_CONTEXT_SYSTEM = ANSWER_SYSTEM + """
Phần KẾT QUẢ TRUY VẤN là số liệu Neo4j tính trên toàn bộ dữ liệu: dùng trực tiếp và ghi nguồn [Cypher].
Chỉ ghi [Tin#ID] với ID có thật trong ngữ cảnh (cột listing_id hoặc dòng [Tin#ID]); không tự đánh số tin.
Đơn vị: price theo VND, price_bn theo tỷ đồng, price_m2 và avg_price_m2 theo triệu đồng/m², area theo m²."""
log = logging.getLogger(__name__)


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


def _graph(question: str) -> dict | None:
    """Text2Cypher; None khi Cypher lỗi, rỗng hoặc Neo4j không trả lời (khi đó dùng Basic RAG)."""
    try:
        return text2cypher(question)
    except Exception as error:  # mất kết nối Neo4j, hết thời gian chờ…: không làm hỏng cả câu trả lời
        log.warning("text2cypher lỗi, lui về Basic RAG: %s: %s", type(error).__name__, error)
        return None


def ask_hybrid(question: str, k: int = TOP_K) -> dict:
    """router -> (basic_rag | graph | hybrid). Đồ thị lỗi hoặc rỗng thì lui về Basic RAG.

    Truy xuất RAG (bộ lọc LLM + hybrid search) chạy song song với router và Text2Cypher, nên tuyến
    hybrid hay đường lui không phải chờ thêm một lượt LLM. Khi đồ thị đã trả lời đủ (tuyến graph),
    kết quả RAG bị bỏ nhưng token của nó vẫn được tính vào chi phí."""
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:
        rag_job = pool.submit(retrieve, question, k)
        routed = router(question)
        route, tokens = routed["route"], [routed["input_tokens"], routed["output_tokens"]]
        graph = None
        if route in ("graph", "hybrid"):
            graph = _graph(question)
            if graph is not None:
                tokens = [tokens[0] + graph["usage"][0], tokens[1] + graph["usage"][1]]
            if graph is not None and (graph["error"] or graph["rows"].empty):
                graph = None
        found = rag_job.result()
    fallback = route != "basic_rag" and graph is None
    tokens = [tokens[0] + sum(c.input_tokens for c in found["calls"]),
              tokens[1] + sum(c.output_tokens for c in found["calls"])]

    contexts, sources, note = [], [], ""
    if graph is not None:
        rows = graph["rows"]
        contexts.append(f"Cypher:\n{graph['cypher']}\n\nKẾT QUẢ TRUY VẤN ({len(rows)} dòng):\n{rows_context(rows)}")
        if "listing_id" in rows:
            sources += rows["listing_id"].dropna().astype(int).tolist()
    if route != "graph" or graph is None:
        contexts += format_context(found["hits"])
        sources += found["hits"]["listing_id"].astype(int).tolist()
        note = found.get("note", "")
    answer = generate(question, contexts, note, system=GRAPH_CONTEXT_SYSTEM if graph is not None else ANSWER_SYSTEM)
    sources = list(dict.fromkeys(sources))
    cited, unsupported = check_citations(answer.text, sources)
    return dict(system="hybrid", question=question, answer=answer.text,
                sources=sources, contexts=contexts, route=route, cited=cited, unsupported_citations=unsupported,
                filters=found["filters"].model_dump(exclude_none=True) if found.get("filters") else {}, note=note,
                cypher=graph["cypher"] if graph else None, fallback=fallback,
                latency=perf_counter() - start,
                input_tokens=tokens[0] + answer.input_tokens, output_tokens=tokens[1] + answer.output_tokens)
