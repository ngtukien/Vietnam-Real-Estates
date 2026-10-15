"""Router GraphRAG lai: chọn nguồn trả lời cho câu hỏi (`basic_rag` | `graph` | `hybrid` | `valuation`).

Luồng đầy đủ (chạy song song RAG với router/Text2Cypher, lui về Basic RAG) nằm ở `vre.qa.pipeline`.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from vre.graph.text2cypher import text2cypher
from vre.llm.client import ask_llm
from vre.rag.generate import ANSWER_SYSTEM

ROUTES = ("basic_rag", "graph", "hybrid", "valuation")
RouteName = Literal["basic_rag", "graph", "hybrid", "valuation"]
# Ngữ cảnh có kết quả Cypher: số tổng hợp không gắn với tin nào, nên không được đòi [Tin#ID] cho mọi ý
# (benchmark cho thấy mô hình khi đó tự đặt [Tin#1], [Tin#2] theo số dòng) và phải nói rõ đơn vị từng cột.
_TIN_RULE = "Mỗi nhận định phải kèm nguồn dạng [Tin#ID]; chép NGUYÊN VẸN dãy số ID trong ngữ cảnh."
assert _TIN_RULE in ANSWER_SYSTEM
GRAPH_CONTEXT_SYSTEM = ANSWER_SYSTEM.replace(_TIN_RULE, "Nhận định lấy từ một tin cụ thể kèm nguồn [Tin#ID] "
                                             "(chép nguyên ID); số liệu tổng hợp kèm nguồn [Cypher].") + """
Phần KẾT QUẢ TRUY VẤN là số liệu Neo4j tính trên toàn bộ dữ liệu: dùng trực tiếp và ghi nguồn [Cypher].
Chỉ ghi [Tin#ID] với ID có thật trong ngữ cảnh (cột listing_id hoặc dòng [Tin#ID]); không tự đánh số tin.
Đơn vị: price_vnd theo VND, *_m2_mil theo triệu đồng/m², area_m2 theo m²."""
log = logging.getLogger(__name__)


class Route(BaseModel):
    route: RouteName
    reason: str


ROUTER_SYSTEM = """Phân loại câu hỏi về bất động sản vào đúng một nguồn trả lời:
- basic_rag: tìm tin theo mô tả tự do (tiện ích, vị trí 'gần...', đặc điểm trong bài đăng), không cần tính toán.
- graph: câu hỏi tổng hợp hoặc đa chặng trên dữ liệu có cấu trúc: đếm, trung vị, xếp hạng, xu hướng theo tháng,
  so sánh quận/phường/tỉnh/dự án/đường, khu vực tương tự, hoặc lọc thuần theo cột (loại hình, quận, giá, diện tích, số phòng).
- hybrid: vừa có điều kiện cấu trúc (giá, quận, loại hình, số phòng) vừa có yêu cầu nằm trong mô tả tự do.
- valuation: hỏi định giá / giá hợp lý / "đáng bao nhiêu" cho MỘT bất động sản được mô tả (loại, khu vực, diện tích),
  hoặc hỏi một giá chào cụ thể có đắt không.

Ví dụ:
- "Căn hộ nào có view sông và hồ bơi?" -> basic_rag
- "Nhà gần trường học, hẻm xe hơi ở Gò Vấp" -> basic_rag
- "Quận nào ở Hà Nội có giá/m² trung vị cao nhất?" -> graph
- "Giá đất Thủ Đức thay đổi thế nào qua các tháng?" -> graph
- "Liệt kê căn hộ 2 phòng ngủ ở Quận 7 dưới 4 tỷ" -> graph
- "Căn hộ 2PN dưới 3 tỷ ở Quận 7 có hồ bơi" -> hybrid
- "Nhà ở Cầu Giấy dưới 10 tỷ có thang máy" -> hybrid
- "Nhà 60m² 3 phòng ngủ ở Gò Vấp đáng giá bao nhiêu?" -> valuation
- "Căn hộ 70m2 ở Quận 7 chào 4,5 tỷ có đắt không?" -> valuation"""


def router(question: str) -> dict:
    """`{route, reason, input_tokens, output_tokens, call}`."""
    result = ask_llm(f"Câu hỏi: {question}", system=ROUTER_SYSTEM, output=Route)
    return {**(result.data or {}), "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "call": result}


def safe_text2cypher(question: str) -> dict | None:
    """Text2Cypher; None khi Neo4j không trả lời (mất kết nối, hết giờ…) — người gọi lui về Basic RAG."""
    try:
        return text2cypher(question)
    except Exception as error:
        log.warning("text2cypher lỗi, lui về Basic RAG: %s: %s", type(error).__name__, error)
        return None
