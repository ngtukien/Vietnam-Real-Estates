"""Router GraphRAG lai: chọn nguồn trả lời cho câu hỏi (`basic_rag` | `graph` | `hybrid` | `valuation`).

Luồng đầy đủ (chạy song song RAG với router/Text2Cypher, lui về Basic RAG) nằm ở `vre.qa.pipeline`.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from vre.graph.text2cypher import text2cypher
from vre.llm.client import ask_llm
from vre.rag.generate import ANSWER_SYSTEM, NO_ECHO_RULE

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
Đơn vị: price_vnd theo VND, *_m2_mil theo triệu đồng/m², area_m2 theo m².
""" + NO_ECHO_RULE
log = logging.getLogger(__name__)


class Route(BaseModel):
    route: RouteName
    reason: str


ROUTER_SYSTEM = """Phân loại câu hỏi về bất động sản vào đúng một nguồn trả lời:
- basic_rag: TÌM TIN — một tin cụ thể hoặc danh sách tin theo điều kiện (loại hình, tỉnh/quận/phường, giá,
  diện tích, số phòng, tháng đăng) và/hoặc theo mô tả tự do (tiện ích, 'gần…', hướng, pháp lý); hoặc hỏi chi
  tiết của MỘT tin đã nêu (số WC, mặt tiền, hướng…), kể cả khi câu nhắc tên dự án/đường để mô tả căn đó.
  Lọc theo cột rồi liệt kê tin vẫn là basic_rag, không phải graph.
- graph: THỐNG KÊ trên nhiều tin: đếm số tin, trung bình/trung vị giá hay giá/m², xếp hạng khu vực, xu hướng
  theo tháng, so sánh quận/phường/tỉnh/loại hình/dự án/đường, câu nhiều bước (quận đắt nhất rồi phường đắt nhất
  trong quận đó), khu vực tương tự.
- hybrid: TÌM CÁC TIN NẰM TRONG một dự án có tên hoặc TRÊN một con đường có tên (thường kèm giá/số phòng):
  cần đồ thị dự án/đường để khoanh tin rồi mới tìm tin.
- valuation: hỏi định giá / giá hợp lý / "đáng bao nhiêu" / "nên rao bao nhiêu" cho MỘT bất động sản được mô tả
  (loại, khu vực, diện tích), hoặc hỏi một giá chào cụ thể có đắt không.

Ví dụ:
- "Mình thấy tin bán nhà 4 tầng ở Hoàng Mai giá 9 tỷ, nhà đó hướng gì?" -> basic_rag
- "Liệt kê nhà ở Gò Vấp 50-70m2 giá dưới 6 tỷ đăng tháng 2/2026" -> basic_rag
- "Căn hộ 2PN ở Quận 7 dưới 4 tỷ có hồ bơi" -> basic_rag
- "Nhà gần trường học, hẻm xe hơi ở Gò Vấp" -> basic_rag
- "Quận nào ở Hà Nội có giá/m² trung vị cao nhất?" -> graph
- "Có bao nhiêu tin bán căn hộ ở Hải Phòng tháng 1/2026?" -> graph
- "Giá đất Thủ Đức thay đổi thế nào qua các tháng?" -> graph
- "Ở Hà Đông, nhà phố hay căn hộ đắt hơn tính theo m²?" -> graph
- "Giá/m² trung vị căn hộ ở dự án Masteri Thảo Điền là bao nhiêu?" -> graph
- "Tìm căn hộ 2PN ở dự án The Sun Avenue dưới 4 tỷ" -> hybrid
- "Có nhà mặt tiền nào trên đường Lê Văn Lương giá dưới 15 tỷ không?" -> hybrid
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
