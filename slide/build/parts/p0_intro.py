"""Mở đầu: bìa, lộ trình, ba mức câu hỏi, vì sao LLM thuần không đủ."""
from kit import MX, SIZE, W, Deck


def build(d: Deck) -> None:
    cover(d)
    agenda(d)


def cover(d: Deck) -> None:
    s = d.slide("intro-cover", "", "", dark=True, notes=(
        "Giới thiệu nhóm và đề tài. Mua nhà là quyết định lớn, còn thông tin lại nằm rải rác trong hàng triệu tin "
        f"đăng. Bản này chạy trên toàn bộ dataset: {d.N('ingest_report.json:raw_rows')} dòng thô, còn "
        f"{d.N('ingest_report.json:dedup.kept')} tin sau khi bỏ trùng, cắt thành "
        f"{d.N('night_report.json:counters.chunks')} chunk trong Qdrant và một đồ thị Neo4j. "
        "Chuyển sang slide sau: lộ trình của bài."))
    s.rect(MX, 2.05, 0.12, 2.3, fill="teal", line=None)
    s.text(MX + 0.4, 1.9, 11, 0.5, d.A("Nhóm 7 · RAG và Knowledge Graph", "tên nhóm"), size=SIZE["small"], color="teal_bg", bold=True)
    s.text(MX + 0.4, 2.4, 11.5, 1.2, "Từ RAG đến GraphRAG", size=54, bold=True, color="white")
    s.text(MX + 0.4, 3.55, 11.5, 0.8, "Trợ lý hỏi đáp thị trường bất động sản Việt Nam", size=26, color="CBD5E1")
    stats = [(f"{d.N('ingest_report.json:raw_rows', 'mil1')} triệu", "tin đăng thô"),
             (f"{d.N('ingest_report.json:dedup.kept', 'mil2')} triệu", "tin sau khi bỏ trùng"),
             (f"{d.N('night_report.json:counters.chunks', 'mil2')} triệu", "chunk trong Qdrant")]
    for i, (v, lab) in enumerate(stats):
        x = MX + 0.4 + i * 3.4
        s.text(x, 4.75, 3.2, 0.7, v, size=32, bold=True, color="FBBF24")
        s.text(x, 5.4, 3.2, 0.4, lab, size=SIZE["small"], color="CBD5E1")
    s.text(MX + 0.4, 6.25, 12, 0.4, d.A("Dữ liệu: tinixai/vietnam-real-estates (Hugging Face, CC BY-NC 4.0)", "tên giấy phép"),
           size=SIZE["caption"], color="94A3B8")


def agenda(d: Deck) -> None:
    s = d.slide("intro-agenda", "Lộ trình", "Bốn chặng đi từ tra cứu văn bản đến suy luận trên đồ thị", notes=(
        "Mỗi chặng trả lời một câu hỏi: nạp dữ liệu thế nào, tìm và trả lời thế nào, biểu diễn quan hệ thế nào, "
        "ghép lại và đo ra sao. Phần cuối là hai mở rộng làm thêm trên dữ liệu đầy đủ: mô-đun định giá và webapp "
        "so sánh bốn hệ. Nhãn nhỏ ở góc trên mỗi slide cho biết đang ở chặng nào."))
    stages = [("Chặng 1", "Nền tảng RAG", "Nạp và lập chỉ mục dữ liệu", "Nguyễn Thành Trung", "blue"),
              ("Chặng 2", "Truy xuất và sinh", "Tìm đúng tin, trả lời có nguồn", "Nguyễn Minh Toàn", "teal"),
              ("Chặng 3", "Knowledge Graph", "Biểu diễn quan hệ bằng đồ thị", "Nguyễn Tự Kiên", "amber"),
              ("Chặng 4", "GraphRAG và đánh giá", "Ghép hai nguồn và đo kết quả", "Nguyễn Duy Nghĩa", "violet")]
    bw = (W - 2 * MX - 3 * 0.3) / 4
    for i, (tag, name, q, who, col) in enumerate(stages):
        x = MX + i * (bw + 0.3)
        s.rect(x, 1.95, bw, 3.4, fill=f"{col}_bg", line=None, radius=0.05)
        s.rect(x, 1.95, bw, 0.1, fill=col, line=None)
        s.text(x + 0.25, 2.25, bw - 0.5, 0.4, tag, size=SIZE["caption"], bold=True, color=col)
        s.text(x + 0.25, 2.7, bw - 0.5, 1.0, name, size=24, bold=True, color="ink")
        s.text(x + 0.25, 3.75, bw - 0.5, 0.9, q, size=SIZE["small"])
        s.text(x + 0.25, 4.75, bw - 0.5, 0.4, who, size=SIZE["caption"], color="muted")
    s.rect(MX, 5.6, W - 2 * MX, 0.85, fill="soft", line=None, radius=0.1)
    s.text(MX + 0.3, 5.75, W - 2 * MX - 0.6, 0.6, [[("Mở rộng  ", {"bold": True, "color": "teal"}),
           ("định giá bất động sản bằng XGBoost · webapp so sánh bốn hệ trên cùng câu hỏi", {})]],
           size=SIZE["small"], anchor="middle")
