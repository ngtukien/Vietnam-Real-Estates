"""Mở đầu: bìa, lộ trình, ba mức câu hỏi, vì sao LLM thuần không đủ."""
from kit import MX, SIZE, W, Deck


def build(d: Deck) -> None:
    cover(d)
    agenda(d)
    questions(d)
    llm_vs_rag(d)


def excerpt(d: Deck, key: str, limit: int = 420) -> str:
    """Trích nguyên văn chuỗi trong results (rút gọn theo ranh giới từ nếu dài)."""
    value = d.V(key)
    if value is None:
        return d.A("⟨?⟩", f"thiếu {key}")
    text = " ".join(str(value).split())
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " …"
    return d.A(text, f"trích nguyên văn {key}")


def pick_pair(d: Deck, a: str, b: str) -> str | None:
    """Câu đầu tiên trong examples.answers có cả hệ a và hệ b (lấy từ lần chạy eval thật)."""
    rows = d.V("examples.json:answers") or []
    seen: dict[str, set] = {}
    for r in rows:
        seen.setdefault(r["id"], set()).add(r["system"])
    return next((q for q, sys in seen.items() if {a, b} <= sys), None)


def questions(d: Deck) -> None:
    gt = "benchmark_gt.json:items"
    lk, cs, ag = d.Q("LK01"), d.Q("CS01"), d.Q("AG04")
    s = d.slide("intro-questions", "Bài toán", "Ba mức câu hỏi đòi hỏi ba mức công nghệ khác nhau", notes=(
        f"Bộ benchmark có {d.N(gt + '|len')} câu chia chín nhóm, từ tra cứu một tin tới tổng hợp trên toàn bộ dữ liệu. "
        "Ba câu trên slide là ba mức khó. Câu tra cứu chỉ cần tìm đúng một tin, mã đầy đủ là "
        f"{d.N(gt + '[id=LK01].gt_ids[0]', 'str')}. Câu ràng buộc có nhiều điều kiện cứng cùng lúc, có "
        f"{d.N(gt + '[id=CS01].gt_ids|len')} tin thoả, nên embedding không đủ mà phải lọc. Câu tổng hợp phải gom "
        f"{d.N(gt + '[id=AG04].gt_rows[0].n')} tin căn hộ của một quận để tính trung vị, không tin đơn lẻ nào chứa đáp án. "
        "Đáp án chuẩn của mọi câu tính bằng SQL trên bảng sạch, không nhờ LLM. "
        "Không có một công cụ nào hợp cho cả ba; phần còn lại của bài đi lần lượt từng mức."))
    levels = [
        ("Mức 1 · Tra cứu", lk, "blue", "Basic RAG",
         [("Đáp án chuẩn  ", {"bold": True}), (d.A("đúng 1 tin", "đề bài tra cứu một tin") + f" · Tin#…{d.N(gt + '[id=LK01].gt_ids[0]', 'last6')}", {})]),
        ("Mức 2 · Ràng buộc", cs, "teal", "RAG + lọc metadata",
         [("Đáp án chuẩn  ", {"bold": True}), (f"{d.N(gt + '[id=CS01].gt_ids|len')} tin thoả mọi điều kiện", {})]),
        ("Mức 3 · Tổng hợp", ag, "amber", "Knowledge Graph",
         [("Đáp án chuẩn  ", {"bold": True}),
          (f"{d.T(gt + '[id=AG04].gt_value')} · {d.N(gt + '[id=AG04].gt_rows[0].median_price_m2_mil', 'dec1')} "
           f"triệu/m², trên {d.N(gt + '[id=AG04].gt_rows[0].n')} tin", {})]),
    ]
    cw = (W - 2 * MX - 2 * 0.35) / 3
    for i, (tag, q, col, tech, answer) in enumerate(levels):
        x = MX + i * (cw + 0.35)
        s.rect(x, 1.9, cw, 4.35, fill="soft", line=None, radius=0.04)
        s.rect(x, 1.9, cw, 0.09, fill=col, line=None)
        s.text(x + 0.3, 2.15, cw - 0.6, 0.4, tag, size=SIZE["caption"], bold=True, color=col)
        s.text(x + 0.3, 2.6, cw - 0.6, 1.9, "“" + d.A(q["question"], f"đề bài benchmark {q['id']}") + "”",
               size=18, color="ink")
        s.text(x + 0.3, 4.55, cw - 0.6, 0.9, [answer], size=SIZE["small"])
        s.rect(x + 0.3, 5.55, cw - 0.6, 0.48, fill=f"{col}_bg", line=None, radius=0.3)
        s.text(x + 0.3, 5.55, cw - 0.6, 0.48, tech, size=SIZE["small"], bold=True, color=col, align="center",
               anchor="middle")
    s.caption(f"Ba câu lấy từ benchmark {d.N(gt + '|len')} câu; đáp án chuẩn tính bằng SQL trên "
              f"{d.N('benchmark_gt.json:data_fingerprint.n_listings')} tin sạch.")


def llm_vs_rag(d: Deck) -> None:
    qid = pick_pair(d, "llm_only", "basic_rag")
    ans = "examples.json:answers"
    q = d.Q(qid) if qid else {"id": "?", "question": "⟨?⟩"}
    s = d.slide("intro-llm-vs-rag", "Vì sao cần RAG", "LLM thuần không biết dữ liệu riêng nên không trả lời được, "
                "cũng không dẫn được nguồn", notes=(
        "Ba điểm yếu của LLM thuần: bịa số, kiến thức đóng băng ở thời điểm huấn luyện, và không biết dữ liệu riêng "
        "của nhóm nên không dẫn nguồn được. Hai cột là câu trả lời thật của cùng một mô hình cho cùng một câu hỏi trong "
        "lượt chạy benchmark: bên trái không có ngữ cảnh, bên phải là Basic RAG của nhóm, kèm mã tin kiểm chứng được. "
        "Fine-tuning dạy phong cách; RAG đưa sự kiện mới vào và cho phép dẫn nguồn, hợp với tin đăng thay đổi mỗi ngày. "
        "Chuyển sang chặng 1: RAG hoạt động thế nào."))
    s.text(MX, 1.85, W - 2 * MX, 0.8, [[("Câu hỏi  ", {"bold": True, "color": "teal"}),
           ("“" + d.A(q["question"], f"đề bài benchmark {q['id']}") + "”", {"color": "ink"})]], size=20)
    cw = (W - 2 * MX - 0.4) / 2
    for i, (system, col) in enumerate((("llm_only", "muted"), ("basic_rag", "blue"))):
        x = MX + i * (cw + 0.4)
        s.rect(x, 2.75, cw, 3.6, fill="soft", line=None, radius=0.04)
        s.rect(x, 2.75, 0.08, 3.6, fill=col, line=None)
        label = {"llm_only": "LLM thuần · không ngữ cảnh", "basic_rag": "Basic RAG · trích dẫn mã tin"}[system]
        s.text(x + 0.3, 2.9, cw - 0.5, 0.4, label, size=SIZE["caption"], bold=True, color=col)
        s.text(x + 0.3, 3.35, cw - 0.5, 2.9, excerpt(d, f"{ans}[id={q['id']},system={system}].answer", 380),
               size=SIZE["small"], color="ink")
    s.caption(d.A("Đầu ra thật của cùng một mô hình sinh trong lượt chạy eval ", "chú thích") +
              f"({d.T('eval_summary.json:llm.generator')}); câu {q['id']}.")


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
