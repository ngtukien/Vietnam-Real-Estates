"""Chặng 1: lý thuyết RAG, dữ liệu 3,5 triệu tin + EDA, chunking, embedding, thực hành 1, scale, kiến trúc."""
from pptx.enum.shapes import MSO_SHAPE

from kit import MX, SIZE, W, Deck

SEC = "Chặng 1 · Nền tảng RAG"
RAW, KEPT = "ingest_report.json:raw_rows", "ingest_report.json:dedup.kept"
CHUNKS = "night_report.json:counters.chunks"
E5 = "embedding_benchmark.json:[model=intfloat/multilingual-e5-large]"
DROP = "=({ingest_report.json:raw_rows}-{ingest_report.json:dedup.kept})/{ingest_report.json:raw_rows}"
CFG = "tham số trong src/vre/config.py"


def build(d: Deck) -> None:
    rag_theory(d)
    rag_two_phases(d)
    data_overview(d)
    eda_price(d)
    eda_geo(d)
    eda_trend(d)
    eda_quality(d)
    rag_two_kinds(d)
    rag_chunking(d)
    emb_theory(d)
    emb_choice(d)
    rag_practice_1(d)
    scale_3_5m(d)
    arch_overview(d)


# ---- helper riêng của module ----
def kpi(s, x, y, w, value, label, accent="teal", h=1.45, size=36):
    """Như s.stat nhưng chọn được cỡ số (giá trị dài như khoảng tháng)."""
    s.rect(x, y, w, h, fill="soft", line=None, radius=0.06)
    s.text(x + 0.2, y + 0.12, w - 0.4, 0.7, value, size=size, bold=True, color=accent)
    s.text(x + 0.2, y + 0.85, w - 0.4, h - 0.9, label, size=SIZE["caption"], color="text")


def arrow(s, x, y, w, h, kind="right"):
    shape = {"right": MSO_SHAPE.RIGHT_ARROW, "down": MSO_SHAPE.DOWN_ARROW, "up": MSO_SHAPE.UP_ARROW}[kind]
    s.rect(x, y, w, h, fill="muted", line=None, shape=shape)


def box(s, x, y, w, h, title, sub, fill="blue_bg"):
    s.rect(x, y, w, h, fill=fill, line=None, radius=0.08)
    s.text(x + 0.1, y + 0.1, w - 0.2, 0.4, title, size=18, bold=True, color="ink", align="center")
    s.text(x + 0.1, y + 0.5, w - 0.2, h - 0.55, sub, size=SIZE["caption"], align="center")


def grid(s, x, y, w, rows, col_w, size=SIZE["small"], row_h=0.45, highlight_row=None):
    """Bảng dựng bằng hộp chữ rời: mỗi ô một shape, để map số khớp đúng ô (s.table gộp chữ cả bảng vào một frame)."""
    for r, row in enumerate(rows):
        yy = y + r * row_h
        fill = "navy" if r == 0 else ("teal_bg" if r == highlight_row else ("white" if r % 2 else "soft"))
        s.rect(x, yy, w, row_h, fill=fill, line=None)
        cx = x
        for c, val in enumerate(row):
            s.text(cx + 0.08, yy, col_w[c] - 0.16, row_h, val, size=size, bold=r == 0, anchor="middle",
                   color="white" if r == 0 else "ink", align="left" if c == 0 else "center")
            cx += col_w[c]


# ---- slide ----
def rag_theory(d: Deck) -> None:
    s = d.slide("rag-theory", SEC, "RAG: tra bằng chứng trước, rồi mới để LLM trả lời", notes=(
        "LLM thuần giống thí sinh thi đóng sách: chỉ dựa vào những gì đã học, kiến thức đóng băng từ lúc huấn luyện "
        "và không biết dữ liệu riêng của nhóm. RAG cho mô hình mở sách. Retrieval tìm các đoạn liên quan trong kho tin "
        "đăng. Augmentation ghép các đoạn đó vào prompt cùng chỉ thị. Generation để LLM viết câu trả lời dựa trên ngữ "
        "cảnh và ghi mã tin làm nguồn. Vì sao không fine-tune: fine-tune chỉnh cách diễn đạt, không hợp để chứa giá "
        "thay đổi hằng ngày và không dẫn nguồn được cho từng con số. Muốn cập nhật thì nạp lại chỉ mục, không huấn "
        "luyện lại. Chuyển sang slide sau: quy trình này chạy trong hai pha."))
    parts = [("R", "Retrieval", "tìm đoạn tin đăng liên quan đến câu hỏi", "blue"),
             ("A", "Augmentation", "ghép các đoạn đó vào prompt cùng chỉ thị", "teal"),
             ("G", "Generation", "LLM viết câu trả lời, kèm mã tin làm nguồn", "violet")]
    bw = (W - 2 * MX - 2 * 0.4) / 3
    for i, (letter, name, desc, col) in enumerate(parts):
        x = MX + i * (bw + 0.4)
        s.rect(x, 2.0, bw, 3.3, fill=f"{col}_bg", line=None, radius=0.05)
        s.text(x + 0.35, 2.2, bw - 0.7, 1.1, letter, size=60, bold=True, color=col)
        s.text(x + 0.35, 3.3, bw - 0.7, 0.5, name, size=24, bold=True, color="ink")
        s.text(x + 0.35, 3.9, bw - 0.7, 1.3, desc, size=SIZE["body"])
    s.text(MX, 5.7, W - 2 * MX, 0.6, [[("Ý chính  ", {"bold": True, "color": "teal"}),
           ("mô hình được đưa đúng tin đăng rồi mới trả lời, và chỉ ra tin nào", {})]], size=SIZE["body"])


def rag_two_phases(d: Deck) -> None:
    s = d.slide("rag-two-phases", SEC, "Hai pha: lập chỉ mục một lần, truy vấn cho từng câu hỏi", notes=(
        f"Pha lập chỉ mục chạy ngoại tuyến. Nhóm đọc {d.N(RAW)} dòng thô, làm sạch và bỏ trùng còn {d.N(KEPT)} tin, "
        f"cắt thành {d.N(CHUNKS)} chunk, nhúng bằng multilingual-e5-large ra vector {d.N(E5 + '.dim')} chiều, rồi lưu "
        "vào Qdrant cùng vector thưa BM25. Pha này chỉ chạy lại khi dữ liệu đổi. Pha truy vấn chạy cho mỗi câu hỏi: "
        "tìm bằng cả vector dày và BM25 rồi gộp hai danh sách bằng RRF, ghép ngữ cảnh vào prompt, LLM trả lời kèm mã "
        "tin. Mọi phần sau của bài là cải tiến một hộp trong hai hàng này. Chuyển sang slide sau: trước khi xây, "
        "nhìn kỹ dữ liệu."))
    s.text(MX, 1.85, 8, 0.4, "Indexing · ngoại tuyến, chạy một lần", size=SIZE["small"], bold=True, color="blue")
    s.flow([("Dữ liệu thô", f"{d.N(RAW)} dòng"),
            ("Làm sạch", f"còn {d.N(KEPT)} tin"),
            ("Chunk", f"{d.N(CHUNKS)} chunk"),
            ("Embedding", f"e5-large, {d.N(E5 + '.dim')} chiều"),
            ("Lưu", "Qdrant: dense + BM25")], y=2.3, h=1.35)
    s.text(MX, 4.25, 8, 0.4, "Querying · chạy cho mỗi câu hỏi", size=SIZE["small"], bold=True, color="teal")
    s.flow([("Câu hỏi", "tiếng Việt tự nhiên"),
            ("Truy xuất", "dense + BM25, gộp RRF"),
            ("Ghép prompt", "chỉ thị + ngữ cảnh"),
            ("LLM sinh", "trả lời kèm mã tin")], y=4.7, h=1.35, fills=["teal_bg"] * 4)


def data_overview(d: Deck) -> None:
    s = d.slide("data-overview", "Dữ liệu", f"{d.N(RAW, 'mil1')} triệu dòng thô, còn {d.N(KEPT, 'mil2')} triệu tin "
                "sau khi bỏ dòng trống và tin trùng", notes=(
        f"Dataset có {d.N(RAW)} dòng. Bỏ {d.N('ingest_report.json:dropped.no_text')} dòng không có chữ, rồi bỏ "
        f"{d.N('ingest_report.json:dedup.exact_dup')} dòng trùng chính xác và {d.N('ingest_report.json:dedup.prefix_dup')} "
        f"dòng trùng phần đầu nội dung, tức tin đăng lại gần như nguyên văn. Còn {d.N(KEPT)} tin, nghĩa là "
        f"{d.N(DROP, 'pct1')} phần trăm dòng thô bị loại trước khi tốn giờ GPU để embed. Tin đăng từ tháng "
        f"{d.T('eda_summary.json:period.min_month')} đến {d.T('eda_summary.json:period.max_month')}, phủ "
        f"{d.N('graph_report.json:nodes.Province')} tỉnh thành theo địa giới cũ, nhưng Hà Nội và TP.HCM chiếm "
        f"{d.N('eda_summary.json:hn_hcm_share', 'pct1')} phần trăm. Giá ở đây là giá rao, không phải giá giao dịch. "
        "Chuyển sang slide sau: giá theo loại hình."))
    raw = d.V(RAW) or 1
    steps = [("Dòng thô", RAW, "muted"),
             ("Bỏ dòng không có chữ", "={ingest_report.json:raw_rows}-{ingest_report.json:dropped.no_text}", "blue"),
             ("Bỏ trùng chính xác", "={ingest_report.json:raw_rows}-{ingest_report.json:dropped.no_text}"
                                     "-{ingest_report.json:dedup.exact_dup}", "violet"),
             ("Bỏ trùng tiền tố, giữ lại", KEPT, "teal")]
    full = 6.8
    for i, (lab, key, col) in enumerate(steps):
        y = 1.95 + i * 1.12
        v = d.V(key) or 0
        s.text(MX, y, full, 0.38, lab, size=SIZE["small"], bold=True, color="ink")
        s.rect(MX, y + 0.42, max(full * v / raw, 0.3), 0.5, fill=col, line=None)
        s.text(MX + 0.15, y + 0.42, 4.0, 0.5, d.N(key), size=SIZE["small"], bold=True, color="white", anchor="middle")
    x2, w2 = 7.9, W - MX - 7.9
    kpi(s, x2, 1.95, w2, f"{d.N(DROP, 'pct1')}%", "dòng thô bị loại (trống, trùng chính xác, trùng tiền tố)",
        accent="coral")
    kpi(s, x2, 3.55, w2, f"{d.T('eda_summary.json:period.min_month')} → "
        f"{d.T('eda_summary.json:period.max_month')}", "giai đoạn đăng tin (năm-tháng)", accent="blue", size=30)
    kpi(s, x2, 5.15, w2, f"{d.N('eda_summary.json:hn_hcm_share', 'pct1')}%",
        f"tin ở Hà Nội và TP.HCM, trên {d.N('graph_report.json:nodes.Province')} tỉnh thành (địa giới cũ)",
        accent="teal")


def eda_price(d: Deck) -> None:
    pt = "eda_summary.json:by_property_type"
    house, land = f"{pt}[property_type=Nhà]", f"{pt}[property_type=Đất]"
    apt = f"{pt}[property_type=Căn hộ chung cư]"
    ratio = f"={{{house}.median_price_m2_mil}}/{{{land}.median_price_m2_mil}}"
    s = d.slide("eda-price", "Khám phá dữ liệu", "Nhà đắt nhất theo m², đất rẻ nhất: trung vị chênh "
                f"{d.N(ratio, 'dec1')} lần", notes=(
        f"Cột là trung vị giá rao trên mỗi mét vuông của tin bán, tính bằng triệu đồng. Nhà ở mức "
        f"{d.N(house + '.median_price_m2_mil', 'dec1')} triệu, căn hộ chung cư {d.N(apt + '.median_price_m2_mil', 'dec1')} "
        f"triệu, đất {d.N(land + '.median_price_m2_mil', 'dec1')} triệu. Dùng trung vị chứ không dùng trung bình vì giá "
        "lệch phải rất mạnh: vài tin rao hàng trăm tỷ kéo trung bình lên xa. Nhà chiếm nhiều tin nhất với "
        f"{d.N(house + '.n')} tin. Với trợ lý, câu hỏi giá phải lọc đúng loại hình trước, nếu không "
        "con số trộn nhà với đất sẽ vô nghĩa. Chuyển sang slide sau: tin tập trung ở đâu."))
    s.chart(MX, 1.85, 7.6, 4.55, "column", f"{pt}[*].property_type",
            [("Giá/m² trung vị (triệu đồng)", f"{pt}[*].median_price_m2_mil", "blue")], number_format="0")
    x2, w2 = 8.6, W - MX - 8.6
    s.text(x2, 2.0, w2, 1.6, [[("Nhà  ", {"bold": True, "color": "blue"}),
           (f"{d.N(house + '.median_price_m2_mil', 'dec1')} triệu/m²", {})],
           [("Căn hộ  ", {"bold": True, "color": "blue"}), (f"{d.N(apt + '.median_price_m2_mil', 'dec1')} triệu/m²", {})],
           [("Đất  ", {"bold": True, "color": "blue"}), (f"{d.N(land + '.median_price_m2_mil', 'dec1')} triệu/m²", {})]],
           size=SIZE["body"], spacing=1.3)
    s.text(x2, 3.9, w2, 2.2, ["Giá lệch phải mạnh nên mô tả bằng trung vị.",
                              "Câu hỏi về giá phải lọc loại hình trước khi tính."], size=SIZE["small"], bullets=True)
    s.caption("Trung vị giá rao/m² của tin bán (không phải giá giao dịch), tính trên toàn bộ tin sau bỏ trùng.")


def eda_geo(d: Deck) -> None:
    tp = "eda_summary.json:top_provinces"
    s = d.slide("eda-geo", "Khám phá dữ liệu", f"Hà Nội và TP.HCM chiếm {d.N('eda_summary.json:hn_hcm_share', 'pct1')}% "
                "số tin: dữ liệu rất lệch về hai đô thị lớn", notes=(
        f"TP.HCM dẫn đầu với {d.N(tp + '[0].n')} tin, Hà Nội {d.N(tp + '[1].n')} tin, rồi tụt hẳn xuống "
        f"{d.T(tp + '[2].province')} với {d.N(tp + '[2].n')} tin. Bên phải là mười quận huyện nhiều tin nhất ở "
        "mỗi thành phố. Hệ quả cho trợ lý: ở hai thành phố lớn câu trả lời có nhiều bằng chứng, còn ở tỉnh nhỏ một quận "
        "có khi chỉ vài chục tin, nên thống kê giá ở đó kém tin cậy và hệ thống nên nói rõ số tin làm căn cứ. "
        "Tên tỉnh theo địa giới trước đợt sáp nhập năm 2025. Chuyển sang slide sau: số tin theo thời gian."))
    gf = s.chart(MX, 1.85, 5.9, 4.6, "bar", f"{tp}[*].province", [("Nghìn tin", f"{tp}[*].n", "teal", 0.001)],
                 number_format="0")
    gf.chart.category_axis.reverse_order = True
    img = d.res.results_dir / "eda" / "top_districts_hcm_hn.png"
    s.image(img, 6.85, 1.9, w=W - MX - 6.85)
    s.text(6.85, 5.4, W - MX - 6.85, 0.8, "Mười quận/huyện nhiều tin nhất ở Hà Nội và TP.HCM",
           size=SIZE["small"], color="text")
    s.caption("Trái: số tin (nghìn) của các tỉnh/thành đứng đầu, địa giới cũ. Phải: hình xuất từ notebook EDA.")


def eda_trend(d: Deck) -> None:
    mo = "eda_summary.json:monthly"
    months = d.V(mo) or []
    hi = max(months, key=lambda r: r["n"])["month"] if months else "?"
    lo = min(months, key=lambda r: r["n"])["month"] if months else "?"
    p_hi = max(months, key=lambda r: r["median_price_m2_mil"])["month"] if months else "?"
    p_lo = min(months, key=lambda r: r["median_price_m2_mil"])["month"] if months else "?"
    s = d.slide("eda-trend", "Khám phá dữ liệu", "Số tin dao động mạnh theo tháng, còn giá/m² trung vị gần như "
                "đi ngang", notes=(
        f"Bên trái là số tin mỗi tháng. Tháng {d.T(f'{mo}[month={hi}].month')} nhiều nhất với "
        f"{d.N(f'{mo}[month={hi}].n')} tin, tháng {d.T(f'{mo}[month={lo}].month')} ít nhất với "
        f"{d.N(f'{mo}[month={lo}].n')} tin; tháng ít tin rơi vào dịp Tết, nhưng nhóm chưa kiểm chứng nguyên nhân. "
        f"Bên phải là trung vị giá trên mét vuông: chỉ chạy từ {d.N(f'{mo}[month={p_lo}].median_price_m2_mil', 'dec1')} "
        f"đến {d.N(f'{mo}[month={p_hi}].median_price_m2_mil', 'dec1')} triệu. Mười tháng là quá ngắn để nói về xu hướng "
        "thị trường, và cơ cấu loại hình mỗi tháng cũng khác nhau. Với trợ lý, điểm đáng giữ là mỗi tin có tháng đăng, "
        "nên trả lời được câu hỏi theo thời gian. Chuyển sang slide sau: chất lượng dữ liệu."))
    s.chart(MX, 1.85, 6.0, 4.4, "column", f"{mo}[*].month", [("Nghìn tin", f"{mo}[*].n", "blue", 0.001)],
            number_format="0")
    s.chart(6.95, 1.85, W - MX - 6.95, 4.4, "line", f"{mo}[*].month",
            [("Giá/m² trung vị (triệu đồng)", f"{mo}[*].median_price_m2_mil", "amber")], number_format="0")
    s.text(MX, 6.2, 6.0, 0.35, "Số tin mỗi tháng (nghìn)", size=SIZE["caption"], bold=True, color="blue", align="center")
    s.text(6.95, 6.2, W - MX - 6.95, 0.35, "Giá/m² trung vị của tin bán (triệu đồng)", size=SIZE["caption"],
           bold=True, color="amber", align="center")


def eda_quality(d: Deck) -> None:
    mp = "ingest_report.json:missing_pct"
    s = d.slide("eda-quality", "Khám phá dữ liệu", "Cột có cấu trúc thiếu nhiều, giá có outlier: không thể chỉ lọc "
                "theo cột", notes=(
        f"Bảng trái là tỷ lệ ô trống. {d.N(mp + '.project', 'dec1')} phần trăm tin không ghi dự án, "
        f"{d.N(mp + '.bedrooms', 'dec1')} phần trăm không ghi số phòng ngủ, {d.N(mp + '.ward', 'dec1')} phần trăm không có "
        "phường. Nếu lọc cứng theo số phòng ngủ thì gần một nửa số tin biến mất khỏi kết quả, dù mô tả có thể ghi rõ. "
        f"Bên phải: {d.N('ingest_report.json:price.outliers')} tin bị gắn cờ outlier giá và "
        f"{d.N('ingest_report.json:area_outliers')} tin outlier diện tích, được loại khỏi thống kê giá; "
        f"{d.N('ingest_report.json:price.fixed_thousand')} tin ghi giá sai đơn vị đã được sửa. Chuyển sang slide sau: "
        "vì sao điều này dẫn tới hai loại tri thức."))
    rows = [["Cột", "Thiếu (%)"],
            ["Dự án", d.N(mp + ".project", "dec1")],
            ["Số phòng ngủ", d.N(mp + ".bedrooms", "dec1")],
            ["Tên đường", d.N(mp + ".street", "dec1")],
            ["Phường/xã", d.N(mp + ".ward", "dec1")],
            ["Giá", d.N("eda_summary.json:missing_pct.price_vnd", "dec1")]]
    s.text(MX, 1.85, 5.2, 0.4, "Tỷ lệ ô trống sau làm sạch", size=SIZE["small"], bold=True, color="ink")
    grid(s, MX, 2.35, 5.2, rows, col_w=[3.2, 2.0], row_h=0.55)
    x2 = 6.4
    w2 = (W - MX - x2 - 0.3) / 2
    kpi(s, x2, 1.85, w2, d.N("ingest_report.json:price.outliers"), "tin gắn cờ outlier giá/m²", accent="coral")
    kpi(s, x2 + w2 + 0.3, 1.85, w2, d.N("ingest_report.json:area_outliers"), "tin gắn cờ outlier diện tích",
        accent="coral")
    kpi(s, x2, 3.6, w2, d.N("ingest_report.json:price.null_or_unparsable"), "tin không có giá hoặc giá không đọc được",
        accent="amber")
    kpi(s, x2 + w2 + 0.3, 3.6, w2, d.N("ingest_report.json:price.fixed_thousand"), "tin ghi giá sai đơn vị, đã sửa",
        accent="teal")
    s.text(x2, 5.35, W - MX - x2, 0.9, "Outlier chỉ bị gắn cờ: tin vẫn nằm trong chỉ mục để tìm theo nội dung, "
           "nhưng không vào thống kê giá.", size=SIZE["small"])
    s.caption("Nguồn: báo cáo ingest và EDA trên toàn bộ tin sau bỏ trùng.")


def rag_two_kinds(d: Deck) -> None:
    s = d.slide("rag-two-kinds", SEC, "Mỗi tin đăng mang hai loại tri thức, nên cần cả RAG lẫn đồ thị", notes=(
        "Phần phi cấu trúc là tiêu đề và mô tả viết tự do, nơi có những ý "
        "như thang máy, ô tô đỗ cửa, có thương lượng; muốn tìm chúng phải tìm theo nghĩa, đó là việc của embedding và "
        "RAG. Phần có cấu trúc là các cột loại hình, phường, giá, diện tích; câu hỏi như quận nào có giá trên mét vuông "
        "thấp nhất phải đếm và tính trên cột, không đọc ra từ vài đoạn văn. Hai nguồn bù cho nhau: cột thiếu thì mô tả "
        f"vẫn còn, ví dụ {d.N('ingest_report.json:missing_pct.bedrooms', 'dec1')} phần trăm tin không ghi số phòng ngủ ở "
        "cột. Vì vậy nhóm làm cả hai và dùng router để chọn. Chuyển sang slide sau: cắt văn bản thành chunk."))
    bw = (W - 2 * MX - 0.5) / 2
    cols = [("Văn bản tự do", "tiêu đề, mô tả: “thang máy, ô tô đỗ cửa, có thương lượng”",
             "tìm theo nghĩa", "Embedding + RAG", "blue"),
            ("Cột có cấu trúc", "loại hình, phường, giá, diện tích, số phòng ngủ",
             "lọc, đếm, tính, đi theo quan hệ", "Knowledge Graph", "amber")]
    for i, (name, what, how, tool, col) in enumerate(cols):
        x = MX + i * (bw + 0.5)
        s.rect(x, 1.95, bw, 3.5, fill=f"{col}_bg", line=None, radius=0.05)
        s.text(x + 0.35, 2.15, bw - 0.7, 0.55, name, size=26, bold=True, color=col)
        s.text(x + 0.35, 2.85, bw - 0.7, 1.15, what, size=SIZE["body"])
        s.text(x + 0.35, 4.05, bw - 0.7, 0.75, [[("Cách truy xuất  ", {"bold": True, "color": "ink"}), (how, {})]],
               size=SIZE["body"])
        s.text(x + 0.35, 4.85, bw - 0.7, 0.5, tool, size=22, bold=True, color="ink")
    s.text(MX, 5.8, W - 2 * MX, 0.6, [[("Ý chính  ", {"bold": True, "color": "teal"}),
           ("câu hỏi thật thường cần cả hai nguồn, router chọn đường đi", {})]], size=SIZE["body"])


def rag_chunking(d: Deck) -> None:
    tk = "eda_summary.json:tokens_per_listing"
    per = f"={{{CHUNKS}}}/{{{KEPT}}}"
    s = d.slide("rag-chunking", SEC, "Chunking: header giá và địa chỉ lặp ở mọi chunk, tin dài mới bị cắt cửa sổ",
                notes=(
        "Chunk quá lớn thì nhiễu và vượt giới hạn mô hình, quá nhỏ thì mất ngữ cảnh. Nhóm cắt theo bản ghi: mỗi tin "
        f"một chunk, tin dài thì trượt cửa sổ {d.A('400 token, chồng 100 token', CFG)}. Header gồm tiêu đề, loại hình, "
        "địa chỉ, giá, diện tích được lặp ở mọi chunk, nên chunk thứ hai của một tin vẫn mang giá và địa chỉ để lọc và "
        "trích dẫn. Vì header chiếm chỗ, cỡ cửa sổ cho phần mô tả là S trừ độ dài header. Tin trung vị chỉ có "
        f"{d.N(tk + '.p50', 'int')} token nên phần lớn là một chunk; trung bình mỗi tin ra {d.N(per, 'dec2')} chunk. "
        f"Công thức cho số chunk đúng bằng đếm thực của code, và không tin nào bị cắt cụt "
        f"({d.N('night_report.json:counters.chunks_truncated_listings')} tin). Chuyển sang slide sau: embedding."))
    s.rect(MX, 1.95, 6.3, 1.75, fill="soft", line=None, radius=0.06)
    s.text(MX + 0.3, 2.05, 5.8, 0.4, "Số chunk của một văn bản dài L token", size=SIZE["small"], bold=True,
           color="teal")
    s.text(MX + 0.3, 2.5, 5.9, 0.6, "n = ⌈(L − O) / (S − O)⌉ khi L > S", size=24, bold=True, color="ink")
    s.text(MX + 0.3, 3.15, 5.8, 0.45, d.A("n = 1 khi L ≤ S", "công thức") + " · S: cỡ chunk · O: chồng lấn", size=SIZE["small"])
    s.text(MX, 3.85, 6.3, 0.45, d.A("Ví dụ: L = 2000, S = 400, O = 100 → n = ⌈1900 / 300⌉ = 7", "ví dụ minh hoạ công thức"),
           size=SIZE["small"], color="text")
    rows = [["Tham số", "Giá trị"],
            ["Cỡ chunk S", d.A("400 token", CFG)],
            ["Chồng lấn O", d.A("100 token", CFG)],
            ["Header lặp ở mọi chunk", d.A("tối đa 96 token", CFG)]]
    grid(s, MX, 4.45, 6.3, rows, col_w=[3.8, 2.5], row_h=0.48)
    x2, w2 = 7.4, W - MX - 7.4
    kpi(s, x2, 1.95, w2, f"{d.N(tk + '.p50', 'int')} · {d.N(tk + '.p90', 'int')} · {d.N(tk + '.p99', 'int')}",
        "token mỗi tin ở p50 · p90 · p99 (tokenizer của e5-large)", accent="blue", size=32)
    kpi(s, x2, 3.6, w2, d.N(per, "dec2"), f"chunk trung bình mỗi tin: {d.N(CHUNKS)} chunk từ {d.N(KEPT)} tin",
        accent="teal")
    s.text(x2, 5.25, w2, 0.9, "Tin dài ở đuôi phân phối vượt một chunk; cửa sổ giữ mọi chunk trong giới hạn "
           "đọc của e5-large nên không mất chữ.", size=SIZE["small"])


def emb_theory(d: Deck) -> None:
    s = d.slide("emb-theory", SEC, "Embedding đưa văn bản vào không gian vector: gần nghĩa thì gần nhau", notes=(
        "Embedding biến một đoạn văn thành vector; hai câu cùng nghĩa cho vector gần nhau dù khác chữ, ví dụ hai phòng "
        "ngủ gần trường và căn hộ hai phòng ngủ cạnh trường. Độ gần đo bằng cosine. Họ mô hình E5 được huấn luyện với "
        "tiền tố: chunk lưu vào kho mang tiền tố passage, câu hỏi mang tiền tố query, và phải dùng cùng một mô hình cho "
        "hai pha. Với hàng triệu vector, Qdrant không so từng vector mà đi trên đồ thị HNSW "
        f"({d.A('m = 16, ef_construct = 100', 'tham số HNSW trong src/vre/index/qdrant_store.py')}). Bản nén int8 nằm "
        "trong RAM để duyệt nhanh, vector gốc nằm trên đĩa để chấm lại các ứng viên tốt nhất. Đổi một chút độ chính xác "
        "lấy tốc độ. Chuyển sang slide sau: chọn mô hình nào."))
    items = [("Vector hoá", "câu cùng nghĩa cho vector gần nhau, đo bằng cosine", "blue"),
             ("Tiền tố", "“passage: ” khi lưu, “query: ” khi hỏi; cùng một mô hình cho hai pha", "teal"),
             ("Tìm gần đúng", "HNSW trên vector int8 trong RAM, chấm lại bằng vector gốc", "violet")]
    bw = (W - 2 * MX - 2 * 0.4) / 3
    for i, (name, desc, col) in enumerate(items):
        x = MX + i * (bw + 0.4)
        s.rect(x, 2.0, bw, 0.1, fill=col, line=None)
        s.text(x, 2.3, bw, 0.6, name, size=26, bold=True, color=col)
        s.text(x, 3.05, bw, 2.2, desc, size=SIZE["body"])
    s.text(MX, 5.6, W - 2 * MX, 0.6, [[("Ý chính  ", {"bold": True, "color": "teal"}),
           ("tìm theo nghĩa thay vì khớp từ, đủ nhanh cho hàng triệu chunk", {})]], size=SIZE["body"])


def emb_choice(d: Deck) -> None:
    eb = "embedding_benchmark.json"
    base = "embedding_benchmark.json:[model=intfloat/multilingual-e5-base]"
    bge = "embedding_benchmark.json:[model=BAAI/bge-m3]"
    s = d.slide("emb-choice", SEC, "Chọn multilingual-e5-large: truy xuất tốt nhất trong bốn model, đổi lại chậm hơn",
                notes=(
        f"Nhóm đo bốn model trên cùng {d.N(E5 + '.corpus')} chunk và {d.N(E5 + '.queries')} truy vấn, cùng GPU. "
        f"e5-large đứng đầu cả ba chỉ số: Recall@10 {d.N(E5 + '.recall_at_10', 'pct1')} phần trăm so với "
        f"{d.N(bge + '.recall_at_10', 'pct1')} của bge-m3, MRR@10 {d.N(E5 + '.mrr_at_10', 'dec3')}. Cái giá là tốc độ: "
        f"{d.N(E5 + '.docs_per_s', 'dec1')} văn bản mỗi giây, chậm hơn e5-base khoảng "
        f"{d.N(f'={{{base}.docs_per_s}}/{{{E5}.docs_per_s}}', 'dec1')} lần. Vì chỉ mục chỉ dựng một lần còn truy vấn "
        "chạy mãi, nhóm chọn chất lượng. Lưu ý giới hạn: bộ đo là mẫu nhỏ, truy vấn sinh tự động, nên khoảng cách vài "
        "điểm giữa các model chỉ mang tính tham khảo. Chuyển sang slide sau: thực hành 1."))
    s.chart(MX, 1.85, 7.5, 4.45, "column", f"{eb}:[*].model",
            [("Recall@1", f"{eb}:[*].recall_at_1", "blue", 100), ("Recall@10", f"{eb}:[*].recall_at_10", "teal", 100),
             ("MRR@10", f"{eb}:[*].mrr_at_10", "violet", 100)], number_format="0.0")
    x2, w2 = 8.4, W - MX - 8.4
    rows = [["Model", "docs/s", "Chiều"]]
    hl = None
    for i, m in enumerate(d.V(f"{eb}:[*].model") or []):
        rows.append([m.split("/")[-1], d.N(f"{eb}:[{i}].docs_per_s", "dec1"), d.N(f"{eb}:[{i}].dim")])
        if m == "intfloat/multilingual-e5-large":
            hl = i + 1
    s.text(x2, 1.85, w2, 0.4, "Tốc độ đo trên cùng GPU", size=SIZE["small"], bold=True, color="ink")
    grid(s, x2, 2.3, w2, rows, col_w=[2.45, 0.95, 0.93], size=SIZE["caption"], row_h=0.45, highlight_row=hl)
    s.text(x2, 4.75, w2, 1.4, [[("Chốt  ", {"bold": True, "color": "teal"}),
           ("chỉ mục dựng một lần, truy vấn chạy mãi: ưu tiên chất lượng.", {})]], size=SIZE["small"])
    s.caption(f"{d.A('Chỉ số × 100', 'đơn vị hiển thị chart')}. Bộ đo: {d.N(E5 + '.corpus')} chunk, {d.N(E5 + '.queries')} truy vấn sinh tự động; "
              "hàng tô màu là model được chốt.")


def rag_practice_1(d: Deck) -> None:
    nr = "night_report.json"
    s = d.slide("rag-practice-1", SEC, f"Thực hành 1: {d.N(KEPT)} tin thành {d.N(nr + ':qdrant.points')} point "
                "trong Qdrant", notes=(
        f"Pipeline gồm năm bước. Ingest bằng DuckDB đọc {d.N(RAW)} dòng, làm sạch, ẩn số điện thoại, bỏ trùng còn "
        f"{d.N(KEPT)} tin. Chunk theo quy tắc vừa nói, tính vector thưa BM25, rồi embed bằng e5-large trên GPU với "
        f"thông lượng trung bình {d.N(nr + ':embed.chunks_per_s_avg', 'dec1')} chunk mỗi giây. Kết quả chia thành "
        f"{d.N(nr + ':counters.parts_total')} part và nạp vào Qdrant. Kiểm tra nhất quán: số point trong Qdrant bằng "
        f"đúng số chunk đã cắt, {d.N(nr + ':counters.points_in_qdrant')}, collection ở trạng thái "
        f"{d.T(nr + ':qdrant.status')}, và ba truy vấn thử đều trả về kết quả. Chuyển sang slide sau: chạy ở quy mô "
        "này đòi hỏi những gì."))
    s.flow([("Ingest", f"DuckDB: {d.N(RAW)} dòng → {d.N(KEPT)} tin"),
            ("Chunk", "header lặp + cửa sổ token"),
            ("BM25 + embed", f"e5-large trên GPU, {d.N(nr + ':embed.chunks_per_s_avg', 'dec1')} chunk/s"),
            ("Upload", f"{d.N(nr + ':counters.parts_uploaded')} part vào Qdrant"),
            ("HNSW + kiểm", "chỉ mục ANN, truy vấn thử")], y=1.95, h=1.55)
    bw = (W - 2 * MX - 3 * 0.3) / 4
    vals = [(d.N(nr + ":qdrant.points"), "point trong Qdrant, khớp số chunk đã cắt", "teal"),
            (d.N(nr + ":qdrant.indexed_vectors"), "vector đã lập chỉ mục (dense và BM25)", "blue"),
            (d.T(nr + ":qdrant.status"), "trạng thái collection sau khi dựng HNSW", "teal"),
            (f"{d.N(nr + ':counters.parts_uploaded')}/{d.N(nr + ':counters.parts_total')}",
             "part đã nạp, mỗi part một checkpoint", "violet")]
    for i, (v, lab, col) in enumerate(vals):
        kpi(s, MX + i * (bw + 0.3), 4.0, bw, v, lab, accent=col, size=32, h=1.6)
    s.caption(f"Collection {d.T(nr + ':collection')}, model {d.T(E5 + '.model')}, cosine, int8 trong RAM.", y=6.0)


def scale_3_5m(d: Deck) -> None:
    nr = "night_report.json"
    st = f"{nr}:steps"
    embed_est = f"={{{CHUNKS}}}/{{{nr}:embed.chunks_per_s_avg}}"
    s = d.slide("scale-3-5m", SEC, f"Scale lên {d.N(RAW, 'mil1')} triệu tin: dựng chỉ mục trong "
                f"{d.N(nr + ':wall_hours', 'dec1')} giờ trên một laptop", notes=(
        f"Toàn bộ lượt dựng chỉ mục kéo dài {d.N(nr + ':wall_hours', 'dec1')} giờ từ lúc bắt đầu đến lúc xong, trên "
        f"{d.A('RTX 4060 Laptop 8GB', 'phần cứng')}. Phần nặng nhất là embed: với thông lượng trung bình "
        f"{d.N(nr + ':embed.chunks_per_s_avg', 'dec1')} chunk mỗi giây thì riêng bước này cần khoảng "
        f"{d.N(embed_est, 'h1')} giờ; đây là ước tính, không phải số đo trực tiếp. Lượt chạy bị dừng tay một lần vì "
        "thiếu bộ nhớ commit, rồi resume từ checkpoint theo part nên không phải làm lại. Vì lượt cuối là lượt resume, "
        "log của nó không có thời gian thật cho chunk, BM25 và embed, và đỉnh RAM, VRAM không được ghi lại; nhóm không "
        "báo các số đó. Bỏ trùng trước khi embed là tiết kiệm lớn nhất. Chuyển sang slide sau: kiến trúc toàn hệ."))
    rows = [["Bước", "Thời gian"],
            ["Ingest + bỏ trùng (DuckDB)", f"{d.N('ingest_report.json:seconds', 'dec1')} giây"],
            ["Embed, ước từ thông lượng", f"≈ {d.N(embed_est, 'h1')} giờ"],
            ["Upload vào Qdrant", f"{d.N(st + '[name=upload].seconds', 'min0')} phút"],
            ["Dựng HNSW", f"{d.N(st + '[name=hnsw].seconds', 'min0')} phút"],
            ["PCA cho mô-đun định giá", f"{d.N(st + '[name=pca].seconds', 'min0')} phút"],
            ["Import đồ thị (neo4j-admin)", f"{d.N('graph_report.json:import_seconds', 'dec1')} giây"],
            ["Cả lượt, đầu đến cuối", f"{d.N(nr + ':wall_hours', 'dec1')} giờ"]]
    grid(s, MX, 1.9, 6.3, rows, col_w=[4.2, 2.1], row_h=0.46, highlight_row=7)
    x2, w2 = 7.3, W - MX - 7.3
    s.text(x2, 1.85, w2, 0.4, "Kỹ thuật giúp chạy được", size=SIZE["small"], bold=True, color="ink")
    s.text(x2, 2.3, w2, 3.4, [
        f"Bỏ trùng trước khi embed: bớt {d.N(DROP, 'pct1')}% dòng",
        f"Checkpoint theo {d.N(nr + ':counters.parts_total')} part; đã resume {d.N(nr + ':resumed_runs')} lần",
        "Dry-run hai part trước lượt dài để bắt lỗi code",
        "Hai phiên song song: đồ thị và chỉ mục vector",
        "neo4j-admin import: nạp đồ thị ngoại tuyến",
        "HNSW dựng song song khi upload; vector int8 trong RAM"], size=SIZE["small"], bullets=True)
    s.text(MX, 5.8, W - 2 * MX, 0.7, "Lượt cuối là lượt resume: log không có thời gian thật cho chunk, BM25, embed, "
           "và không ghi đỉnh RAM/VRAM, nên bảng không nêu các số đó.", size=SIZE["caption"], color="muted")
    s.caption(f"Phần cứng: {d.A('RTX 4060 Laptop 8GB', 'phần cứng')}.", y=6.45)


def arch_overview(d: Deck) -> None:
    s = d.slide("arch-overview", SEC, "Kiến trúc end-to-end: nạp ngoại tuyến vào hai kho, hỏi trực tuyến qua router",
                notes=(
        "Hàng trên là pha nạp. File Parquet của dataset đi qua DuckDB để làm sạch và bỏ trùng. Từ bảng sạch có hai "
        "nhánh: nhánh văn bản được cắt chunk và nhúng bằng e5-large rồi vào Qdrant; nhánh cột được nạp vào Neo4j bằng "
        f"neo4j-admin import, thành đồ thị với {d.N('graph_report.json:nodes.Listing')} node tin đăng. Hàng dưới là pha "
        "hỏi. Người dùng gõ câu hỏi trên web, API chuyển xuống router; router quyết định lấy ngữ cảnh từ Qdrant, từ "
        "đồ thị hay cả hai, rồi LLM viết câu trả lời có mã tin và trả ngược về web. Ba chặng sau của bài lần lượt đi "
        "sâu vào truy xuất, đồ thị và router. Chuyển sang chặng 2: truy xuất và sinh câu trả lời."))
    gap = 0.45
    cw = (W - 2 * MX - 4 * gap) / 5
    cx = [MX + i * (cw + gap) for i in range(5)]
    y1, y2, y3, bh = 1.95, 3.35, 5.0, 1.0
    box(s, cx[0], y1, cw, y2 + bh - y1, "Parquet", f"{d.N(RAW)} dòng thô, {d.N('ingest_report.json:shards')} shard",
        fill="soft")
    box(s, cx[1], y1, cw, y2 + bh - y1, "DuckDB", f"làm sạch, bỏ trùng → {d.N(KEPT)} tin", fill="soft")
    box(s, cx[2], y1, cw, bh, "e5-large", f"{d.N(E5 + '.dim')} chiều, GPU")
    box(s, cx[3], y1, cw, bh, "Qdrant", f"{d.N('night_report.json:qdrant.points')} chunk · dense + BM25")
    box(s, cx[3], y2, cw, bh, "Neo4j", f"{d.N('graph_report.json:nodes.Listing')} tin + địa danh, dự án",
        fill="amber_bg")
    for x in (cx[0] + cw, cx[1] + cw, cx[2] + cw):
        arrow(s, x + 0.05, y1 + bh / 2 - 0.12, gap - 0.1, 0.24)
    arrow(s, cx[0] + cw + 0.05, y2 + bh / 2 - 0.12, gap - 0.1, 0.24)
    arrow(s, cx[1] + cw + 0.05, y2 + bh / 2 - 0.12, cx[3] - cx[1] - cw - 0.1, 0.24)
    s.text(cx[2], y2 + 0.05, cw, 0.35, "neo4j-admin import", size=SIZE["caption"], color="muted", align="center")
    box(s, cx[0], y3, cw, bh, "Web", "hỏi, so sánh bốn hệ", fill="teal_bg")
    box(s, cx[1], y3, cw, bh, "API", "nhận câu hỏi, trả JSON", fill="teal_bg")
    box(s, cx[3], y3, cw, bh, "Router", "chọn RAG, đồ thị hay cả hai", fill="teal_bg")
    box(s, cx[4], y3, cw, bh, "LLM", "trả lời kèm mã tin", fill="teal_bg")
    arrow(s, cx[0] + cw + 0.05, y3 + bh / 2 - 0.12, gap - 0.1, 0.24)
    arrow(s, cx[1] + cw + 0.05, y3 + bh / 2 - 0.12, cx[3] - cx[1] - cw - 0.1, 0.24)
    arrow(s, cx[3] + cw + 0.05, y3 + bh / 2 - 0.12, gap - 0.1, 0.24)
    arrow(s, cx[3] + cw / 2 - 0.15, y2 + bh + 0.06, 0.3, y3 - y2 - bh - 0.12, kind="up")
    s.text(cx[4], y1 + 0.1, cw, 0.9, ["Pha nạp", "ngoại tuyến, chạy một lần"], size=SIZE["small"], color="blue",
           align="center")
    s.text(cx[2], y3 + 0.05, cw, 0.35, "pha hỏi, mỗi câu", size=SIZE["caption"], color="teal", align="center")
    s.caption("Câu trả lời đi ngược LLM → API → Web. Router đọc cả hai kho trước khi gọi LLM.", y=6.2)
