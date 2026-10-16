"""Phần cuối: mô-đun định giá, demo webapp, hạn chế, điều cần nhớ, hỏi đáp và tài liệu tham khảo."""
from kit import MX, ROOT, SIZE, W, Deck

SEC_X = "Mở rộng"
SEC_E = "Kết luận"
VM = "valuation_metrics.json"
ES = "eval_summary.json"
VARIANTS = {"baseline_ward_median": ("Trung vị phường", "muted"),
            "tabular": ("XGBoost bảng số", "blue"),
            "full": ("XGBoost + embedding", "teal")}
FEATURE = {"property_type": "Loại hình", "te_district_key": "Quận/huyện", "te_ward_key": "Phường/xã",
           "te_province": "Tỉnh/thành", "area_m2": "Diện tích", "te_street_key": "Đường", "te_project_key": "Dự án",
           "am_mat_tien": "Mặt tiền", "month_idx": "Tháng đăng", "bedrooms": "Phòng ngủ", "bathrooms": "Phòng tắm",
           "house_direction": "Hướng nhà", "am_thang_may": "Thang máy", "frontage_m": "Chiều ngang",
           "floors": "Số tầng", "road_width_m": "Độ rộng đường", "am_hem_xe_hoi": "Hẻm xe hơi",
           "am_so_hong": "Sổ hồng", "am_gan_truong": "Gần trường", "am_ho_boi": "Hồ bơi", "am_noi_that": "Nội thất",
           "am_view_song": "View sông", "am_gara": "Gara", "am_an_ninh": "An ninh"}
SHAP_TOP = 10
DEMO = [("compare", "So sánh bốn hệ trên một câu hỏi"), ("map", "Bản đồ giá theo khu vực"),
        ("valuation", "Định giá một tin đăng"), ("cypher", "Cypher và nguồn trích dẫn")]


def build(d: Deck) -> None:
    valuation(d)
    valuation_detail(d)
    demo(d)
    limits(d)
    takeaways(d)
    qa(d)


def variants(d: Deck) -> list[str]:
    present = d.V(f"{VM}:variants") or {}
    return [v for v in VARIANTS if v in present] or list(VARIANTS)


def best_variant(d: Deck) -> str | None:
    mape = {v: d.V(f"{VM}:variants.{v}.mape") for v in variants(d)}
    known = [v for v in mape if mape[v] is not None]
    return min(known, key=lambda v: mape[v]) if known else None


def feature_labels(names: list[str]) -> list[str]:
    """Nhãn đọc được cho đặc trưng; thành phần PCA đánh chữ cái theo thứ tự xuất hiện (nhãn chart không chứa số)."""
    out, letter = [], 0
    for n in names:
        if n.startswith("pca_"):
            out.append(f"Embedding mô tả {chr(ord('A') + letter)}")
            letter += 1
        else:
            out.append(FEATURE.get(n, n if not any(ch.isdigit() for ch in n) else "Đặc trưng khác"))
    return out


# ---------------------------------------------------------------- định giá
def valuation(d: Deck) -> None:
    vs, best = variants(d), best_variant(d)
    sp = f"{VM}:split"
    pca = d.A("PCA 32 chiều", "hằng số vre.config pca_dim")
    emb = d.A("e5-large", "tên model embedding")
    if best == "full" and "tabular" in vs:
        title = (f"Thêm embedding mô tả cho sai số định giá thấp nhất: MAPE {d.N(f'{VM}:variants.full.mape', 'pct1')}% "
                 f"so với {d.N(f'{VM}:variants.tabular.mape', 'pct1')}% khi chỉ dùng bảng số")
    elif best:
        title = (f"{VARIANTS[best][0]} cho sai số định giá thấp nhất: "
                 f"MAPE {d.N(f'{VM}:variants.{best}.mape', 'pct1')}%")
    else:
        title = "Mô-đun định giá XGBoost: ba biến thể trên cùng tập test"
    s = d.slide("ext-valuation", SEC_X, title, notes=(
        "Mô-đun định giá ước lượng giá trên mét vuông của một tin bán. Mốc so sánh đơn giản nhất là lấy trung vị giá của "
        "phường. Biến thể bảng số dùng XGBoost trên loại hình, diện tích, số phòng, tiện ích trích từ mô tả, và mã hoá mục "
        "tiêu theo tỉnh, quận, phường, đường, dự án. Biến thể đầy đủ thêm embedding mô tả của " + emb + ", nén "
        + pca + ". Dữ liệu chia theo thời gian: học trên các tháng cũ, chấm trên các tháng mới nhất, nên không rò tương "
        f"lai vào quá khứ. Trên {d.N(f'{VM}:variants.{vs[-1]}.n_test')} tin test, MAE của ba biến thể lần lượt "
        + ", ".join(f"{d.N(f'{VM}:variants.{v}.mae_mil_m2', 'dec1')}" for v in vs)
        + " triệu mỗi mét vuông. Đây là giá rao, không phải giá giao dịch. Slide sau: mô hình dựa vào đặc trưng nào."))
    s.chart(MX, 1.9, 5.6, 4.35, "column", [VARIANTS[v][0] for v in vs],
            [("MAE (triệu/m²)", [f"{VM}:variants.{v}.mae_mil_m2" for v in vs], "teal")], number_format="0.0")
    rows = [["Biến thể", "MAE", "MAPE", "R²"]]
    for v in vs:
        k = f"{VM}:variants.{v}"
        rows.append([VARIANTS[v][0], d.N(k + ".mae_mil_m2", "dec1"), d.N(k + ".mape", "pct1") + "%",
                     d.N(k + ".r2", "dec2")])
    s.table(6.6, 1.95, W - MX - 6.6, rows, col_w=[2.75, 1.1, 1.15, 1.13], size=SIZE["small"], row_h=0.55,
            highlight_row=1 + vs.index(best) if best else None)
    s.text(6.6, 4.35, W - MX - 6.6, 0.4, "Chia theo thời gian", size=SIZE["caption"], bold=True, color="teal")
    # một dòng một shape: mẫu số của checker không khớp qua ranh giới đoạn văn
    for j, (lab, part, n) in enumerate([("Học  ", "train_months", "n_train"), ("Chấm  ", "test_months", "n_test")]):
        s.text(6.6, 4.75 + j * 0.5, W - MX - 6.6, 0.45, [[
            (lab, {"bold": True}),
            (f"{d.T(f'{sp}.{part}[0]')} → {d.T(f'{sp}.{part}[-1]')}"
             f" · {d.N(f'{VM}:variants.{vs[-1]}.{n}')} tin", {})]], size=18, bullets=True)
    s.caption("MAE tính bằng triệu/m² trên tập test; mô hình học log giá/m²; biến thể đầy đủ = bảng số + embedding "
              + emb + " nén " + pca + ".", y=6.3)


def valuation_detail(d: Deck) -> None:
    best = best_variant(d) or "full"
    k = f"{VM}:variants.{best}"
    shap = (d.V(k + ".shap_top") or [])[:SHAP_TOP]
    names = [r["feature"] for r in shap]
    labels = feature_labels(names)
    n_pca = sum(1 for n in names if n.startswith("pca_"))
    title = ("Khu vực và loại hình quyết định giá; embedding mô tả góp phần còn lại" if n_pca
             else "Khu vực và loại hình là các đặc trưng quyết định giá nhiều nhất")
    trained = d.V(f"{VM}:rent_model.trained")
    s = d.slide("ext-valuation-detail", SEC_X, title, notes=(
        "Bên trái là mức đóng góp SHAP trung bình của các đặc trưng mạnh nhất trong biến thể "
        + VARIANTS.get(best, (best,))[0] + ", tính trên thang log giá. Các đặc trưng vị trí dẫn đầu, đúng với trực giác "
        "rằng giá nhà trước hết do vị trí. Các thanh tên embedding mô tả là thành phần PCA của văn bản tin đăng: chúng "
        "mang thông tin mà cột bảng không có, như mô tả mặt tiền hay tình trạng nhà. Bảng bên phải là sai số theo loại "
        "hình; căn hộ dễ đoán nhất vì đồng chất, đất khó nhất. Mô hình cho thuê "
        + ("chưa được huấn luyện vì" if trained is False else "có trong code, với")
        + f" chỉ {d.N(f'{VM}:rent_model.n_rent')} tin thuê sau làm sạch. Slide sau: webapp đặt các hệ cạnh nhau."))
    if shap:
        idx = list(range(len(shap)))[::-1]  # bar ngang vẽ từ dưới lên: đảo để đặc trưng mạnh nhất nằm trên cùng
        s.chart(MX, 1.85, 5.5, 4.3, "bar", [labels[i] for i in idx],
                [("SHAP", [f"{k}.shap_top[{i}].mean_abs_contribution" for i in idx], "teal")],
                number_format="0.00")
    else:
        s.text(MX, 1.9, 5.5, 0.5, "⟨?⟩ thiếu shap_top", size=SIZE["small"])
    types = d.V(k + ".by_property_type") or []
    base = {r["property_type"]: i for i, r in enumerate(d.V(f"{VM}:variants.baseline_ward_median.by_property_type")
                                                       or [])}
    rows = [["Loại hình", "Tin test", "MAE", "MAPE", "MAPE mốc"]]
    for i, r in enumerate(types):
        kk = f"{k}.by_property_type[{i}]"
        bi = base.get(r["property_type"])
        rows.append([d.T(kk + ".property_type"), d.N(kk + ".n"), d.N(kk + ".mae_mil_m2", "dec1"),
                     d.N(kk + ".mape", "pct0") + "%",
                     d.N(f"{VM}:variants.baseline_ward_median.by_property_type[{bi}].mape", "pct0") + "%"
                     if bi is not None else "–"])
    x = 6.4
    s.table(x, 1.9, W - MX - x, rows, col_w=[2.5, 1.15, 0.85, 0.85, 0.983], size=SIZE["small"], row_h=0.44)
    y = 1.9 + 0.44 * len(rows) + 0.25
    s.rect(x, y, W - MX - x, 6.15 - y, fill="amber_bg", line=None, radius=0.06)
    s.text(x + 0.25, y + 0.12, W - MX - x - 0.5, 6.15 - y - 0.2, [
        [("Cho thuê: ", {"bold": True, "color": "amber"}),
         ("chưa huấn luyện" if trained is False else "đã huấn luyện", {"bold": True})],
        f"Chỉ {d.N(f'{VM}:rent_model.n_rent')} tin thuê, dưới ngưỡng tối thiểu "
        + d.A("20.000 tin", "hằng số vre.config rent_min_rows")],
        size=SIZE["small"])
    s.caption("SHAP: trung bình |đóng góp| trên thang log giá/m²; embedding mô tả A, B…: thành phần PCA. "
              "MAE triệu/m². MAPE mốc = trung vị phường.", y=6.3)


# ---------------------------------------------------------------- demo
def demo(d: Deck) -> None:
    s = d.slide("ext-demo", SEC_X, "Webapp đặt các hệ cạnh nhau trên cùng một câu hỏi, kèm bản đồ và định giá",
                notes=(
        "Webapp là nơi gom mọi thứ lại để người dùng thử. Màn hình so sánh gửi một câu hỏi tới cả bốn hệ và hiện câu trả "
        "lời song song, mỗi câu kèm mã tin để bấm vào kiểm chứng. Màn hình bản đồ cho xem giá theo khu vực. Màn hình định "
        "giá nhận thông tin một căn và trả về mức giá ước lượng của mô-đun vừa trình bày. Màn hình cuối hiện câu Cypher mà "
        "hệ Graph đã chạy, để người xem tự đối chiếu. Câu hỏi dùng khi chụp là câu đã chạy trong lượt đánh giá. "
        "Tiếp theo là những gì bài này còn hạn chế."))
    gap, top, bottom = 0.3, 1.85, 6.8
    bw = (W - 2 * MX - gap) / 2
    bh = (bottom - top - gap) / 2
    for i, (key, label) in enumerate(DEMO):
        x, y = MX + (i % 2) * (bw + gap), top + (i // 2) * (bh + gap)
        path = ROOT / "slide" / "assets" / "demo" / f"{key}.png"
        img_h = bh - 0.45
        s.text(x, y + img_h + 0.05, bw, 0.38, label, size=SIZE["caption"], bold=True, color="teal")
        if path.is_file():
            fit_image(s, path, x, y, bw, img_h)
        else:
            s.rect(x, y, bw, img_h, fill="soft", line="line", radius=0.03)
            s.text(x, y, bw, img_h, ["Chờ ảnh chụp màn hình", label], size=SIZE["small"], color="muted",
                   align="center", anchor="middle")


def fit_image(s, path, x, y, w, h) -> None:
    """Đặt ảnh vừa khung, giữ tỉ lệ, căn giữa."""
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    s.image(path, x + (w - pw) / 2, y + (h - ph) / 2, pw, ph)


# ---------------------------------------------------------------- hạn chế, kết luận
def limits(d: Deck) -> None:
    mp = "ingest_report.json:missing_pct"
    n_q, n_b = d.N(ES + ":n_questions"), d.N("benchmark_gt.json:items|len")
    gen = d.T(ES + ":llm.generator")
    planned = d.A("gpt-oss-120b", "tên model sinh dự kiến ban đầu trong plan")
    full = bool(d.V(ES + ":complete")) and d.V(ES + ":n_questions") == d.V("benchmark_gt.json:items|len")
    s = d.slide("end-limits", SEC_E,
                ("Hạn chế: chấm bằng luật, không giám khảo LLM, dữ liệu là giá rao; hướng tiếp theo đã rõ" if full else
                 "Hạn chế: đánh giá mới trên tập con và dữ liệu là giá rao; hướng tiếp theo đã rõ"),
                notes=(
        (f"Về đánh giá, lượt chạy dùng đủ {n_q} câu benchmark với cả bốn hệ, nhưng câu trả lời chấm bằng luật so với "
         "đáp án SQL. Nhóm chủ động không dùng LLM làm giám khảo và không chạy RAGAS để tiết kiệm chi phí, nên chưa đo "
         "faithfulness. Mô hình sinh " if full else
         f"Về đánh giá, benchmark có {n_b} câu nhưng lượt chạy chỉ dùng tập con {n_q} câu vì hạn "
         "mức API miễn phí, chấm bằng luật so với đáp án SQL và chưa đo faithfulness bằng mô hình giám khảo. Mô hình sinh ")
        +
        f"là {gen}, khác {planned} dự kiến ban đầu. Về dữ liệu, giá là giá rao, không phải giá giao dịch; địa giới có cả "
        "tên cũ và tên sau sáp nhập năm 2025; tin không có toạ độ; nhiều cột thiếu, ví dụ đường thiếu "
        f"{d.N(mp + '.street', 'dec0')}% và dự án thiếu {d.N(mp + '.project', 'dec0')}%. Giấy phép dữ liệu chỉ cho "
        "phép dùng phi thương mại. Image Docker của API và web đã dựng và kiểm qua nginx: trang chủ, health, chat "
        "stream và định giá đều trả về thành công. Cột bên phải là việc làm tiếp. Slide sau: những điều cần nhớ."))
    cols = [
        ("Đánh giá", "coral", [
            (f"Đủ {n_q} câu benchmark, bốn hệ" if full else f"Tập con {n_q} câu trên {n_b} câu benchmark"),
            "Chấm bằng luật so với đáp án SQL; không giám khảo LLM, không RAGAS (tiết kiệm chi phí)",
            [("Mô hình sinh: ", {}), (gen, {"bold": True})],
            f"Khác {planned} dự kiến, do hạn mức và rate limit API",
            "Docker api + web đã dựng, kiểm qua nginx"]),
        ("Dữ liệu", "amber", [
            "Giá rao ≠ giá giao dịch",
            "Địa giới cũ lẫn tên sau sáp nhập 2025",
            "Không có toạ độ",
            f"Thiếu cột: phường {d.N(mp + '.ward', 'dec0')}%, đường {d.N(mp + '.street', 'dec0')}%, "
            f"dự án {d.N(mp + '.project', 'dec0')}%",
            d.A("Giấy phép CC-BY-NC 4.0: chỉ phi thương mại", "tên giấy phép dataset")]),
        ("Hướng phát triển", "teal", [
            ("Thêm giám khảo LLM và RAGAS khi có ngân sách" if full else "Chạy đủ benchmark, thêm giám khảo LLM và RAGAS"),
            "Rerank trên GPU để bật mặc định mà không tăng độ trễ",
            "Chuẩn hoá địa giới mới, gắn toạ độ",
            "Định giá cho thuê khi đủ dữ liệu"]),
    ]
    gap = 0.35
    cw = (W - 2 * MX - 2 * gap) / 3
    for i, (head, col, items) in enumerate(cols):
        x = MX + i * (cw + gap)
        s.rect(x, 1.9, cw, 4.85, fill=f"{col}_bg", line=None, radius=0.04)
        s.text(x + 0.3, 2.05, cw - 0.6, 0.45, head.upper(), size=SIZE["caption"], bold=True, color=col)
        s.text(x + 0.3, 2.55, cw - 0.6, 4.1, items, size=SIZE["small"], bullets=True)


def takeaways(d: Deck) -> None:
    s = d.slide("end-takeaways", SEC_E, "Mỗi loại câu hỏi cần đúng công cụ; router ghép chúng, benchmark đo chúng",
                notes=(
        "Bốn điều xin các bạn mang về. Một, RAG giỏi tìm tin theo nghĩa và dẫn được nguồn, nhưng không đếm hay tính được. "
        "Hai, điều kiện cứng như giá, diện tích, quận phải đi vào bộ lọc trước khi tìm, đừng trông vào embedding. Ba, câu "
        "tổng hợp và đa chặng cần đồ thị hoặc truy vấn có cấu trúc: cơ sở dữ liệu tính, mô hình chỉ viết truy vấn và diễn "
        "giải. Bốn, router ghép các nguồn theo từng câu, và muốn biết hệ nào tốt hơn thì phải đo bằng benchmark có đáp án "
        "tính bằng SQL, kèm cỡ mẫu và khoảng tin cậy. Xin cảm ơn, mời các bạn đặt câu hỏi."))
    items = [("RAG để tìm tin", "Tìm theo nghĩa, dẫn nguồn bằng mã tin; không đếm, không tính", "blue"),
             ("Lọc trước cho điều kiện cứng", "Giá, diện tích, khu vực vào bộ lọc trước khi tìm vector", "teal"),
             ("Đồ thị cho câu tổng hợp", "Đếm, trung vị, đa chặng: cơ sở dữ liệu tính, mô hình viết truy vấn", "amber"),
             ("Router ghép, benchmark đo", "Chọn nguồn từng câu; đo bằng đáp án SQL, ghi rõ cỡ mẫu", "violet")]
    gap = 0.35
    cw, ch = (W - 2 * MX - gap) / 2, 2.2
    for i, (t, body, col) in enumerate(items):
        x, y = MX + (i % 2) * (cw + gap), 1.95 + (i // 2) * (ch + gap)
        s.card(x, y, cw, ch, t, body, accent=col, size=SIZE["body"], title_size=24)


def qa(d: Deck) -> None:
    s = d.slide("end-qa", "", "", dark=True, notes=(
        "Phần trình bày đến đây là hết. Không đọc danh sách tài liệu, chỉ để người nghe tra cứu sau. Mã nguồn có notebook "
        "và webapp để chạy lại, dataset công khai trên Hugging Face. Nếu được hỏi về so sánh phương pháp, quay lại slide so "
        "sánh ba kiến trúc; nếu được hỏi về độ tin cậy của số, quay lại slide thiết lập đánh giá. Mời thầy cô và các bạn "
        "đặt câu hỏi."))
    s.text(MX, 0.9, 5.2, 1.1, "Hỏi đáp", size=54, bold=True, color="white")
    s.text(MX, 2.0, 5.2, 0.4, "MÃ NGUỒN", size=SIZE["caption"], bold=True, color="teal_bg")
    s.text(MX, 2.4, 5.2, 0.5, "github.com/ngtukien/Vietnam-Real-Estates", size=SIZE["small"], color="white")
    s.text(MX, 3.1, 5.2, 0.4, "DATASET", size=SIZE["caption"], bold=True, color="teal_bg")
    s.text(MX, 3.5, 5.2, 0.5, "huggingface.co/datasets/tinixai/vietnam-real-estates", size=SIZE["small"],
           color="white")
    s.text(MX, 4.0, 5.2, 0.5, d.A("Giấy phép CC-BY-NC 4.0", "tên giấy phép dataset"), size=SIZE["caption"],
           color="CBD5E1")
    refs = [
        "[1] Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.",
        "[2] Edge, D. et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. "
        + d.A("arXiv:2404.16130", "mã arXiv tài liệu tham khảo") + ".",
        "[3] Traag, V. A., Waltman, L., van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected "
        "communities. Scientific Reports.",
        "[4] Es, S. et al. (2024). RAGAS: Automated Evaluation of Retrieval Augmented Generation. EACL.",
        "[5] Cormack, G. V., Clarke, C. L. A., Büttcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and "
        "individual rank learning methods. SIGIR.",
        "[6] Wang, L. et al. (2022). Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5; bản đa ngôn ngữ "
        + d.A("multilingual-e5-large", "tên model embedding") + ").",
    ]
    s.text(6.3, 0.9, W - MX - 6.3, 0.4, "TÀI LIỆU THAM KHẢO", size=SIZE["caption"], bold=True, color="teal_bg")
    s.text(6.3, 1.35, W - MX - 6.3, 5.4, refs, size=SIZE["caption"], color="E2E8F0", spacing=1.05)
