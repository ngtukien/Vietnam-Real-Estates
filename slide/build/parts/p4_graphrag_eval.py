"""Chặng 4: GraphRAG, router và đánh giá bốn hệ."""
from kit import MX, SIZE, SYSTEM, W, Deck

SEC_G = "Chặng 4 · GraphRAG"
SEC_E = "Chặng 4 · Đánh giá"
ES = "eval_summary.json"
CATS = {"lookup": "Tra cứu", "constraint": "Ràng buộc", "aggregate": "Tổng hợp", "multihop": "Đa chặng",
        "temporal": "Thời gian", "comparison": "So sánh", "project_street": "Dự án, đường", "valuation": "Định giá",
        "unanswerable": "Không trả lời được"}
CAUSE = {"runtime_error": "Lỗi chạy", "router_wrong": "Router chọn sai", "cypher_error": "Cypher lỗi",
         "cypher_wrong_result": "Cypher ra sai", "filter_wrong": "Lọc sai", "retrieval_miss": "Truy xuất trượt",
         "llm_hallucination": "Bịa ngoài nguồn", "refusal_missing": "Không từ chối", "insufficient_context": "Thiếu ngữ cảnh"}


def build(d: Deck) -> None:
    theory(d)
    compare(d)
    patterns(d)
    router(d)
    layers(d)
    setup(d)
    ablation(d)
    results(d)
    pairwise(d)
    cost(d)
    errors(d)


def systems(d: Deck) -> list[str]:
    return d.V(f"{ES}:systems") or list(SYSTEM)


def categories(d: Deck) -> list[str]:
    present = d.V(f"{ES}:by_category") or {}
    return [c for c in CATS if c in present] or list(CATS)


def theory(d: Deck) -> None:
    s = d.slide("gr-theory", SEC_G, "GraphRAG truy xuất đồ thị con hoặc kết quả truy vấn thay cho đoạn văn", notes=(
        "RAG thông thường lấy đoạn văn gần nghĩa nhất làm đơn vị truy xuất. GraphRAG thay đơn vị đó bằng đồ thị con hoặc "
        "kết quả truy vấn trên đồ thị, gồm thực thể, quan hệ và cả phép tính. Hai hàng trên slide chỉ khác nhau ở hai ô giữa. "
        "Về nguồn gốc đồ thị có hai hướng: Microsoft GraphRAG dùng LLM trích thực thể từ văn bản rồi tóm tắt từng cộng đồng; "
        "dữ liệu của nhóm có sẵn cột cấu trúc nên đồ thị dựng bằng luật, không tốn lời gọi LLM và đối chiếu được với SQL. "
        "Đồ thị mạnh ở câu tổng hợp, đa chặng và toàn cục; mô tả mờ như gần trường thì tìm vector vẫn hợp hơn. Slide sau: "
        "so sánh ba kiến trúc."))
    rows = [("RAG", "Tìm top-k đoạn văn gần nghĩa", "Ngữ cảnh: k đoạn văn", "blue"),
            ("GraphRAG", "Duyệt đồ thị: thực thể, quan hệ, phép tính", "Ngữ cảnh: đồ thị con, kết quả truy vấn", "amber")]
    for i, (name, step, ctx, col) in enumerate(rows):
        y = 2.0 + i * 1.75
        s.text(MX, y + 0.35, 1.9, 0.6, name, size=24, bold=True, color=col)
        s.flow([("Câu hỏi", ""), (step, ""), (ctx, ""), ("LLM", "")], y, h=1.3, x0=2.6, x1=W - MX,
               fills=["soft", f"{col}_bg", f"{col}_bg", "soft"], size=18)
    s.text(MX, 5.7, W - 2 * MX, 0.9, [[("Đồ thị vượt trội  ", {"bold": True, "color": "amber"}),
           ("đếm, trung vị, xếp hạng trên toàn tập · đi nhiều chặng · câu hỏi về cả kho dữ liệu", {})]], size=20)


def compare(d: Deck) -> None:
    s = d.slide("gr-compare", SEC_G, "Ba kiến trúc mạnh ở ba loại câu khác nhau; Hybrid để router chọn", notes=(
        "Bảng đặt ba kiến trúc cạnh nhau. Basic RAG lấy chunk gần nghĩa, mạnh với mô tả tự do. Graph dùng Text2Cypher: "
        "mô hình viết Cypher, Neo4j tính chính xác trên toàn bộ dữ liệu, mạnh ở đếm, trung vị, xếp hạng, đa chặng, nhưng cần "
        "schema tốt. Hybrid có router chọn nguồn theo câu hỏi, nên mạnh ở câu vừa có điều kiện cột vừa có yêu cầu trong mô "
        "tả; đổi lại, router chọn sai thì trả lời sai. Trong phần đánh giá còn có LLM thuần, không truy xuất, làm mốc so sánh. "
        "Nói rõ thuật ngữ: nhóm làm Graph-based RAG dựa trên Text2Cypher, không làm community summarization của Microsoft. "
        "Slide sau: các mẫu truy xuất trên đồ thị và lớp an toàn."))
    rows = [["Tiêu chí", "Basic RAG", "Graph (Text2Cypher)", "Hybrid"],
            ["Đơn vị truy xuất", "chunk gần nghĩa", "kết quả Cypher", "cả hai, router chọn"],
            ["Mạnh ở", "mô tả tự do", "đếm, trung vị, đa chặng", "câu lai cột + mô tả"],
            ["Yếu ở", "tổng hợp toàn tập", "mô tả mờ, cần schema", "router chọn sai"],
            ["Chi phí mỗi câu", "một lượt sinh", "viết Cypher + sinh", "router + nguồn + sinh"]]
    s.table(MX, 2.0, W - 2 * MX, rows, col_w=[2.6, 3.0, 3.4, 3.13], size=18, row_h=0.78)


def patterns(d: Deck) -> None:
    ex = d.pick("examples.json:graph", "AG03") or "examples.json:graph[0]"
    s = d.slide("gr-patterns", SEC_G, "Text2Cypher: mô hình viết truy vấn, Neo4j tính; năm lớp an toàn bọc quanh", notes=(
        "Có ba mẫu truy xuất trên đồ thị: local đi từ một thực thể ra láng giềng, global dùng bản tóm tắt cộng đồng, và "
        "Text2Cypher để mô hình viết truy vấn. Câu hỏi bất động sản phần lớn là lọc và tính trên cột nên nhóm chọn Text2Cypher. "
        "Rủi ro là Cypher sai hoặc nguy hiểm, nên có năm lớp: regex chặn lệnh ghi, tự thêm LIMIT, giao dịch chỉ đọc, timeout "
        "và giới hạn số dòng, và thử lại có gửi kèm lỗi. Đoạn code là câu Cypher mô hình sinh thật trong lượt eval. Phép tính "
        "do Neo4j thực hiện, mô hình chỉ viết truy vấn và diễn giải kết quả. Slide sau: router."))
    layers_ = [("Regex chặn", "CREATE, MERGE, DELETE, SET, CALL…"), ("Thêm LIMIT", "nếu RETURN cuối chưa có"),
               ("Chỉ đọc", "Neo4j từ chối mọi lệnh ghi"),
               ("Timeout", d.A("10 s, tối đa 50 dòng", "hằng số vre.config cypher_timeout, cypher_max_rows")),
               ("Thử lại", "gửi lỗi cho mô hình viết lại")]
    for i, (t, body) in enumerate(layers_):
        y = 1.95 + i * 0.92
        s.rect(MX, y, 5.2, 0.8, fill="soft", line=None, radius=0.08)
        s.text(MX + 0.2, y + 0.08, 5.0, 0.65, [[(t + "  ", {"bold": True, "color": "amber"}), (body, {})]],
               size=SIZE["small"], anchor="middle")
    s.text(6.2, 1.95, 6.5, 0.4, "Cypher mô hình sinh · " + d.A("câu đếm", "nhãn"), size=SIZE["caption"],
           bold=True, color="teal")
    s.text(6.2, 2.35, 6.5, 0.8, "“" + d.T(ex + ".question") + "”", size=SIZE["small"], color="ink")
    s.code(6.2, 3.2, 6.5, 3.2, d.T(ex + ".cypher"), highlight=["count(", "IN_PROVINCE", "OF_TYPE", "WHERE", "LIMIT"])


def router(d: Deck) -> None:
    r = f"{ES}:router"
    s = d.slide("gr-router", SEC_G, "Router chọn nguồn cho từng câu; đồ thị rỗng hoặc lỗi thì lui về Basic RAG", notes=(
        "Router là một lượt gọi mô hình với structured output, chọn một trong các nhãn: Basic RAG, Graph, cả hai, hoặc định "
        f"giá. Trên tập con đã chạy, router chọn đúng nhãn ở {d.N(r + '.accuracy', 'pct0')}% số câu; bỏ nhóm không trả lời "
        f"được thì {d.N(r + '.accuracy_excl_unanswerable', 'pct0')}%. Nếu Cypher lỗi, trả về rỗng hoặc Neo4j mất kết nối, "
        "hệ thống lui về Basic RAG. Truy xuất RAG chạy song song với router nên đường lui không phải chờ thêm. "
        "Slide sau: đánh giá được thiết kế thế nào."))
    s.flow([("Câu hỏi", ""), ("Router", "một lượt LLM"), ("Nguồn", "RAG · Graph · cả hai · định giá"),
            ("Sinh", "kèm [Tin#ID] và Cypher")], 2.0, h=1.45, fills=["soft", "violet_bg", "amber_bg", "teal_bg"],
           size=18)
    s.stat(MX, 4.0, 3.6, f"{d.N(r + '.accuracy', 'pct0')}%", f"router đúng nhãn · {d.N(r + '.n')} câu", accent="violet")
    s.stat(MX + 3.9, 4.0, 3.6, f"{d.N(r + '.accuracy_excl_unanswerable', 'pct0')}%",
           f"không tính câu không trả lời được · {d.N(r + '.n_excl_unanswerable')} câu", accent="violet")
    s.text(8.2, 4.0, 4.5, 1.6, ["Graph rỗng hoặc lỗi → lui về Basic RAG", "RAG chạy song song, lui không phải chờ"],
           size=18, bullets=True)


def layers(d: Deck) -> None:
    s = d.slide("ev-layers", SEC_E, "Đánh giá ở hai tầng: truy xuất có tìm đúng tin, câu trả lời có đúng đáp án", notes=(
        "Một hệ RAG có thể sai ở hai tầng nên đo cả hai. Tầng truy xuất: Recall cho biết đã lấy đủ tin đúng trong top-k chưa, "
        "MRR cho biết tin đúng đầu tiên đứng hạng mấy. Tầng câu trả lời: so với đáp án chuẩn tính bằng SQL, có sai số cho "
        "phép, và câu không trả lời được thì phải từ chối. Faithfulness kiểm từng nhận định có căn cứ trong ngữ cảnh không. "
        "Chỉ số nào thấp thì sửa đúng khâu đó: Recall thấp sửa truy xuất, faithfulness thấp sửa prompt. "
        "Slide sau: thiết lập của lượt đánh giá."))
    tiers = [("Tầng 1 · Truy xuất", "có tìm đúng tin không?", ["Recall@k: đủ tin đúng trong top-k chưa",
                                                              "MRR: tin đúng đầu tiên đứng hạng mấy"], "blue"),
             ("Tầng 2 · Câu trả lời", "có đúng đáp án không?", ["So với đáp án SQL, có sai số cho phép",
                                                               "Câu không trả lời được: phải từ chối"], "teal")]
    cw = (W - 2 * MX - 0.4) / 2
    for i, (t, q, items, col) in enumerate(tiers):
        x = MX + i * (cw + 0.4)
        s.rect(x, 1.95, cw, 3.7, fill=f"{col}_bg", line=None, radius=0.05)
        s.text(x + 0.35, 2.15, cw - 0.7, 0.4, t.upper(), size=SIZE["caption"], bold=True, color=col)
        s.text(x + 0.35, 2.6, cw - 0.7, 0.7, q, size=26, bold=True, color="ink")
        s.text(x + 0.35, 3.5, cw - 0.7, 2.0, items, size=20, bullets=True)
    s.text(MX, 5.95, W - 2 * MX, 0.6, "Recall thấp → sửa truy xuất · đúng tin mà trả lời sai → sửa prompt hoặc Cypher",
           size=SIZE["small"], bold=True, color="teal")


def setup(d: Deck) -> None:
    s = d.slide("ev-setup", SEC_E, "Thiết lập: benchmark đủ chín nhóm, chạy trên tập con, chấm bằng luật", notes=(
        f"Benchmark có {d.N('benchmark_gt.json:items|len')} câu, đáp án chuẩn tính bằng SQL trên bảng sạch. Vì hạn mức API "
        f"miễn phí, lượt đánh giá chạy trên tập con {d.N(ES + ':n_questions')} câu, đủ chín nhóm, với cả bốn hệ. Mô hình sinh "
        "là bản miễn phí ghi trên slide, khác mô hình dự kiến ban đầu. Câu trả lời được chấm bằng luật so với đáp án chuẩn, "
        "không dùng LLM làm giám khảo. Vì vậy mọi con số phía sau là chỉ dấu trên cỡ mẫu nhỏ, và slide nào cũng ghi n với "
        "khoảng tin cậy. Slide sau: ablation truy xuất."))
    items = [(d.N("benchmark_gt.json:items|len"), "câu trong benchmark, đáp án SQL"),
             (d.N(ES + ":n_questions"), "câu đã chạy (tập con, đủ chín nhóm)"),
             (d.N(ES + ":systems|len"), "hệ so sánh trên cùng câu hỏi")]
    for i, (v, lab) in enumerate(items):
        s.stat(MX + i * 4.1, 1.95, 3.8, v, lab, accent="teal")
    s.text(MX, 3.75, W - 2 * MX, 0.4, "Mô hình sinh", size=SIZE["caption"], bold=True, color="teal")
    s.text(MX, 4.1, W - 2 * MX, 0.5, d.T(ES + ":llm.generator"), size=20, bold=True, color="ink")
    s.text(MX, 4.75, W - 2 * MX, 0.4, "Cách chấm", size=SIZE["caption"], bold=True, color="teal")
    rules = [d.A("Câu hỏi tìm tin: Recall@5 theo cụm tin trùng; ≥ 0,8 được 2 điểm, > 0 được 1 điểm",
                 "luật chấm, eval_summary.scoring_rules"),
             d.A("Câu hỏi ra số: có số trong câu trả lời lệch ≤ 5% so với đáp án SQL được 2 điểm, ≤ 20% được 1 điểm",
                 "luật chấm, eval_summary.scoring_rules"),
             d.A("Câu hỏi ra tên (quận, tháng): câu trả lời chứa đúng tên được 2 điểm", "luật chấm, eval_summary.scoring_rules"),
             "Câu không trả lời được: phải có câu từ chối mới được điểm",
             d.A("“Đúng” = đạt 2/2 điểm; faithfulness để trống vì cần LLM chấm", "định nghĩa accuracy trong eval")]
    s.text(MX, 5.1, W - 2 * MX, 1.5, rules, size=SIZE["small"], bullets=True, spacing=1.0)


def ablation(d: Deck) -> None:
    runs = d.V("ablation.json:runs") or []
    names = {"dense": "Dense", "bm25": "BM25", "hybrid": "Hybrid"}
    labels = [names.get(r.get("retrieval"), r.get("retrieval")) + (" + lọc" if r.get("prefilter") else "") for r in runs]
    best = max(runs, key=lambda r: r.get("recall_at_5") or 0) if runs else None
    lifted = runs and all((next((x["recall_at_5"] for x in runs if x["retrieval"] == r["retrieval"] and x["prefilter"]), 0)
                           > r["recall_at_5"]) for r in runs if not r["prefilter"])
    title = (f"Lọc trước tăng Recall@5 ở cả ba cách truy xuất; {labels[runs.index(best)]} cao nhất" if lifted and best
             else "Ablation truy xuất: Recall@5 và MRR của từng cấu hình")
    s = d.slide("ev-ablation", SEC_E, title, notes=(
        "Ablation chạy thẳng trên chỉ mục đầy đủ, không qua mô hình sinh, nên không tốn lời gọi API. Ba cách truy xuất dense, "
        "BM25 và hybrid, mỗi cách có và không có lọc trước. Rerank không có trong lượt đo vì model cross-encoder không có "
        f"trong cache. Số câu có tập tin đúng: {d.N('ablation.json:runs[0].n_questions')}. Bước nhảy lớn nhất đến từ bộ lọc, "
        "vì câu tìm tin có điều kiện cứng mà embedding không đảm bảo. Trên dữ liệu này BM25 có lọc ngang hoặc hơn hybrid "
        "có lọc, vì đề bài chứa tên dự án và con số khớp đúng từ. Hybrid không tự động thắng; phải đo mới biết. So "
        "sánh model embedding nằm ở chặng 1. Slide sau: kết quả câu trả lời theo nhóm câu hỏi."))
    s.chart(MX, 1.9, 8.2, 4.5, "column", labels or ["⟨?⟩"],
            [("Recall@5", "ablation.json:runs[*].recall_at_5", "teal"), ("MRR", "ablation.json:runs[*].mrr", "blue")],
            number_format="0.00", max_value=1)
    s.text(9.1, 1.95, 3.6, 4.3, ["Cùng chỉ mục e5-large đầy đủ", "Lọc trước = điều kiện cứng trong Qdrant",
                                 d.A("Không qua LLM, 0 lời gọi API", "mô tả thiết kế ablation")], size=18, bullets=True)


def results(d: Deck) -> None:
    cats, syss = categories(d), systems(d)
    acc = {s_: d.V(f"{ES}:overall.{s_}.accuracy") for s_ in syss}
    top = max((s_ for s_ in syss if acc.get(s_) is not None), key=lambda s_: acc[s_], default=None)
    title = (f"{SYSTEM[top][0]} đúng nhiều nhất trên tập con; mỗi hệ mạnh ở nhóm câu riêng" if top
             else "Độ chính xác bốn hệ theo chín nhóm câu hỏi")
    s = d.slide("ev-results", SEC_E, title, notes=(
        "Biểu đồ là độ chính xác của bốn hệ theo từng nhóm câu. Tổng trên tập con: "
        + ", ".join(f"{SYSTEM[x][0]} {d.N(f'{ES}:overall.{x}.accuracy', 'pct0')}%" for x in syss)
        + f". Mỗi nhóm chỉ có {d.N(f'{ES}:by_category.{cats[0]}.{syss[0]}.n')} câu, nên một câu đúng sai làm số đổi "
        "rất nhiều. Đọc theo hình dạng: hệ nào mạnh ở nhóm nào, chứ không so từng điểm phần trăm. Khoảng tin cậy ở slide sau."))
    s.chart(MX, 1.85, W - 2 * MX, 4.2, "column", [CATS[c] for c in cats],
            [(SYSTEM[x][0], [f"{ES}:by_category.{c}.{x}.accuracy" for c in cats], SYSTEM[x][1]) for x in syss],
            number_format="0%", labels=False, max_value=1)
    s.caption("Tổng · " + " · ".join(f"{SYSTEM[x][0]} {d.N(f'{ES}:overall.{x}.accuracy', 'pct0')}%" for x in syss)
              + f" · n = {d.N(ES + ':n_questions')} câu mỗi hệ", y=6.2)


def pairwise(d: Deck) -> None:
    rows_all = d.V(f"{ES}:pairwise") or []
    keep = [i for i, r in enumerate(rows_all) if r.get("metric") == "accuracy" and r.get("scope") in ("overall", "aggregate")]
    keep.sort(key=lambda i: (rows_all[i]["scope"] != "overall", rows_all[i]["a"], rows_all[i]["b"]))
    s = d.slide("ev-pairwise", SEC_E, "So từng cặp trên cùng câu hỏi: khoảng tin cậy cho biết chênh lệch nào đáng tin",
                notes=(
        "Bảng so ba cặp trên cùng câu hỏi: Basic RAG với LLM thuần, Hybrid với Basic RAG, và Hybrid với Graph, ở toàn tập "
        "và ở nhóm tổng hợp. Chênh lệch là hiệu độ chính xác tính bằng điểm phần trăm; khoảng tin cậy chín mươi "
        "lăm phần trăm lấy bằng bootstrap mười nghìn mẫu theo câu hỏi, ghép cặp giữa hai hệ. Khoảng nào chứa số không thì "
        "chưa đủ bằng chứng nói hệ này hơn hệ kia trên tập con này. Với bốn câu mỗi nhóm, phần lớn khoảng tin cậy rất rộng. "
        "Slide sau: chi phí và độ trễ."))
    rows = [["So sánh", "Phạm vi", "Chênh lệch", "CI95", "p"]]
    for i in keep:
        k = f"{ES}:pairwise[{i}]"
        r = rows_all[i]
        rows.append([f"{SYSTEM[r['a']][0]} − {SYSTEM[r['b']][0]}", "Tổng" if r["scope"] == "overall" else CATS[r["scope"]],
                     d.N(k + ".diff", "spct0") + " điểm %", f"{d.N(k + '.ci95[0]', 'spct0')} … {d.N(k + '.ci95[1]', 'spct0')}",
                     d.N(k + ".p_value", "dec2")])
    if len(rows) == 1:
        rows.append(["⟨?⟩"] * 5)
    s.table(MX, 1.95, W - 2 * MX, rows, col_w=[3.4, 2.6, 2.0, 2.6, 1.53], size=SIZE["small"], row_h=0.5)


def cost(d: Deck) -> None:
    syss = systems(d)
    c = f"{ES}:cost"
    lat = {x: d.V(f"{c}.{x}.latency_ms_p50") for x in syss}
    tok = {x: d.V(f"{c}.{x}.input_tokens_mean") for x in syss}
    known = [x for x in syss if lat.get(x) is not None]
    slowest = max(known, key=lambda x: lat[x]) if known else None
    heavy = max((x for x in syss if tok.get(x) is not None), key=lambda x: tok[x], default=None)
    title = (f"Chi phí và độ trễ: {SYSTEM[heavy][0]} tốn token nhất, {SYSTEM[slowest][0]} chậm nhất ở trung vị"
             if slowest and heavy else "Chi phí và độ trễ của bốn hệ")
    s = d.slide("ev-cost", SEC_E, title, notes=(
        "Độ trễ đo trên từng câu trong lượt eval, gồm router và Cypher nếu có. "
        + ", ".join(f"{SYSTEM[x][0]} trung vị {d.N(f'{c}.{x}.latency_ms_p50', 'k1')} giây" for x in syss)
        + ". Token vào phản ánh lượng ngữ cảnh: Graph và Hybrid nhận thêm kết quả Cypher và lượt router. Độ trễ còn phụ "
        "thuộc API miễn phí và số lượt thử lại, nên chỉ so tương đối. Các rủi ro triển khai đều có biện pháp trong code: "
        "giá là giá rao chứ không phải giá giao dịch, nên luôn kèm nguồn; số điện thoại bị ẩn trước khi lưu; Cypher chỉ đọc; "
        "API có giới hạn tần suất và cache. Slide sau: hệ thống sai vì đâu."))
    s.chart(MX, 1.9, 6.0, 3.9, "column", [SYSTEM[x][0] for x in syss],
            [("p50 (giây)", [f"{c}.{x}.latency_ms_p50" for x in syss], "teal", 0.001),
             ("p95 (giây)", [f"{c}.{x}.latency_ms_p95" for x in syss], "amber", 0.001)], number_format="0.0")
    s.chart(6.9, 1.9, 5.8, 3.9, "column", [SYSTEM[x][0] for x in syss],
            [("Token vào", [f"{c}.{x}.input_tokens_mean" for x in syss], "blue"),
             ("Token ra", [f"{c}.{x}.output_tokens_mean" for x in syss], "violet")], number_format="#,##0")
    s.text(MX, 6.0, W - 2 * MX, 0.5, "Giá rao ≠ giá giao dịch · ẩn SĐT trước khi lưu · Cypher chỉ đọc · giới hạn tần suất",
           size=SIZE["small"], color="text")


def errors(d: Deck) -> None:
    counts = d.V("error_analysis.json:counts") or {}
    syss = [x for x in systems(d) if x in counts] or ["basic_rag", "graph", "hybrid"]
    causes = [k for k in CAUSE if any(k in counts.get(x, {}) for x in syss)] or list(CAUSE)[:5]
    s = d.slide("ev-errors", SEC_E, "Hệ thống sai vì đâu: phân loại từng câu sai theo nguyên nhân", notes=(
        "Mỗi câu sai được gán đúng một nguyên nhân theo thứ tự ưu tiên: lỗi chạy, không từ chối khi phải từ chối, router chọn "
        "sai, Cypher lỗi, lọc sai, truy xuất trượt, bịa ngoài nguồn, Cypher chạy được nhưng ra sai, và còn lại là thiếu ngữ "
        "cảnh. Bảng đếm theo hệ. Hai ví dụ bên phải là câu sai thật trong lượt eval. Nguyên nhân phổ biến nhất của từng hệ "
        "là chỗ nên sửa trước. Slide sau sang phần mở rộng."))
    rows = [["Nguyên nhân"] + [SYSTEM[x][0] for x in syss]]
    for k in causes:
        rows.append([CAUSE[k]] + [d.N(f"error_analysis.json:counts.{x}.{k}") if k in counts.get(x, {}) else "–"
                                  for x in syss])
    s.table(MX, 1.95, 6.6, rows, col_w=[2.4] + [4.2 / len(syss)] * len(syss), size=SIZE["small"], row_h=0.44)
    ex = d.V("error_analysis.json:examples") or []
    picked, seen = [], set()
    for i, e in enumerate(ex):
        if e.get("system") != "llm_only" and e.get("id") not in seen:
            picked.append(i)
            seen.add(e.get("id"))
    for j, i in enumerate(picked[:2]):
        k = f"error_analysis.json:examples[{i}]"
        y = 1.95 + j * 2.3
        s.rect(7.5, y, 5.2, 2.1, fill="soft", line=None, radius=0.04)
        s.text(7.7, y + 0.1, 4.9, 0.4, f"{SYSTEM[ex[i]['system']][0]} · {CAUSE.get(ex[i]['cause'], '')}",
               size=SIZE["caption"], bold=True, color="coral")
        s.text(7.7, y + 0.5, 4.9, 1.5, "“" + _short(d, k + ".question", 150) + "”", size=SIZE["small"], color="ink")


def _short(d: Deck, key: str, limit: int) -> str:
    v = d.V(key)
    t = " ".join(str(v).split()) if v is not None else "⟨?⟩"
    return d.A(t if len(t) <= limit else t[:limit].rsplit(" ", 1)[0] + " …", f"trích nguyên văn {key}")
