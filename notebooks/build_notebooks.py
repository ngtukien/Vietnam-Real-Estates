"""Sinh và chạy 5 notebook trình bày từ `results/*.json` (Phase 3, Lane D).

Notebook chỉ ĐỌC kết quả đã có; không tính lại, không gọi mạng, không nạp model. Mọi con số hiển thị được in từ JSON
bởi code trong notebook (ô markdown không chứa số gõ tay). File thiếu (vd eval chưa chạy) ⇒ ô "chưa có" kèm lệnh sinh.

Dùng (từ gốc repo, sau `. scripts\\vre_env.ps1`):
    python notebooks/build_notebooks.py            # sinh lại nguồn + chạy + lưu output cho cả 5 notebook
    python notebooks/build_notebooks.py 02 04      # chỉ notebook có tiền tố 02, 04 (vd sau khi eval xong)
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parent
KERNEL = "python3"
TIMEOUT_S = 120

SETUP = r'''import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import IMPORTS_PLACEHOLDER

# Gốc dữ liệu: VRE_DATA_ROOT (vre_env.ps1 đặt về checkout chính) hoặc gốc repo (CI, chạy từ notebooks/).
_here = Path.cwd().resolve()
DATA_ROOT = Path(os.environ.get("VRE_DATA_ROOT") or (_here.parent if _here.name == "notebooks" else _here))
RESULTS = DATA_ROOT / "results"
pd.set_option("display.max_colwidth", 120)


def load(name):
    """Đọc results/<name>; trả None nếu file chưa được sinh."""
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def md(text):
    display(Markdown(text))


def missing(name, command):
    md(f"> **chưa có** — `results/{name}` chưa được sinh. Sinh bằng `{command}` rồi chạy lại notebook.")


def num(x):
    """Số nguyên kiểu Việt Nam: 2.537.920."""
    return f"{int(x):,}".replace(",", ".")


def dec(x, digits=2):
    """Số thực kiểu Việt Nam: 0,71."""
    return f"{x:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, digits=1):
    """Tỉ lệ 0..1 → phần trăm."""
    return dec(x * 100, digits) + "%"


def provenance(obj):
    if isinstance(obj, dict) and "generated_at" in obj:
        md(f"*Nguồn: sinh lúc `{obj['generated_at']}`, commit `{obj.get('git_sha')}`.*")


md("Đọc `results/` dưới " + ("`VRE_DATA_ROOT`." if os.environ.get("VRE_DATA_ROOT") else "gốc repo."))'''


def setup(with_image: bool = False) -> str:
    """Ô cài đặt chung; chỉ notebook hiển thị ảnh mới import `Image` (ruff F401)."""
    return SETUP.replace("IMPORTS_PLACEHOLDER", "Image, Markdown, display" if with_image else "Markdown, display")


def md(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(src.strip())


def code(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(src.strip())


# ---------------------------------------------------------------------------------------------- 01 EDA
NB01 = [
    md("""
# Khám phá dữ liệu (EDA)

Notebook đọc `results/ingest_report.json`, `results/eda_summary.json` và các hình trong `results/eda/` do lệnh
`python scripts/vre_pipeline.py ingest` và `python scripts/vre_pipeline.py eda` sinh ra. Không đọc lại dữ liệu thô.
"""),
    code(setup(with_image=True)),
    md("## Nạp và làm sạch"),
    code(r'''ing = load("ingest_report.json")
if ing is None:
    missing("ingest_report.json", "python scripts/vre_pipeline.py ingest")
else:
    provenance(ing)
    d, p = ing["dedup"], ing["price"]
    md(
        f"Dữ liệu thô có **{num(ing['raw_rows'])}** dòng trong {num(ing['shards'])} shard, nạp trong {dec(ing['seconds'], 1)} giây. "
        f"Bỏ {num(ing['dropped']['no_text'])} tin không có chữ và {num(ing['dropped']['bad_type'])} tin sai loại. "
        f"Khử trùng lặp: {num(d['exact_dup'])} tin trùng chính xác, {num(d['prefix_dup'])} tin trùng tiền tố; "
        f"còn lại **{num(d['kept'])}** tin. SimHash gom {num(d['simhash_listings_in_multi'])} tin vào "
        f"{num(d['simhash_clusters_multi'])} cụm gần trùng (giữ lại, đánh dấu `dup_cluster_id`)."
    )
    md(
        f"Giá: {num(p['null_or_unparsable'])} tin thiếu hoặc không đọc được, {num(p['fixed_thousand'])} tin sửa đơn vị nghìn đồng, "
        f"{num(p['outliers'])} tin giá ngoại lai; {num(ing['area_outliers'])} tin diện tích ngoại lai. "
        f"Loại giao dịch: {num(ing['deal_type']['sale'])} tin bán, {num(ing['deal_type']['rent'])} tin thuê. "
        f"Va chạm `listing_id`: {num(ing['listing_id_collisions'])}."
    )
    display(pd.Series(ing["missing_pct"], name="% thiếu").to_frame())'''),
    md("## Tổng quan bảng sạch"),
    code(r'''eda = load("eda_summary.json")
if eda is None:
    missing("eda_summary.json", "python scripts/vre_pipeline.py eda")
else:
    provenance(eda)
    md(
        f"Bảng sạch có **{num(eda['n_listings'])}** tin, đăng từ `{eda['period']['min_month']}` đến `{eda['period']['max_month']}`. "
        f"Hà Nội và Hồ Chí Minh chiếm {pct(eda['hn_hcm_share'])} số tin."
    )
    dc, tk = eda["description_chars"], eda["tokens_per_listing"]
    md(
        f"Độ dài mô tả (ký tự) p50/p90/p99: {num(dc['p50'])} / {num(dc['p90'])} / {num(dc['p99'])}; "
        f"số token mỗi tin p50/p90/p99: {num(tk['p50'])} / {num(tk['p90'])} / {num(tk['p99'])}."
    )
    display(pd.DataFrame(eda["by_property_type"]).rename(columns={"median_price_m2_mil": "trung vị giá/m² (triệu)"}))'''),
    code(r'''if eda is not None:
    prov = pd.DataFrame(eda["top_provinces"])
    prov["share_%"] = (prov["share"] * 100).round(2)
    display(prov.drop(columns="share"))'''),
    code(r'''if eda is not None:
    monthly = pd.DataFrame(eda["monthly"])
    fig, ax1 = plt.subplots(figsize=(9, 3.5))
    ax1.bar(monthly["month"], monthly["n"], color="#9ecae1", label="số tin")
    ax1.set_ylabel("số tin")
    ax2 = ax1.twinx()
    ax2.plot(monthly["month"], monthly["median_price_m2_mil"], color="#d62728", marker="o", label="trung vị giá/m²")
    ax2.set_ylabel("triệu đồng/m²")
    ax1.set_title("Số tin và trung vị giá/m² theo tháng (đọc từ eda_summary.monthly)")
    ax1.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    plt.show()
    display(monthly)'''),
    md("## Hình EDA"),
    code(r'''if eda is not None:
    for fig_meta in eda["figures"]:
        path = RESULTS / "eda" / fig_meta["file"]
        md(f"### {fig_meta['title']}\n\n{fig_meta['caption']}")
        if path.exists():
            display(Image(filename=str(path), width=760))
        else:
            missing(f"eda/{fig_meta['file']}", "python scripts/vre_pipeline.py eda")'''),
]

# ---------------------------------------------------------------------------------------------- 02 RAG
NB02 = [
    md("""
# Truy hồi (RAG): embedding, chỉ mục và ablation

Đọc `results/embedding_benchmark.json` (số đo so sánh model embedding), `results/night_report.json` (lượt dựng chỉ mục đầy đủ),
`results/ablation.json` và `results/examples.json` (sinh ở bước đánh giá). File chưa có sẽ hiện ô "chưa có".
"""),
    code(setup()),
    md("## Vì sao chọn `multilingual-e5-large`"),
    code(r'''bench = load("embedding_benchmark.json")
if bench is None:
    missing("embedding_benchmark.json", "python scripts/vre_pipeline.py ingest")
else:
    cols = ["model", "params_m", "dim", "recall_at_1", "recall_at_10", "mrr_at_10", "docs_per_s", "hours_2_8m_chunks",
            "peak_vram_gb", "token_budget"]
    emb = pd.DataFrame(bench)[cols].sort_values("mrr_at_10", ascending=False).reset_index(drop=True)
    display(emb)
    best, fastest = emb.iloc[0], emb.sort_values("docs_per_s", ascending=False).iloc[0]
    q = bench[0]
    md(
        f"Cùng một phép đo ({num(q['corpus'])} tin ngẫu nhiên đã khử trùng, {num(q['queries'])} truy vấn tiêu đề → mô tả, fp16): "
        f"**`{best['model']}`** có MRR@10 cao nhất ({dec(best['mrr_at_10'], 3)}), recall@1 {dec(best['recall_at_1'], 3)}, "
        f"recall@10 {dec(best['recall_at_10'], 3)}. Model nhanh nhất `{fastest['model']}` đạt {dec(fastest['docs_per_s'], 1)} tin/giây "
        f"nhưng MRR@10 chỉ {dec(fastest['mrr_at_10'], 3)}. Với `{best['model']}` ước tính "
        f"{dec(best['hours_2_8m_chunks'], 2)} giờ cho toàn bộ chunk, chấp nhận được cho một lượt chạy qua đêm, "
        f"nên chất lượng truy hồi được ưu tiên hơn tốc độ."
    )'''),
    code(r'''if bench is not None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(emb["docs_per_s"], emb["mrr_at_10"], s=emb["params_m"] / 2, alpha=0.6)
    for _, r in emb.iterrows():
        ax.annotate(r["model"].split("/")[-1], (r["docs_per_s"], r["mrr_at_10"]), fontsize=8)
    ax.set_xlabel("tin/giây (càng phải càng nhanh)")
    ax.set_ylabel("MRR@10 (càng cao càng tốt)")
    ax.set_title("Chất lượng và tốc độ embedding (kích thước chấm = số tham số)")
    fig.tight_layout()
    plt.show()
    sweep = pd.DataFrame({m["model"]: {k: v["docs_per_s"] for k, v in m["sweep"].items()} for m in bench})
    sweep.index.name = "token_budget"
    md("Thông lượng (tin/giây) theo ngân sách token mỗi lô:")
    display(sweep)'''),
    md("## Cấu hình embedding cố định"),
    code(r'''night = load("night_report.json")
if night is None:
    missing("night_report.json", "python scripts/vre_pipeline.py night --resume")
else:
    md(
        f"Chỉ mục dùng model `{night['model_slug']}`, collection `{night['collection']}`, "
        f"ngân sách token mỗi lô `{night['embed']['token_budget']}` (lấy từ số đo trên, không đo lại)."
    )'''),
    md("## Lượt dựng chỉ mục đầy đủ"),
    code(r'''if night is not None:
    provenance(night)
    c, qd = night["counters"], night["qdrant"]
    md(
        f"Trạng thái **{night['status']}**: bắt đầu `{night['started_at']}`, xong `{night['finished_at']}`, tổng "
        f"{dec(night['wall_hours'], 2)} giờ chạy qua {num(night['resumed_runs'])} lần resume. "
        f"{num(c['chunks'])} chunk từ {num(c['parts_total'])} phần; Qdrant `{qd['status']}` với {num(qd['points'])} point "
        f"và {num(qd['indexed_vectors'])} vector đã đánh chỉ mục. Tốc độ embed trung bình "
        f"{dec(night['embed']['chunks_per_s_avg'], 1)} chunk/giây; BM25 avgdl {dec(night['bm25']['avgdl'], 2)}."
    )
    steps = pd.DataFrame(night["steps"])[["name", "status", "seconds", "items_done", "items_total", "note"]]
    display(steps)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(steps["name"], steps["seconds"] / 60)
    ax.invert_yaxis()
    ax.set_xlabel("phút (lần chạy cuối)")
    ax.set_title("Thời gian từng bước trong night_report.steps")
    fig.tight_layout()
    plt.show()'''),
    code(r'''if night is not None:
    display(pd.Series(night["counters"], name="counters").to_frame())
    peaks = {k: ("không ghi" if v is None else v) for k, v in night["peaks"].items()}
    display(pd.Series(peaks, name="peaks").to_frame())
    if night["errors"]:
        md("Lỗi / cảnh báo ghi trong báo cáo:")
        display(pd.DataFrame(night["errors"]))
    else:
        md("Không có lỗi hay cảnh báo nào được ghi.")'''),
    md("## Ablation truy hồi"),
    code(r'''abl = load("ablation.json")
if abl is None:
    missing("ablation.json", "python scripts/vre_pipeline.py ablation")
else:
    provenance(abl)
    co = abl["corpus"]
    md(f"Subcorpus {num(co['n_listings'])} tin (chứa {num(co['n_gt_listings'])} tin đáp án), seed `{co['seed']}`.")
    runs = pd.DataFrame(abl["runs"])
    display(runs)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(runs["name"], runs["recall_at_5"])
    ax.invert_yaxis()
    ax.set_xlabel("recall@5")
    ax.set_title("Ablation: recall@5 theo cấu hình")
    fig.tight_layout()
    plt.show()
    top = runs.sort_values("recall_at_5", ascending=False).iloc[0]
    md(f"Cấu hình tốt nhất theo recall@5: `{top['name']}` ({dec(top['recall_at_5'], 3)}, MRR {dec(top['mrr'], 3)}).")'''),
    md("## Ví dụ truy hồi"),
    code(r'''examples = load("examples.json")
if examples is None:
    missing("examples.json", "python scripts/vre_pipeline.py report examples")
else:
    provenance(examples)
    for ex in examples.get("rag", []):
        md(f"**Câu hỏi:** {ex['question']}  \nBộ lọc trích được: `{json.dumps(ex['filters'], ensure_ascii=False)}`")
        display(pd.DataFrame(ex["hits"]))'''),
]

# ---------------------------------------------------------------------------------------------- 03 KG
NB03 = [
    md("""
# Đồ thị tri thức (Neo4j)

Đọc `results/graph_report.json` (lệnh `python scripts/vre_pipeline.py graph`) và phần `graph` của `results/examples.json`.
"""),
    code(setup()),
    md("""
## Lược đồ

```mermaid
graph LR
  Listing -->|OF_TYPE| PropertyType
  Listing -->|IN_WARD| Ward
  Listing -->|IN_DISTRICT| District
  Listing -->|IN_PROVINCE| Province
  Listing -->|ON_STREET| Street
  Listing -->|IN_PROJECT| Project
  Listing -->|POSTED_IN| Month
  Listing -->|HAS_AMENITY| Amenity
  Ward -->|IN_DISTRICT| District
  Street -->|IN_DISTRICT| District
  Project -->|IN_DISTRICT| District
  District -->|IN_PROVINCE| Province
  Ward -->|IN_CLUSTER| Cluster
  MarketStat -->|STAT_OF| Province
  MarketStat -->|STAT_OF| District
  MarketStat -->|STAT_OF| Ward
  MarketStat -->|STAT_OF| Project
```

Tin gắn vào cấp hành chính chi tiết nhất có được (phường, nếu thiếu thì quận, nếu thiếu nữa thì tỉnh). `MarketStat` là thống kê giá
tính sẵn theo vùng × loại × tháng để câu Cypher tổng hợp chạy nhanh; `Cluster` là cộng đồng Leiden trên đồ thị kNN các phường.
"""),
    md("## Quy mô đồ thị"),
    code(r'''gr = load("graph_report.json")
if gr is None:
    missing("graph_report.json", "python scripts/vre_pipeline.py graph")
else:
    provenance(gr)
    nodes = pd.Series(gr["nodes"], name="số node").sort_values(ascending=False)
    rels = pd.Series(gr["relationships"], name="số cạnh").sort_values(ascending=False)
    md(
        f"Nạp bằng `neo4j-admin import` trong {dec(gr['import_seconds'], 1)} giây: **{num(nodes.sum())}** node, "
        f"**{num(rels.sum())}** cạnh. Bất biến: số Listing khớp bảng sạch = `{gr['invariants']['listing_count_ok']}`, "
        f"tổng giá khớp = `{gr['invariants']['price_sum_ok']}`."
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    nodes.plot.barh(ax=axes[0], logx=True, title="Node theo nhãn (thang log)")
    rels.plot.barh(ax=axes[1], logx=True, title="Cạnh theo kiểu (thang log)")
    for ax in axes:
        ax.invert_yaxis()
    fig.tight_layout()
    plt.show()
    display(nodes.to_frame())
    display(rels.to_frame())'''),
    md("## Cộng đồng phường"),
    code(r'''if gr is not None:
    cm = gr["communities"]
    md(
        f"Leiden trên đồ thị kNN (k = {num(cm['k'])}, seed {num(cm['seed'])}) của {num(cm['n_wards'])} phường đủ dữ liệu cho "
        f"{num(cm['n_clusters'])} cụm, modularity {dec(cm['modularity'], 3)}."
    )'''),
    md("## Ví dụ Text2Cypher"),
    code(r'''examples = load("examples.json")
graph_examples = None if examples is None else examples.get("graph")
if not graph_examples:
    missing("examples.json", "python scripts/vre_pipeline.py report examples")
else:
    provenance(examples)
    for ex in graph_examples:
        md(f"**Câu hỏi:** {ex['question']}  \n*{num(ex['latency_ms'])} ms*\n\n```cypher\n{ex['cypher']}\n```")
        display(pd.DataFrame(ex["rows"], columns=ex["columns"]))'''),
]

# ---------------------------------------------------------------------------------------------- 04 eval
NB04 = [
    md("""
# Đánh giá: LLM only, Basic RAG, Graph, Hybrid

Đọc `results/eval_summary.json` (độ chính xác theo nhóm câu hỏi kèm khoảng tin cậy bootstrap, so sánh cặp, router, chi phí),
`results/error_analysis.json` và phần `answers` của `results/examples.json`. Sinh bằng `python scripts/vre_pipeline.py eval`.
"""),
    code(setup()),
    md("## Cấu hình lượt đánh giá"),
    code(r'''ev = load("eval_summary.json")
if ev is None:
    missing("eval_summary.json", "python scripts/vre_pipeline.py eval")
else:
    provenance(ev)
    md(
        f"{num(ev['n_questions'])} câu hỏi × {num(len(ev['systems']))} hệ thống ({', '.join(f'`{s}`' for s in ev['systems'])}). "
        f"Chỉ mục `{ev['index']['collection']}` ({num(ev['index']['n_points'])} point, model `{ev['index']['model_slug']}`). "
        f"LLM sinh `{ev['llm']['generator']}`, LLM chấm `{ev['llm']['judge']}`."
    )'''),
    md("## Kết quả tổng"),
    code(r'''if ev is not None:
    overall = pd.DataFrame(ev["overall"]).T
    overall["ci95_low"] = overall["ci95"].map(lambda c: c[0])
    overall["ci95_high"] = overall["ci95"].map(lambda c: c[1])
    overall = overall.drop(columns="ci95").astype(float).sort_values("accuracy", ascending=False)
    display(overall)
    best = overall.index[0]
    md(
        f"Hệ có độ chính xác tổng cao nhất: **`{best}`** ({pct(overall.loc[best, 'accuracy'])}, "
        f"CI95 {pct(overall.loc[best, 'ci95_low'])} – {pct(overall.loc[best, 'ci95_high'])})."
    )'''),
    md("## Theo nhóm câu hỏi"),
    code(r'''if ev is not None:
    rows = [
        {"category": cat, "system": sys_, "n": v["n"], "accuracy": v["accuracy"], "mean_points": v["mean_points"],
         "ci95_low": v["ci95"][0], "ci95_high": v["ci95"][1]}
        for cat, by_sys in ev["by_category"].items() for sys_, v in by_sys.items()
    ]
    cat_df = pd.DataFrame(rows)
    acc = cat_df.pivot(index="category", columns="system", values="accuracy")
    display(acc)
    systems = list(acc.columns)
    width = 0.8 / max(len(systems), 1)
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, s in enumerate(systems):
        sub = cat_df[cat_df["system"] == s].set_index("category").reindex(acc.index)
        xs = [j + i * width for j in range(len(acc.index))]
        err = [sub["accuracy"] - sub["ci95_low"], sub["ci95_high"] - sub["accuracy"]]
        ax.bar(xs, sub["accuracy"], width, yerr=err, capsize=2, label=s)
    ax.set_xticks([j + width * (len(systems) - 1) / 2 for j in range(len(acc.index))])
    ax.set_xticklabels(acc.index, rotation=30, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_title("Độ chính xác theo nhóm (thanh lỗi = CI95 bootstrap)")
    ax.legend()
    fig.tight_layout()
    plt.show()'''),
    md("## So sánh cặp (bootstrap ghép cặp)"),
    code(r'''if ev is not None:
    pw = pd.DataFrame(ev["pairwise"])
    if pw.empty:
        md("Không có so sánh cặp trong `eval_summary.pairwise`.")
    else:
        pw["ci95_low"] = pw["ci95"].map(lambda c: c[0])
        pw["ci95_high"] = pw["ci95"].map(lambda c: c[1])
        pw["có_ý_nghĩa"] = (pw["ci95_low"] > 0) | (pw["ci95_high"] < 0)
        display(pw.drop(columns="ci95"))
        sig = pw[pw["có_ý_nghĩa"]]
        md(
            f"{num(len(sig))} / {num(len(pw))} so sánh có CI95 của hiệu không chứa 0. "
            "Các so sánh còn lại không đủ bằng chứng để kết luận hệ nào tốt hơn."
        )'''),
    md("## Truy hồi và RAGAS"),
    code(r'''if ev is not None:
    display(pd.DataFrame(ev["retrieval"]).T)
    display(pd.DataFrame(ev["ragas"]).T)'''),
    md("## Router"),
    code(r'''if ev is not None:
    rt = ev["router"]
    md(f"Router chọn đúng tuyến ở {pct(rt['accuracy'])} câu hỏi. Ma trận nhầm lẫn (hàng = tuyến đúng, cột = tuyến router chọn):")
    display(pd.DataFrame(rt["confusion"]).T.fillna(0).astype(int))'''),
    md("## Chi phí và độ trễ"),
    code(r'''if ev is not None:
    cost = pd.DataFrame(ev["cost"]).T
    display(cost)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(cost.index, cost["latency_ms_p50"], label="p50")
    ax.bar(cost.index, cost["latency_ms_p95"] - cost["latency_ms_p50"], bottom=cost["latency_ms_p50"], alpha=0.4, label="p95")
    ax.set_ylabel("ms")
    ax.set_title("Độ trễ mỗi câu hỏi")
    ax.legend()
    fig.tight_layout()
    plt.show()'''),
    md("## Phân tích lỗi"),
    code(r'''err = load("error_analysis.json")
if err is None:
    missing("error_analysis.json", "python scripts/vre_pipeline.py eval")
else:
    provenance(err)
    counts = pd.DataFrame(err["counts"]).fillna(0).astype(int)
    display(counts)
    if not counts.empty:
        counts.T.plot.bar(stacked=True, figsize=(8, 3.5), title="Nguyên nhân lỗi theo hệ thống").legend(fontsize=8)
        plt.tight_layout()
        plt.show()
    if err["examples"]:
        display(pd.DataFrame(err["examples"]))'''),
    md("## Ví dụ câu trả lời"),
    code(r'''examples = load("examples.json")
answers = None if examples is None else examples.get("answers")
if not answers:
    missing("examples.json", "python scripts/vre_pipeline.py report examples")
else:
    for a in answers:
        md(f"**`{a['system']}`** (route `{a['route']}`) — *{a['question']}*\n\n{a['answer']}\n\nTrích dẫn: `{a['cited']}`")'''),
]

# ---------------------------------------------------------------------------------------------- 05 valuation
NB05 = [
    md("""
# Định giá (module phụ)

Đọc `results/valuation_metrics.json`: model XGBoost dự đoán giá/m² của tin bán, biến thể `tabular` (đặc trưng cấu trúc) và
`full` (thêm vector mô tả nén PCA), so với baseline trung vị phường. Sinh bằng `python scripts/vre_pipeline.py price --variant tabular|full`.
"""),
    code(setup()),
    md("## Chia tập và phạm vi"),
    code(r'''val = load("valuation_metrics.json")
if val is None:
    missing("valuation_metrics.json", "python scripts/vre_pipeline.py price --variant tabular")
else:
    provenance(val)
    sp, rent = val["split"], val["rent_model"]
    md(
        f"Train: tháng `{sp['train_months'][0]}` → `{sp['train_months'][-1]}`; test: `{sp['test_months'][0]}` → "
        f"`{sp['test_months'][-1]}` (chia theo thời gian, không trộn). Model thuê: "
        + ("đã huấn luyện." if rent["trained"] else f"không huấn luyện ({num(rent['n_rent'])} tin thuê; lý do: `{rent['reason']}`).")
    )'''),
    md("## So sánh biến thể"),
    code(r'''if val is not None:
    variants = {k: v for k, v in val["variants"].items() if v is not None}
    cmp_df = pd.DataFrame(
        {k: {"n_train": v["n_train"], "n_test": v["n_test"], "mae_mil_m2": v["mae_mil_m2"], "mape": v["mape"], "r2": v["r2"],
             "train_seconds": v["train_seconds"]} for k, v in variants.items()}
    ).T
    display(cmp_df)
    if val["variants"].get("full") is None:
        missing("valuation_metrics.json (variants.full)", "python scripts/vre_pipeline.py price --variant full")
    base = val["variants"]["baseline_ward_median"]
    for name in ("tabular", "full"):
        v = val["variants"].get(name)
        if v is None:
            continue
        verb = "giảm" if v["mape"] < base["mape"] else "không giảm"
        md(
            f"`{name}`: MAPE {pct(v['mape'])} so với baseline phường {pct(base['mape'])} ({verb}); "
            f"MAE {dec(v['mae_mil_m2'], 1)} so với {dec(base['mae_mil_m2'], 1)} triệu/m²; R² {dec(v['r2'], 3)} so với {dec(base['r2'], 3)}."
        )
    if val["variants"].get("full") is not None:
        t, f = val["variants"]["tabular"], val["variants"]["full"]
        md(f"Thêm vector mô tả (`full`) đổi MAPE từ {pct(t['mape'])} thành {pct(f['mape'])} trên cùng tập test.")'''),
    md("## Theo loại bất động sản"),
    code(r'''if val is not None:
    parts = []
    for name, v in variants.items():
        df = pd.DataFrame(v["by_property_type"])
        if not df.empty:
            parts.append(df.set_index("property_type")["mape"].rename(name))
    by_type = pd.concat(parts, axis=1) if parts else pd.DataFrame()
    display(by_type)
    if not by_type.empty:
        by_type.plot.barh(figsize=(8, 3.5), title="MAPE theo loại bất động sản").invert_yaxis()
        plt.tight_layout()
        plt.show()'''),
    md("## Theo tỉnh (nhiều tin nhất)"),
    code(r'''if val is not None:
    parts = []
    for name, v in variants.items():
        df = pd.DataFrame(v["by_province_top10"])
        if not df.empty:
            parts.append(df.set_index("province")["mape"].rename(name))
    display(pd.concat(parts, axis=1) if parts else pd.DataFrame())'''),
    md("## Đặc trưng quan trọng (SHAP)"),
    code(r'''if val is not None:
    shap_variants = {k: v for k, v in variants.items() if v["shap_top"]}
if val is not None and shap_variants:
    fig, axes = plt.subplots(1, len(shap_variants), figsize=(6 * len(shap_variants), 4.5), squeeze=False)
    for ax, (name, v) in zip(axes[0], shap_variants.items(), strict=True):
        s = pd.DataFrame(v["shap_top"]).head(15).set_index("feature")["mean_abs_contribution"]
        s.plot.barh(ax=ax, title=f"{name}: trung bình |SHAP| (mười lăm đặc trưng đầu)")
        ax.invert_yaxis()
    fig.tight_layout()
    plt.show()'''),
]

NOTEBOOKS = {
    "01_eda.ipynb": NB01,
    "02_rag.ipynb": NB02,
    "03_knowledge_graph.ipynb": NB03,
    "04_graphrag_eval.ipynb": NB04,
    "05_valuation.ipynb": NB05,
}


def build(name: str, cells: list[nbformat.NotebookNode]) -> Path:
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": KERNEL, "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    NotebookClient(nb, timeout=TIMEOUT_S, kernel_name=KERNEL, resources={"metadata": {"path": str(NB_DIR)}}).execute()
    for cell in nb.cells:  # bỏ dấu thời gian thực thi để diff ổn định
        cell.get("metadata", {}).pop("execution", None)
    path = NB_DIR / name
    nbformat.write(nb, path)
    return path


def main(argv: list[str]) -> int:
    selected = [n for n in NOTEBOOKS if not argv or any(n.startswith(a) for a in argv)]
    if not selected:
        print(f"không có notebook khớp {argv}; có: {list(NOTEBOOKS)}", file=sys.stderr)
        return 2
    for name in selected:
        print("đã chạy và lưu", build(name, NOTEBOOKS[name]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
