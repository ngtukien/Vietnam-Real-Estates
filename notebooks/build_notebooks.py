"""Sinh + chạy 6 notebook trình bày theo bố cục bản cũ (phase-03-lanes §4c): `00_setup` dùng chung, mã cell `Px-yy`.

Mỗi bước = một ô markdown `### Px-yy · <kết luận một câu>` + đoạn giải thích, rồi một ô code có dòng đầu
`# Px-yy · ẨN|CACHE|LIVE`. Con số trong heading được ĐIỀN ở đây từ `results/*.json` (gốc dữ liệu `VRE_DATA_ROOT`) và
cache cell `results/nb_cache/<id>.json` (gốc code) — không gõ tay. Thiếu dữ liệu ⇒ heading nói rõ "chưa có".

Dùng (từ gốc repo, sau `. scripts\\vre_env.ps1`):
    python notebooks/build_notebooks.py            # sinh lại nguồn + chạy + lưu output cả 6 notebook
    python notebooks/build_notebooks.py 02 04      # chỉ notebook có tiền tố 02, 04
Môi trường: `VRE_NB_OFFLINE=1` ⇒ mọi cell LIVE/CACHE đọc cache (như CI); `VRE_NB_REBUILD=1` ⇒ tính lại cell CACHE.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import nbformat

NB_DIR = Path(__file__).resolve().parent
CODE_ROOT = NB_DIR.parent
KERNEL = "python3"
TIMEOUT_S = 300
KINDS = ("ẨN", "CACHE", "LIVE")
NB_CACHE = CODE_ROOT / "results" / "nb_cache"


# ------------------------------------------------------------------------------------------- ngữ cảnh điền heading
class Ctx:
    """Đọc JSON kết quả (gốc dữ liệu) và cache cell (gốc code) để điền số vào heading."""

    def __init__(self) -> None:
        self.data_root = Path(os.environ.get("VRE_DATA_ROOT") or CODE_ROOT)

    def r(self, name: str) -> dict | list | None:
        path = self.data_root / "results" / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def c(self, cell_id: str) -> dict | None:
        path = NB_CACHE / f"{cell_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def N(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", ".")


def D(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def P(x: float, digits: int = 1) -> str:
    return D(x * 100, digits) + "%"


@dataclass(frozen=True)
class Step:
    id: str
    kind: str
    heading: Callable[[Ctx], str] | str
    explain: str
    code: str


def _head(step: Step, ctx: Ctx) -> str:
    text = step.heading(ctx) if callable(step.heading) else step.heading
    return f"### {step.id} · {text}\n**{step.kind}**\n\n{step.explain.strip()}"


RUN_SETUP = "%run ./00_setup.ipynb"


# =============================================================================================== 00_setup
SETUP_01 = r'''# 00-01 · ẨN
import json
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

_here = Path.cwd().resolve()
CODE_ROOT = _here.parent if _here.name == "notebooks" else _here
os.environ.setdefault("VRE_DATA_ROOT", str(CODE_ROOT))      # CI: dữ liệu = repo; máy: vre_env.ps1 trỏ về checkout chính
os.environ.setdefault("VRE_DUCKDB_MEMORY_GB", "3")           # notebook chạy chung máy với dịch vụ: DuckDB nhỏ
os.environ.setdefault("VRE_DUCKDB_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")                 # không bao giờ tải model
DATA_ROOT = Path(os.environ["VRE_DATA_ROOT"])
RESULTS = DATA_ROOT / "results"
NB_CACHE = CODE_ROOT / "results" / "nb_cache"
OFFLINE = os.environ.get("VRE_NB_OFFLINE") == "1"
STRICT = os.environ.get("VRE_NB_STRICT") == "1"                # build_notebooks.py: lỗi live là lỗi build, không lặng lẽ lùi cache
REBUILD = os.environ.get("VRE_NB_REBUILD") == "1"
if str(CODE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "src"))
SEED = 42
np.random.seed(SEED)
pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 160)
display(Markdown("Chế độ: **" + ("offline — mọi cell đọc cache" if OFFLINE else "live khi có dịch vụ/model, không thì đọc cache")
                 + "**; kết quả đọc từ `results/` dưới " + ("`VRE_DATA_ROOT`." if DATA_ROOT != CODE_ROOT else "gốc repo.")))'''

SETUP_02 = r'''# 00-02 · ẨN
from vre.config import EMBED_MODEL_SLUG, EMBED_MODELS, get_settings

S = get_settings()
config_view = {
    "embedding": EMBED_MODELS[EMBED_MODEL_SLUG][0], "collection": S.collection, "alias": S.collection_alias,
    "chunk_size": S.chunk_size, "chunk_overlap": S.chunk_overlap, "header_max_tokens": S.header_max_tokens,
    "top_k": S.top_k, "rrf_k": S.rrf_k, "llm_provider": S.llm_provider, "llm_model": S.llm_model, "judge_model": S.judge_model,
}
display(pd.Series(config_view, name="giá trị").to_frame())'''

SETUP_03 = r'''# 00-03 · ẨN
def md(text):
    display(Markdown(text))


def num(x):
    return f"{int(round(x)):,}".replace(",", ".")


def dec(x, digits=2):
    return f"{x:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, digits=1):
    return dec(x * 100, digits) + "%"


def load(name):
    """results/<name> (JSON) hoặc None nếu chưa sinh."""
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def jsonl(name):
    path = RESULTS / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else None


def missing(what, command):
    md(f"> **chưa có** — {what}. Sinh bằng `{command}` rồi chạy lại notebook.")


_AVAILABLE = {}


def _check(name):
    if name == "data":
        return S.clean_path.exists()
    if name == "answers":
        return (RESULTS / "eval" / "raw" / "answers.jsonl").exists()
    if name == "llmcache":
        return (RESULTS / "cache" / "llm").is_dir()
    if name == "models":
        return (S.models_dir / "valuation" / "xgb_tabular_sale.json").exists()
    if name == "hf":
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(EMBED_MODELS[EMBED_MODEL_SLUG][0], "model.safetensors"), str)
    if name == "qdrant":
        from vre.rag import qdrant_search
        return (qdrant_search.client().get_collection(qdrant_search.collection()).points_count or 0) > 0
    if name == "neo4j":
        from vre.graph.client import run_cypher
        return int(run_cypher("RETURN 1 AS ok", timeout=5)["ok"].iloc[0]) == 1
    raise ValueError(f"không biết điều kiện {name!r}")


def available(name):
    """Dịch vụ/model/dữ liệu có dùng được không (kiểm một lần mỗi kernel). OFFLINE ⇒ luôn False."""
    if OFFLINE:
        return False
    if name not in _AVAILABLE:
        try:
            _AVAILABLE[name] = bool(_check(name))
        except Exception:
            _AVAILABLE[name] = False
    return _AVAILABLE[name]


# Một cài đặt duy nhất cho chế độ chỉ-cache: `vre.llm.client` (cũng là `VRE_LLM_OFFLINE=1` của pipeline/eval).
from vre.llm.client import LLMCacheMiss, cache_only as llm_cache_only  # noqa: E402,F401


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _cache_file(cell_id):
    return NB_CACHE / f"{cell_id}.json"


def read_cache(cell_id):
    path = _cache_file(cell_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_cache(cell_id, data, mode):
    data = _plain(data)
    data["_meta"] = {"cell": cell_id, "mode": mode, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    NB_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = _cache_file(cell_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(_cache_file(cell_id))
    return data


def _from_cache(cell_id, why):
    data = read_cache(cell_id)
    if data is None:
        md(f"> **chưa có** — {why} và chưa có cache cell `results/nb_cache/{cell_id}.json`. Chạy notebook trên máy có dịch vụ để sinh.")
    else:
        md(f"*Đọc cache cell `results/nb_cache/{cell_id}.json` ({why}; sinh lúc `{data['_meta']['generated_at']}`).*")
    return data


def live(cell_id, compute, needs=()):
    """LIVE: chạy thật khi đủ điều kiện `needs` rồi ghi cache cell; thiếu điều kiện hoặc lỗi ⇒ đọc cache cell."""
    absent = [n for n in needs if not available(n)]
    if OFFLINE or absent:
        return _from_cache(cell_id, "chế độ offline" if OFFLINE else "thiếu " + ", ".join(absent))
    try:
        with llm_cache_only():
            data = compute()
    except Exception as error:
        if STRICT:
            raise
        return _from_cache(cell_id, f"chạy live lỗi {type(error).__name__}")
    md("*Chạy live, đã ghi cache cell.*")
    return _write_cache(cell_id, data, "live")


def cached(cell_id, compute, needs=()):
    """CACHE: đọc cache cell nếu có (trừ VRE_NB_REBUILD=1); chưa có và đủ điều kiện ⇒ tính một lần rồi ghi."""
    data = read_cache(cell_id)
    if data is not None and not REBUILD:
        md(f"*Đọc cache cell `results/nb_cache/{cell_id}.json` (sinh lúc `{data['_meta']['generated_at']}`).*")
        return data
    absent = [n for n in needs if not available(n)]
    if OFFLINE or absent:
        return _from_cache(cell_id, "chế độ offline" if OFFLINE else "thiếu " + ", ".join(absent))
    with llm_cache_only():
        data = compute()
    md("*Tính từ dữ liệu, đã ghi cache cell.*")
    return _write_cache(cell_id, data, "cache")


def table(records, columns=None):
    display(pd.DataFrame(records, columns=columns))'''

NB00_TITLE = """# 00 · Thiết lập dùng chung

Mọi notebook gọi `%run ./00_setup.ipynb` ở đầu. Ba cell đều ẨN: đường dẫn và chế độ chạy, cấu hình, hàm dùng chung
(đọc kết quả, cache cell, kiểm tra dịch vụ, chặn lời gọi LLM thật)."""

NB00 = [
    Step("00-01", "ẨN", "Mọi notebook dùng chung đường dẫn, seed và chế độ chạy live/offline", """
Mục tiêu: xác định gốc code (repo chứa notebook) và gốc dữ liệu (`VRE_DATA_ROOT`, trên máy trỏ về checkout chính, trong CI là
repo), thêm `src/` vào `sys.path` để import package `vre`, cố định seed. `VRE_NB_OFFLINE` buộc mọi cell đọc cache — đây là cách
CI chạy notebook khi không có Docker, GPU hay khoá LLM. DuckDB trong notebook bị giới hạn bộ nhớ vì chạy chung máy với Qdrant/Neo4j.
""", SETUP_01),
    Step("00-02", "ẨN", "Cấu hình embedding, chỉ mục, chunk và LLM nằm ở một chỗ: `vre.config`", """
Mục tiêu: đọc `Settings` (pydantic-settings, tiền tố `VRE_`) — cùng nguồn với pipeline, API và eval, nên notebook không có hằng
số riêng. Bảng chỉ in khoá không bí mật; khoá API không bao giờ xuất hiện trong notebook. Cách đọc: `collection` là collection
vật lý, `alias` là tên ổn định mà truy vấn dùng.
""", SETUP_02),
    Step("00-03", "ẨN", "Hàm dùng chung: đọc kết quả, cache cell, kiểm tra dịch vụ, chặn LLM thật", """
Mục tiêu: ba kiểu cell. **CACHE** đọc `results/*.json` hoặc cache cell `results/nb_cache/<id>.json` (tính một lần từ dữ liệu rồi
commit). **LIVE** chạy một demo nhẹ thật khi dịch vụ/model có mặt và ghi lại cache cell; không có thì đọc cache cell, nên notebook
luôn chạy được. Mọi lời gọi LLM trong notebook chỉ đọc cache đĩa LLM (`llm_cache_only`): notebook không tốn token và không cần khoá.
""", SETUP_03),
]


# =============================================================================================== 01_rag_indexing
def h_p1_01(c: Ctx) -> str:
    d = c.c("P1-01")
    ing = c.r("ingest_report.json")
    if d is None or ing is None:
        return "Dataset thô gồm nhiều shard Parquet (chưa có số đo)"
    return f"Dataset thô có {N(ing['raw_rows'])} dòng trong {N(ing['shards'])} shard, {N(len(d['raw_columns']))} cột"


def h_p1_02(c: Ctx) -> str:
    e = c.r("eda_summary.json")
    if e is None:
        return "Khảo sát thiếu dữ liệu (chưa có eda_summary)"
    m = e["missing_pct"]
    return (f"Tiêu đề và mô tả đầy đủ, nhưng {D(m['bedrooms'], 1)}% tin thiếu số phòng ngủ và "
            f"{D(m['project'], 1)}% thiếu dự án")


def h_p1_03(c: Ctx) -> str:
    e = c.r("eda_summary.json")
    if e is None:
        return "Phân phối giá/m² (chưa có eda_summary)"
    by = {r["property_type"]: r["median_price_m2_mil"] for r in e["by_property_type"]}
    return (f"Giá/m² trung vị của nhà gấp {D(by['Nhà'] / by['Đất'], 1)} lần đất; Hà Nội + Hồ Chí Minh chiếm "
            f"{P(e['hn_hcm_share'])} số tin")


def h_p1_04(c: Ctx) -> str:
    i = c.r("ingest_report.json")
    if i is None:
        return "Làm sạch và khử trùng (chưa có ingest_report)"
    d = i["dedup"]
    return (f"Làm sạch giữ {N(d['kept'])} trên {N(i['raw_rows'])} dòng: bỏ {N(d['exact_dup'])} trùng chính xác và "
            f"{N(d['prefix_dup'])} trùng tiền tố")


def h_p1_07(c: Ctx) -> str:
    d = c.c("P1-07")
    if d is None:
        return "Chunking theo cửa sổ token (chưa chạy demo)"
    ok = "khớp" if d["n_formula"] == d["n_actual"] else "KHÔNG khớp"
    return f"Mô tả dài {N(d['body_tokens'])} token ⇒ n = {N(d['n_formula'])} chunk theo công thức, {ok} đếm thực {N(d['n_actual'])}"


def h_p1_08(c: Ctx) -> str:
    n, e = c.r("night_report.json"), c.r("eda_summary.json")
    if n is None or e is None:
        return "Chunk toàn bộ bảng sạch (chưa có night_report)"
    chunks = n["counters"]["chunks"]
    return f"{N(e['n_listings'])} tin thành {N(chunks)} chunk, trung bình {D(chunks / e['n_listings'], 2)} chunk mỗi tin"


def h_p1_09(c: Ctx) -> str:
    d = c.c("P1-09")
    if d is None:
        return "Câu gần nghĩa nằm gần nhau trong không gian embedding (chưa chạy demo)"
    return (f"Câu gần nghĩa có cosine {D(d['cos_pair'], 2)}, câu khác chủ đề chỉ {D(d['cos_off'], 2)}")


def h_p1_10(c: Ctx) -> str:
    b = c.r("embedding_benchmark.json")
    if not b:
        return "So sánh model embedding (chưa có embedding_benchmark)"
    best = max(b, key=lambda m: m["mrr_at_10"])
    return f"`{best['model'].split('/')[-1]}` có MRR@10 cao nhất ({D(best['mrr_at_10'], 3)}) trong {N(len(b))} model đã đo"


def h_p1_11(c: Ctx) -> str:
    n = c.r("night_report.json")
    if n is None:
        return "Lượt chạy dài embed + upload (chưa có night_report)"
    return (f"Lượt chạy dài {n['status']}: {N(n['counters']['chunks'])} chunk, tổng {D(n['wall_hours'], 1)} giờ qua "
            f"{N(n['resumed_runs'])} lần resume")


def h_p1_12(c: Ctx) -> str:
    n = c.r("night_report.json")
    if n is None:
        return "Qdrant chứa đúng số chunk (chưa có night_report)"
    q = n["qdrant"]
    ok = "bằng" if q["points"] == n["counters"]["chunks"] else "KHÁC"
    return f"Qdrant có {N(q['points'])} point, {ok} số chunk; trạng thái `{q['status']}`"


def h_p1_13(c: Ctx) -> str:
    d = c.c("P1-13")
    if d is None:
        return "Chỉ mục tìm được tin liên quan nhưng chưa lọc đúng điều kiện (chưa chạy demo)"
    return f"Dense search trả {N(d['k'])} tin liên quan, chỉ {N(d['n_all_ok'])}/{N(d['k'])} thoả đủ điều kiện trong câu hỏi"


NB01_TITLE = """# 01 · Nền tảng RAG và Indexing

Từ dữ liệu thô đến chỉ mục vector trên **toàn bộ** dataset: khảo sát (EDA), làm sạch và khử trùng, chunk theo token,
embedding `multilingual-e5-large`, lưu dense + BM25 vào Qdrant.

**Đầu ra cho notebook sau:** bảng sạch `data/clean/listings.parquet` (khoá `listing_id`) và collection Qdrant sau alias `vre_chunks`.
**Lệnh sinh:** `python scripts/vre_pipeline.py ingest`, `eda`, `night --resume` (notebook chỉ đọc kết quả của chúng)."""

NB01 = [
    Step("P1-01", "CACHE", h_p1_01, """
Mục tiêu: xem dataset `tinixai/vietnam-real-estates` trông thế nào trước khi xử lý. Schema đọc từ metadata Parquet của shard đầu
(không đọc dữ liệu), dòng mẫu lấy từ **bảng sạch** — số điện thoại và email trong mô tả đã bị ẩn, nên in ra an toàn. Cách đọc:
bảng sạch đổi tên cột sang tiếng Anh thống nhất (`price_vnd`, `area_m2`…) và thêm cột suy ra (`deal_type`, `dup_cluster_id`).
""", r'''# P1-01 · CACHE
def compute():
    import pyarrow.parquet as pq

    raw = pq.ParquetFile(sorted(S.raw_dir.glob("*.parquet"))[0]).schema_arrow
    clean = pq.ParquetFile(S.clean_path)
    cols = ["listing_id", "title", "property_type", "deal_type", "province", "district", "ward", "price_vnd", "area_m2", "bedrooms"]
    sample = clean.read_row_group(0, columns=cols).slice(0, 3).to_pandas()
    return {"raw_columns": [[f.name, str(f.type)] for f in raw],
            "clean_columns": clean.schema_arrow.names, "n_clean": clean.metadata.num_rows,
            "sample": sample.astype(object).where(sample.notna(), None).to_dict("records")}


d = cached("P1-01", compute, needs=("data",))
if d:
    table(d["raw_columns"], ["cột thô", "kiểu"])
    md(f"Bảng sạch: {num(d['n_clean'])} dòng, {num(len(d['clean_columns']))} cột. Ba dòng mẫu:")
    table(d["sample"])'''),
    Step("P1-02", "CACHE", h_p1_02, """
Mục tiêu: đo tỷ lệ thiếu theo cột, độ dài mô tả và số token mỗi tin. Ý nghĩa: cột thiếu nhiều (dự án, số phòng) không thể là bộ lọc
bắt buộc; mô tả là nguồn thông tin chính nên RAG trên văn bản là cần thiết. Số token quyết định cỡ chunk: phần lớn tin vừa một
cửa sổ, chỉ đuôi dài cần cắt. Hình do lệnh `eda` vẽ sẵn, notebook chỉ hiển thị.
""", r'''# P1-02 · CACHE
eda = load("eda_summary.json")
if eda is None:
    missing("`results/eda_summary.json`", "python scripts/vre_pipeline.py eda")
else:
    miss = pd.Series(eda["missing_pct"], name="% thiếu").sort_values(ascending=False)
    display(miss.to_frame().T)
    dc, tk = eda["description_chars"], eda["tokens_per_listing"]
    md(f"Mô tả p50/p90/p99: {num(dc['p50'])} / {num(dc['p90'])} / {num(dc['p99'])} ký tự; "
       f"token mỗi tin p50/p90/p99: {num(tk['p50'])} / {num(tk['p90'])} / {num(tk['p99'])}.")
    display(Image(filename=str(RESULTS / "eda" / "missing_values.png"), width=720))'''),
    Step("P1-03", "CACHE", h_p1_03, """
Mục tiêu: phân phối giá/m² theo loại, theo vùng, theo tháng và theo tỉnh. Giá/m² lệch phải mạnh nên dùng **trung vị**, không dùng
trung bình. Dữ liệu tập trung ở hai thành phố lớn: kết luận cho tỉnh ít tin kém tin cậy hơn, và benchmark cần câu hỏi ở cả
tỉnh nhỏ. Tiêu đề và chú thích của từng hình nằm trong `eda_summary.figures`.
""", r'''# P1-03 · CACHE
if eda is not None:
    display(pd.DataFrame(eda["by_property_type"]))
    for f in eda["figures"]:
        if f["file"] in ("price_m2_by_type.png", "price_m2_by_region.png", "monthly_trend.png", "top_provinces.png",
                         "top_districts_hcm_hn.png"):
            md(f"**{f['title']}** — {f['caption']}")
            display(Image(filename=str(RESULTS / "eda" / f["file"]), width=720))'''),
    Step("P1-04", "CACHE", h_p1_04, """
Mục tiêu: làm sạch bằng DuckDB trên cả mười shard. Bước: bỏ tin không có chữ; trùng chính xác (cùng tiêu đề + mô tả chuẩn hoá) và
trùng tiền tố bị gộp; tin gần trùng (đăng lại có sửa chữ) được gom cụm bằng SimHash nhưng **giữ lại**, đánh dấu `dup_cluster_id`
để eval chấm đúng khi hệ trả một bản sao. Giá nhập theo nghìn đồng được sửa; giá/diện tích ngoại lai bị gắn cờ, không bị xoá.
""", r'''# P1-04 · CACHE
ing = load("ingest_report.json")
if ing is None:
    missing("`results/ingest_report.json`", "python scripts/vre_pipeline.py ingest")
else:
    rows = {"dòng thô": ing["raw_rows"], "bỏ: không có chữ": ing["dropped"]["no_text"], "bỏ: trùng chính xác": ing["dedup"]["exact_dup"],
            "bỏ: trùng tiền tố": ing["dedup"]["prefix_dup"], "còn lại": ing["dedup"]["kept"],
            "tin trong cụm gần trùng": ing["dedup"]["simhash_listings_in_multi"], "cụm gần trùng": ing["dedup"]["simhash_clusters_multi"],
            "giá sửa đơn vị nghìn": ing["price"]["fixed_thousand"], "giá ngoại lai (cờ)": ing["price"]["outliers"],
            "diện tích ngoại lai (cờ)": ing["area_outliers"], "va chạm listing_id": ing["listing_id_collisions"]}
    display(pd.Series(rows, name="số tin").to_frame())
    for name in ("duplicates.png", "outliers.png"):
        display(Image(filename=str(RESULTS / "eda" / name), width=640))'''),
    Step("P1-05", "LIVE", "Địa danh, giá và thông tin liên hệ được chuẩn hoá bằng hàm thuần, kiểm được trên từng chuỗi", """
Mục tiêu: thấy tận mắt các luật làm sạch trên vài chuỗi tự đặt. `normalize_place` là entity resolution cho địa danh (viết tắt,
tiền tố, số quận); `parse_price` đọc giá dạng chữ; `mask_pii` ẩn số điện thoại và email trước khi văn bản đi vào chỉ mục hay LLM.
Cell không cần dịch vụ nào; chỉ lấy cache khi chạy offline.
""", r'''# P1-05 · LIVE
def compute():
    from vre.data.clean import mask_pii, parse_price
    from vre.data.places import normalize_place

    places = [("Q.7", "district"), ("q7", "district"), ("P.5", "ward"), ("TP. Hồ Chí Minh", "province"), ("Huyện Củ Chi", "district")]
    prices = ["7450000000", "abc", "5000000", None]
    texts = ["Liên hệ 0901 234 567 hoặc a@b.com", "Nhà đẹp, sổ hồng chính chủ"]
    return {"places": [[raw, kind, normalize_place(raw, kind)] for raw, kind in places],
            "prices": [[str(p), parse_price(pd.Series([p])).iloc[0]] for p in prices],
            "pii": [[t, m] for t, m in zip(texts, mask_pii(pd.Series(texts)), strict=True)]}


d = live("P1-05", compute)
if d:
    table(d["places"], ["gốc", "loại", "chuẩn hoá"])
    table(d["prices"], ["giá gốc", "price_vnd"])
    table(d["pii"], ["văn bản", "sau khi ẩn"])'''),
    Step("P1-06", "ẨN", "Mọi bước sau đọc chung một bảng sạch; `listing_id` ổn định giữa các lần chạy", """
Mục tiêu: chỉ ra hợp đồng dữ liệu dùng chung. `listing_id` là băm của nội dung chuẩn hoá (không phải số thứ tự dòng), nên chạy lại
ingest cho cùng id; Qdrant (`point_id = listing_id × 64 + chunk`), Neo4j và đáp án chuẩn của benchmark đều khoá theo nó.
Dấu vân tay (số tin, tổng giá) được các bước sau kiểm để không trộn chỉ mục cũ với bảng sạch mới.
""", r'''# P1-06 · ẨN
md(f"Bảng sạch: `{S.clean_path.relative_to(S.data_root).as_posix()}` — "
   + ("có trên máy." if S.clean_path.exists() else "không có trên máy này (CI): các cell cần dữ liệu đọc cache."))
gr = load("graph_report.json")
if gr is not None:
    md(f"Bất biến đồ thị so với bảng sạch: số tin khớp = `{gr['invariants']['listing_count_ok']}`, tổng giá khớp = `{gr['invariants']['price_sum_ok']}`.")'''),
    Step("P1-07", "LIVE", h_p1_07, """
Mục tiêu: cắt một mô tả dài theo cửa sổ token của chính tokenizer e5-large. Mỗi chunk = header (tiêu đề + loại · địa chỉ · giá ·
diện tích, lặp ở mọi chunk để chunk nào cũng tự đủ ngữ cảnh) + một cửa sổ mô tả; các cửa sổ chồng lấn để câu không bị cắt mất ý.
Với ngân sách S cho phần mô tả và chồng lấn O, số chunk là n = ⌈(L − O) / (S − O)⌉ khi L > S. Cell so công thức với đếm thực.
""", r'''# P1-07 · LIVE
def compute():
    import pyarrow.parquet as pq

    from vre.index.chunk import PART_COLUMNS, chunk_rows, listing_header, n_chunks_formula
    from vre.index.tokenize import load_fast_tokenizer, special_ids

    tok = load_fast_tokenizer()
    special = special_ids(tok)
    rows = pq.ParquetFile(S.clean_path).read_row_group(0, columns=list(PART_COLUMNS)).slice(0, 5000).to_pylist()
    row = max(rows, key=lambda r: len(r["description"] or ""))
    body = len(tok(row["description"], add_special_tokens=False)["input_ids"])
    chunks, _, _ = chunk_rows([row], tok, special, S)
    h_ids = len(tok(listing_header(row), add_special_tokens=False)["input_ids"][:S.header_max_tokens])
    budget = S.chunk_size - h_ids - len(special.newline)
    return {"listing_id": row["listing_id"], "body_tokens": body, "header_tokens": h_ids, "budget": budget,
            "overlap": S.chunk_overlap, "n_formula": n_chunks_formula(body, budget, S.chunk_overlap), "n_actual": len(chunks),
            "chunks": [[int(r.chunk_index), int(r.n_tokens), r.text.replace("\n", " · ")[:110]] for r in chunks.itertuples()]}


d = live("P1-07", compute, needs=("data", "hf"))
if d:
    md(f"Tin `{d['listing_id']}`: mô tả L = {num(d['body_tokens'])} token, header {num(d['header_tokens'])} token ⇒ ngân sách mô tả "
       f"S = {num(d['budget'])}, chồng lấn O = {num(d['overlap'])}; công thức n = {num(d['n_formula'])}, đếm thực {num(d['n_actual'])}.")
    table(d["chunks"], ["chunk", "token", "đầu đoạn"])'''),
    Step("P1-08", "CACHE", h_p1_08, """
Mục tiêu: quy mô chunk trên toàn bộ bảng sạch. Bước chunk chạy theo phần (part) độc lập để resume được; tin vượt trần số chunk mỗi
tin bị cắt đuôi và được đếm. Cách đọc: số chunk mỗi tin gần một nghĩa là đa số tin vừa một cửa sổ, chỉ tin dài mới tốn thêm vector.
""", r'''# P1-08 · CACHE
night = load("night_report.json")
if night is None:
    missing("`results/night_report.json`", "python scripts/vre_pipeline.py night --resume")
else:
    display(pd.Series(night["counters"], name="counters").to_frame())'''),
    Step("P1-09", "LIVE", h_p1_09, """
Mục tiêu: embedding biến câu thành vector sao cho câu cùng nghĩa có cosine cao, kể cả khi khác từ ngữ ("chung cư" và "căn hộ",
"hẻm ô tô" và "hẻm xe hơi"). Cell nhúng năm câu bằng encoder truy vấn e5-large (CPU, fp16), in ma trận cosine và chiếu hai chiều bằng
PCA. Lưu ý: PCA hai chiều chỉ để nhìn; khoảng cách thật đo trên toàn bộ số chiều.
""", r'''# P1-09 · LIVE
SENTENCES = ["căn hộ 2 phòng ngủ ở trung tâm Quận 1", "chung cư hai phòng ngủ gần trung tâm quận 1",
             "nhà phố hẻm xe hơi Gò Vấp", "nhà riêng trong hẻm ô tô ở Gò Vấp", "công thức nấu phở bò Hà Nội"]


def compute():
    from vre.rag import query_encoder

    v = np.array([query_encoder.embed(t) for t in SENTENCES])
    cos = v @ v.T
    centered = v - v.mean(axis=0)
    xy = centered @ np.linalg.svd(centered, full_matrices=False)[2][:2].T
    return {"cos": cos.round(4).tolist(), "xy": xy.round(4).tolist(), "dim": v.shape[1],
            "cos_pair": float(cos[0, 1]), "cos_off": float(cos[0, 4])}


d = live("P1-09", compute, needs=("hf",))
if d:
    display(pd.DataFrame(d["cos"], index=SENTENCES, columns=[f"câu {i + 1}" for i in range(len(SENTENCES))]).round(3))
    fig, ax = plt.subplots(figsize=(6, 4))
    xy = np.array(d["xy"])
    ax.scatter(xy[:, 0], xy[:, 1])
    for (x, y), text in zip(xy, SENTENCES, strict=True):
        ax.annotate(text, (x, y), fontsize=8)
    ax.set_title(f"Năm câu chiếu PCA hai chiều (vector gốc {d['dim']} chiều)")
    plt.show()'''),
    Step("P1-10", "CACHE", h_p1_10, """
Mục tiêu: lý do chốt `multilingual-e5-large`. Bốn model được đo cùng một phép thử (tin ngẫu nhiên đã khử trùng, truy vấn tiêu đề →
mô tả, fp16) về chất lượng (recall, MRR) và tốc độ (tin/giây, VRAM). Model nhanh hơn nhiều lần nhưng MRR thấp hơn; với một lượt chạy
qua đêm, chất lượng truy hồi đáng giá hơn. Đây là nguồn duy nhất cho so sánh model; ablation chỉ đổi cấu hình truy hồi.
""", r'''# P1-10 · CACHE
bench = load("embedding_benchmark.json")
if bench is None:
    missing("`results/embedding_benchmark.json`", "python scripts/vre_pipeline.py ingest")
else:
    cols = ["model", "params_m", "dim", "recall_at_1", "recall_at_10", "mrr_at_10", "docs_per_s", "hours_2_8m_chunks", "peak_vram_gb", "token_budget"]
    emb = pd.DataFrame(bench)[cols].sort_values("mrr_at_10", ascending=False).reset_index(drop=True)
    display(emb)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.scatter(emb["docs_per_s"], emb["mrr_at_10"])
    for _, r in emb.iterrows():
        ax.annotate(r["model"].split("/")[-1], (r["docs_per_s"], r["mrr_at_10"]), fontsize=8)
    ax.set_xlabel("tin/giây")
    ax.set_ylabel("MRR@10")
    ax.set_title("Chất lượng và tốc độ embedding")
    plt.show()'''),
    Step("P1-11", "CACHE", h_p1_11, """
Mục tiêu: lượt chạy dài embed toàn bộ chunk trên GPU rồi upload Qdrant, dựng HNSW, kiểm tra và tính PCA cho model định giá. Pipeline
ghi trạng thái từng phần nên bị ngắt thì chạy lại đúng lệnh để tiếp tục. Cách đọc bảng bước: thời gian là của lần chạy cuối; bước
đã xong ở lần trước hiện thời gian bằng không. Cảnh báo (vd số đo đỉnh không ghi được) nằm ở `errors`.
""", r'''# P1-11 · CACHE
if night is not None:
    steps = pd.DataFrame(night["steps"])[["name", "status", "seconds", "items_done", "items_total", "note"]]
    display(steps)
    md(f"Embed trung bình {dec(night['embed']['chunks_per_s_avg'], 1)} chunk/giây, ngân sách token mỗi lô `{night['embed']['token_budget']}`.")
    if night["errors"]:
        display(pd.DataFrame(night["errors"]))'''),
    Step("P1-12", "CACHE", h_p1_12, """
Mục tiêu: kiểm tra chỉ mục khớp dữ liệu. Mỗi point có vector dense (e5-large) và vector sparse BM25 (từ tách bằng `pyvi`, IDF do
Qdrant tính) cùng payload (tỉnh, quận, loại, giá, diện tích, số phòng, tháng) có index để lọc trước. `indexed_vectors` đếm cả hai loại
vector nên gấp đôi số point; trạng thái `green` nghĩa là HNSW đã dựng xong.
""", r'''# P1-12 · CACHE
if night is not None:
    display(pd.Series({**night["qdrant"], "collection": night["collection"], "chunks": night["counters"]["chunks"]}, name="qdrant").to_frame())'''),
    Step("P1-13", "LIVE", h_p1_13, """
Mục tiêu: kiểm tra nhanh chỉ mục bằng một câu hỏi có điều kiện cứng (loại, quận, số phòng, giá) và chỉ dense search, không lọc.
Kết quả cho thấy giới hạn của tìm theo nghĩa: tin "giống" câu hỏi nhưng sai quận hoặc vượt giá vẫn lọt top-k. Đó là lý do
notebook `02` thêm bộ lọc trích từ câu hỏi và lọc trước trên payload.
""", r'''# P1-13 · LIVE
QUESTION = "căn hộ 2 phòng ngủ quận 7 dưới 4 tỷ"


def compute():
    from vre.rag.retrieve import hits_table, search_dense

    hits = search_dense(QUESTION, k=S.top_k)
    ok = pd.DataFrame({"đúng quận": hits["district"].eq("Quận 7"), "đúng số phòng": hits["bedrooms"].eq(2),
                       "dưới giá": hits["price_vnd"].le(4e9), "đúng loại": hits["property_type"].eq("Căn hộ chung cư")})
    view = hits_table(hits).join(ok)
    return {"k": len(hits), "n_all_ok": int(ok.all(axis=1).sum()), "rows": view.astype(object).where(view.notna(), None).to_dict("records")}


d = live("P1-13", compute, needs=("qdrant", "hf"))
if d:
    md(f"Câu hỏi: *{QUESTION}* — {num(d['n_all_ok'])}/{num(d['k'])} tin thoả đủ bốn điều kiện.")
    table(d["rows"])'''),
]


# =============================================================================================== 02_retrieval_generation
def h_p2_02(c: Ctx) -> str:
    d = c.c("P2-02")
    if d is None:
        return "LLM trích bộ lọc thành JSON, pydantic kiểm tra và chuẩn hoá đơn vị (chưa chạy demo)"
    five = next(v for t, v in d["money"] if t == "5 tỷ")
    return f"LLM trích bộ lọc thành JSON; pydantic kiểm tra và đổi '5 tỷ' thành {N(five)} triệu"


def h_p2_03(c: Ctx) -> str:
    d = c.c("P2-03")
    if d is None:
        return "Cosine đo hướng, không đo độ dài (chưa chạy demo)"
    v = d["values"]
    return f"Cosine đo hướng, không đo độ dài: (6, 8) vẫn được {D(v[0][1], 2)}, (4, 3) được {D(v[1][1], 2)}, (−4, 3) được {D(v[2][1], 2)}"


def h_p2_04(c: Ctx) -> str:
    d = c.c("P2-04")
    if d is None:
        return "Tăng k chỉ thêm tin điểm thấp hơn (chưa chạy demo)"
    s = d["scores"]
    return f"Không có ngưỡng cosine: tăng k tới {N(len(s))} chỉ thêm tin điểm thấp dần, từ {D(s[0], 3)} xuống {D(s[-1], 3)}"


def h_p2_05(c: Ctx) -> str:
    d = c.c("P2-05")
    if d is None:
        return "BM25 khớp đúng từ khoá, dense khớp theo nghĩa (chưa chạy demo)"
    k = d["k"]
    b, de = d["bm25_both"], d["dense_both"]
    cmp = "nhiều hơn" if b > de else ("bằng" if b == de else "ít hơn")
    return f"Top-{N(k)} BM25 có {N(b)} tin chứa cả 'sổ hồng' và '2PN', {cmp} dense ({N(de)})"


def h_p2_06(c: Ctx) -> str:
    d = c.c("P2-06")
    if d is None:
        return "Hai bảng xếp hạng gộp thành một bằng RRF (chưa chạy demo)"
    return f"RRF gộp hai bảng xếp hạng: toy đưa `{d['toy'][0][0]}` lên đầu; câu thật có {N(d['overlap'])} tin xuất hiện ở cả hai bảng"


def h_p2_07(c: Ctx) -> str:
    d = c.c("P2-07")
    if d is None:
        return "Pre-filter giữ đủ tin đúng điều kiện, post-filter làm rơi (chưa chạy demo)"
    return f"Pre-filter giữ {N(d['pre'])}/{N(d['k'])} tin đúng điều kiện, post-filter chỉ còn {N(d['post'])}/{N(d['k'])}"


def h_p2_08(c: Ctx) -> str:
    a = c.r("ablation.json")
    if a is None:
        return "Rerank cross-encoder: tác dụng đo ở ablation (chưa có ablation.json)"
    on = [r for r in a["runs"] if r["rerank"]]
    off = [r for r in a["runs"] if not r["rerank"] and r["retrieval"] == "hybrid"]
    if not on or not off:
        return "Rerank cross-encoder: ablation không có cặp bật/tắt rerank"
    return f"Rerank đổi recall@5 từ {D(max(r['recall_at_5'] for r in off), 3)} thành {D(max(r['recall_at_5'] for r in on), 3)} trên ablation"


def h_p2_09(c: Ctx) -> str:
    d = c.c("P2-09")
    if d is None:
        return "Câu trả lời bám nguồn, mỗi ý có [Tin#ID] (chưa có câu trả lời eval)"
    return (f"Câu trả lời `{d['id']}` trích {N(len(d['cited']))} nguồn, {N(len(d['unsupported']))} trích dẫn không có trong ngữ cảnh")


def h_p2_11(c: Ctx) -> str:
    e = c.r("eval_summary.json")
    if e is not None:
        by = e["by_category"]
        if "lookup" in by and "aggregate" in by:
            lk, ag = by["lookup"]["basic_rag"], by["aggregate"]["basic_rag"]
            return (f"Basic RAG đúng {P(lk['accuracy'])} câu tra cứu (n = {N(lk['n'])}) nhưng {P(ag['accuracy'])} câu tổng hợp "
                    f"(n = {N(ag['n'])})")
    d = c.c("P2-11")
    if d is None:
        return "Basic RAG mạnh ở tra cứu, yếu ở tổng hợp (chưa có kết quả eval)"
    return f"Kết quả eval tạm ({N(d['n_rows'])} dòng chấm): Basic RAG theo nhóm câu hỏi"


NB02_TITLE = """# 02 · Retrieval và Generation

Biến câu hỏi thành truy vấn có bộ lọc, tìm bằng dense, BM25 và hybrid (RRF), lọc trước trên payload, rerank tuỳ chọn, rồi sinh
câu trả lời có trích dẫn `[Tin#ID]`.

**Đầu ra cho notebook sau:** `vre.rag.retrieve.retrieve`, `vre.rag.generate`, gói lại thành `vre.qa.answer("basic_rag", …)`.
**Cần trước:** chỉ mục của notebook `01`; LLM chỉ đọc cache đĩa (không gọi API)."""

NB02 = [
    Step("P2-01", "ẨN", "Dùng lại chỉ mục và encoder của notebook 01, không embed lại", """
Mục tiêu: kiểm tra dịch vụ nào đang có trên máy này. Cell LIVE phía dưới chạy thật khi điều kiện của nó có mặt (Qdrant, model trong
cache Hugging Face, dữ liệu sạch, cache LLM), không thì đọc cache cell. Encoder truy vấn chạy trên CPU ở fp16: một câu ngắn mất
một phần nhỏ của giây, nên không cần GPU.
""", r'''# P2-01 · ẨN
display(pd.Series({n: available(n) for n in ("qdrant", "hf", "data", "llmcache", "answers")}, name="có trên máy này").to_frame())'''),
    Step("P2-02", "LIVE", h_p2_02, """
Mục tiêu: hiểu câu hỏi. LLM trích bộ lọc (tỉnh, quận, loại, số phòng, khoảng giá, diện tích, tháng) bằng structured output; pydantic
(`QueryFilters`) kiểm tra kiểu, chuẩn hoá địa danh như dữ liệu sạch và đổi tiền về triệu đồng. Phần đổi tiền là hàm thuần nên luôn
chạy; phần LLM chỉ có khi câu đã nằm trong cache LLM. Cách đọc: trường không có trong câu hỏi để trống, nghĩa là không lọc.
""", r'''# P2-02 · LIVE
from vre.eval.questions import load_questions

FILTER_QUESTION = next(q.question for q in load_questions() if q.id == "CS01")  # câu ràng buộc đã chạy eval: bộ lọc có trong cache LLM


def compute():
    from vre.rag.filters import parse_filters, parse_money_mil

    out = {"money": [[t, parse_money_mil(t)] for t in ("5 tỷ", "1 tỷ 2", "1 tỷ 50 triệu", "850 triệu", "3.000.000.000")],
           "question": FILTER_QUESTION, "filters": None}
    try:
        filters, _ = parse_filters(FILTER_QUESTION)
        out["filters"] = filters.model_dump(exclude_none=True)
    except LLMCacheMiss:
        pass
    return out


d = live("P2-02", compute)
if d:
    table(d["money"], ["chuỗi", "triệu đồng"])
    if d["filters"] is None:
        md(f"Câu *{d['question']}* chưa có trong cache LLM: không trích bộ lọc (notebook không gọi API).")
    else:
        md(f"Bộ lọc trích từ *{d['question']}*: `{json.dumps(d['filters'], ensure_ascii=False)}`")'''),
    Step("P2-03", "LIVE", h_p2_03, """
Mục tiêu: cosine = tích vô hướng chia tích độ dài, tức chỉ đo góc giữa hai vector. Vector cùng hướng gấp đôi độ dài vẫn được cosine
tối đa; vector vuông góc được không. Vì e5 trả vector đã chuẩn hoá L2, Qdrant dùng tích vô hướng là đủ. Cell tính bằng numpy cho
`q = (3, 4)` với ba vector đối chiếu.
""", r'''# P2-03 · LIVE
def compute():
    from vre.rag.retrieve import cosine

    q = (3, 4)
    return {"values": [[str(v), round(cosine(q, v), 4)] for v in ((6, 8), (4, 3), (-4, 3))]}


d = live("P2-03", compute)
if d:
    table(d["values"], ["v", "cos(q, v)"])'''),
    Step("P2-04", "LIVE", h_p2_04, """
Mục tiêu: dense search top-k. Bản cũ có ngưỡng cosine cố định nhưng ngưỡng đó không cắt được gì vì điểm e5 dồn trong một khoảng hẹp,
nên đã bỏ (bug B5); số tin trả về do k quyết định. Cách đọc: điểm giảm đều khi k tăng, không có "vách" rõ để chọn ngưỡng.
""", r'''# P2-04 · LIVE
DENSE_QUESTION = "nhà mặt tiền kinh doanh gần chợ"


def compute():
    from vre.rag.retrieve import search_dense

    hits = search_dense(DENSE_QUESTION, k=8)
    return {"scores": hits["score"].round(4).tolist(), "titles": hits["title"].str.slice(0, 80).tolist()}


d = live("P2-04", compute, needs=("qdrant", "hf"))
if d:
    display(pd.DataFrame({"hạng": range(1, len(d["scores"]) + 1), "điểm": d["scores"], "tiêu đề": d["titles"]}))'''),
    Step("P2-05", "LIVE", h_p2_05, """
Mục tiêu: BM25 chấm theo từ khớp đúng (sau khi tách từ tiếng Việt, "sổ_hồng" là một từ), mạnh với ký hiệu và thuật ngữ như `2PN`;
dense chấm theo nghĩa, mạnh với cách diễn đạt khác ("gần trường học"). Cell đếm trong top-k của mỗi cách bao nhiêu tin chứa đủ hai từ
khoá của câu hỏi. Lưu ý: đếm bằng regex trên đoạn chunk, chỉ là thước đo minh hoạ.
""", r'''# P2-05 · LIVE
KEYWORD_QUESTION = "sổ hồng 2PN"


def compute():
    from vre.rag.retrieve import search_bm25, search_dense

    def both(texts):
        low = texts.str.lower()
        return int((low.str.contains("sổ hồng") & low.str.contains(r"\b2\s*pn\b|2 phòng ngủ", regex=True)).sum())

    bm, de = search_bm25(KEYWORD_QUESTION, k=S.top_k), search_dense(KEYWORD_QUESTION, k=S.top_k)
    return {"k": S.top_k, "bm25_both": both(bm["text"]), "dense_both": both(de["text"]),
            "bm25": bm["title"].str.slice(0, 80).tolist(), "dense": de["title"].str.slice(0, 80).tolist()}


d = live("P2-05", compute, needs=("qdrant", "hf"))
if d:
    display(pd.DataFrame({"BM25": d["bm25"], "dense": d["dense"]}))'''),
    Step("P2-06", "LIVE", h_p2_06, """
Mục tiêu: Reciprocal Rank Fusion cộng `1 / (k + hạng)` của mỗi tin qua các bảng xếp hạng, nên tin đứng cao ở cả hai bảng thắng mà
không cần chuẩn hoá điểm dense và BM25 về cùng thang. Cell chạy toy ba tin X, Y, Z rồi chạy câu thật bằng `search_hybrid`
(dense và BM25 đi chung một lượt gọi Qdrant).
""", r'''# P2-06 · LIVE
HYBRID_QUESTION = "căn hộ view sông có hồ bơi Thủ Đức"


def compute():
    from vre.rag.retrieve import rrf, search_bm25, search_dense, search_hybrid

    toy = rrf([["X", "Y", "Z"], ["Y", "X", "Z"], ["Y", "Z", "X"]])
    k = S.top_k
    dense = search_dense(HYBRID_QUESTION, k=20)["listing_id"].tolist()
    bm25 = search_bm25(HYBRID_QUESTION, k=20)["listing_id"].tolist()
    fused = search_hybrid(HYBRID_QUESTION, k=k)
    return {"toy": [[doc, round(score, 5)] for doc, score in toy], "overlap": len(set(dense) & set(bm25)),
            "fused": [[int(r.listing_id), round(r.rrf, 5), r.title[:80]] for r in fused.itertuples()]}


d = live("P2-06", compute, needs=("qdrant", "hf"))
if d:
    table(d["toy"], ["tin", "điểm RRF"])
    table(d["fused"], ["listing_id", "rrf", "tiêu đề"])'''),
    Step("P2-07", "LIVE", h_p2_07, """
Mục tiêu: so lọc trước (pre-filter: Qdrant chỉ tìm trong các point thoả payload) với lọc sau (post-filter: tìm top-k rồi bỏ tin sai
điều kiện). Lọc sau làm rơi gần hết top-k khi điều kiện chặt, vì tin gần nghĩa nhất thường ở quận khác hoặc vượt giá. Khi lọc trước
trả rỗng, `retrieve` bỏ lọc và báo cho LLM rằng tin chỉ gần đúng (`FILTER_DROPPED`).
""", r'''# P2-07 · LIVE
def compute():
    from vre.rag.filters import QueryFilters, post_filter
    from vre.rag.retrieve import search_hybrid

    f = QueryFilters(province="Hồ Chí Minh", district="Quận 7", property_type="Căn hộ chung cư", bedrooms=2, max_price_mil=4000)
    q, k = "căn hộ 2 phòng ngủ có hồ bơi", S.top_k
    pre = search_hybrid(q, k, f)
    post = post_filter(search_hybrid(q, k), f)
    return {"k": k, "filters": f.model_dump(exclude_none=True), "pre": len(pre), "post": len(post),
            "pre_rows": [[int(r.listing_id), r.district, r.price_vnd, r.title[:70]] for r in pre.itertuples()]}


d = live("P2-07", compute, needs=("qdrant", "hf"))
if d:
    md(f"Bộ lọc: `{json.dumps(d['filters'], ensure_ascii=False)}`")
    table(d["pre_rows"], ["listing_id", "quận", "giá", "tiêu đề"])'''),
    Step("P2-08", "CACHE", h_p2_08, """
Mục tiêu: rerank cross-encoder chấm lại từng cặp (câu hỏi, chunk) — chậm hơn bi-encoder nhưng chính xác hơn — trên một tập ứng viên
rồi giữ top-k. Notebook không nạp cross-encoder (thêm một model lớn vào RAM); tác dụng của rerank được đo ở ablation trên subcorpus
chứa mọi tin đáp án. Mặc định rerank tắt trong chatbot vì tăng độ trễ.
""", r'''# P2-08 · CACHE
ablation = load("ablation.json")
if ablation is None:
    missing("`results/ablation.json`", "python scripts/vre_pipeline.py ablation")
else:
    display(pd.DataFrame(ablation["runs"]))'''),
    Step("P2-09", "LIVE", h_p2_09, """
Mục tiêu: ghép prompt gồm chỉ thị, ngữ cảnh và câu hỏi. Mỗi tài liệu được bọc trong `<tai_lieu>` và thẻ giả mạo trong nội dung bị vô
hiệu hoá, vì tin đăng là văn bản người dùng tự nhập (chống prompt injection). Câu trả lời lấy từ lượt eval đã chạy
(`results/eval/raw/answers.jsonl`), không gọi LLM lại; `check_citations` tách trích dẫn có trong nguồn và trích dẫn bịa.
""", r'''# P2-09 · LIVE
def compute():
    from vre.eval.questions import load_questions
    from vre.rag.citations import check_citations
    from vre.rag.generate import build_prompt

    questions = {q.id: q.question for q in load_questions()}
    row = next(r for r in jsonl("eval/raw/answers.jsonl") if r["system"] == "basic_rag" and r["sources"])
    cited, unsupported = check_citations(row["answer"], row["sources"])
    prompt = build_prompt(questions[row["id"]], row["contexts"][:2])
    return {"id": row["id"], "question": questions[row["id"]], "answer": row["answer"], "sources": row["sources"],
            "cited": cited, "unsupported": unsupported, "prompt_head": prompt[:900]}


d = live("P2-09", compute, needs=("answers",))
if d:
    md(f"**Câu hỏi `{d['id']}`:** {d['question']}")
    md("**Đầu prompt (hai tài liệu đầu):**\n\n```text\n" + d["prompt_head"] + "\n```")
    md(f"**Câu trả lời:**\n\n{d['answer']}")
    md(f"Nguồn đưa vào ngữ cảnh: `{d['sources']}`; trích dẫn hợp lệ `{d['cited']}`; không có trong nguồn `{d['unsupported']}`.")'''),
    Step("P2-10", "ẨN", "Đóng gói thành một hàm `vre.qa.answer` cho API và eval", """
Mục tiêu: một facade duy nhất cho bốn hệ thống (`llm_only`, `basic_rag`, `graph`, `hybrid`). API stream từng sự kiện qua
`stream_answer`, eval gọi bản không stream `answer` và ghi đúng các khoá của `answers.jsonl`. Cell chỉ in chữ ký và docstring.
""", r'''# P2-10 · ẨN
import inspect

from vre import qa

md(f"`qa.answer{inspect.signature(qa.answer)}`\n\n{inspect.getdoc(qa.answer)}")'''),
    Step("P2-11", "CACHE", h_p2_11, """
Mục tiêu: Basic RAG theo nhóm câu hỏi. RAG tìm đúng tin cụ thể (tra cứu, ràng buộc) nhưng không thể đếm hay lấy trung vị trên hàng
triệu tin chỉ từ vài chunk — câu tổng hợp là chỗ cần đồ thị (notebook `03`, `04`). Đọc `eval_summary.json` khi eval đã xong; trước
đó chỉ có bảng tạm từ các dòng đã chấm, cỡ mẫu nhỏ nên chưa kết luận được.
""", r'''# P2-11 · CACHE
summary = load("eval_summary.json")
if summary is not None:
    acc = pd.DataFrame({cat: {s: v["accuracy"] for s, v in by.items()} for cat, by in summary["by_category"].items()}).T
    display(acc)
else:
    def compute():
        scores = pd.DataFrame(jsonl("eval/raw/scores.jsonl"))
        tab = scores.pivot_table(index="category", columns="system", values="correct", aggfunc="mean")
        return {"n_rows": len(scores), "table": tab.round(3).reset_index().to_dict("records")}

    d = cached("P2-11", compute, needs=("answers",))
    if d:
        md(f"Eval chưa xong: bảng tạm từ {num(d['n_rows'])} dòng chấm (tỷ lệ đúng).")
        table(d["table"])'''),
]


# =============================================================================================== 03_knowledge_graph
def h_p3_02(c: Ctx) -> str:
    d = c.c("P3-02")
    if d is None:
        return "Một tin đăng là một chùm node và cạnh (chưa chạy demo)"
    return f"Một tin đăng nối tới {N(len(d['triples']))} node qua {N(len({t[1] for t in d['triples']}))} kiểu cạnh"


def h_p3_03(c: Ctx) -> str:
    g = c.r("graph_report.json")
    if g is None:
        return "Schema đồ thị (chưa có graph_report)"
    return f"Schema {N(len(g['nodes']))} nhãn node và {N(len(g['relationships']))} kiểu cạnh, mỗi nhãn có khoá duy nhất"


def h_p3_04(c: Ctx) -> str:
    d = c.c("P3-04")
    if d is None:
        return "Chuẩn hoá địa danh trước khi nạp (chưa chạy demo)"
    raw, _, norm = d["examples"][0]
    return f"Entity resolution: '{raw}' thành '{norm}'; khoá phường gồm cả quận và tỉnh"


def h_p3_05(c: Ctx) -> str:
    g = c.r("graph_report.json")
    if g is None:
        return "Nạp đồ thị bằng neo4j-admin import (chưa có graph_report)"
    return f"Nạp toàn bộ đồ thị bằng `neo4j-admin import` trong {D(g['import_seconds'], 1)} giây"


def h_p3_06(c: Ctx) -> str:
    g = c.r("graph_report.json")
    if g is None:
        return "Quy mô đồ thị (chưa có graph_report)"
    return f"Đồ thị có {N(sum(g['nodes'].values()))} node và {N(sum(g['relationships'].values()))} cạnh"


def h_p3_07(c: Ctx) -> str:
    g = c.r("graph_report.json")
    if g is None:
        return "Kiểm tra nhất quán đồ thị (chưa có graph_report)"
    inv = g["invariants"]
    ok = inv["listing_count_ok"] and inv["price_sum_ok"]
    return "Số Listing và tổng giá trong đồ thị khớp bảng sạch" if ok else "Đồ thị KHÔNG khớp bảng sạch — xem bất biến"


def h_p3_08(c: Ctx) -> str:
    d = c.c("P3-08")
    if d is None:
        return "Tiện ích trích bằng regex thành node Amenity (chưa có số đếm)"
    top = d["rows"][0]
    return f"{N(len(d['rows']))} tiện ích trích bằng regex; phổ biến nhất là '{top[0]}' với {N(top[1])} tin"


def h_p3_09(c: Ctx) -> str:
    d = c.c("P3-09")
    if d is None:
        return "Cypher tra cứu đọc như câu hỏi (chưa chạy demo)"
    return f"Cypher tra cứu trả {N(len(d['rows']))} tin trong {N(d['ms'])} ms, tương đương một câu SQL có JOIN"


def h_p3_10(c: Ctx) -> str:
    d = c.c("P3-10")
    if d is None:
        return "Cypher tổng hợp khớp DuckDB (chưa chạy demo)"
    return (f"Quận có giá/m² trung vị thấp nhất ở Hồ Chí Minh là {d['district']}: Cypher {D(d['cypher'], 2)} = DuckDB "
            f"{D(d['duckdb'], 2)} triệu/m²")


def h_p3_11(c: Ctx) -> str:
    d = c.c("P3-11")
    if d is None:
        return "Truy vấn đa chặng Tin → Phường → Quận → Tỉnh (chưa chạy demo)"
    top = d["rows"][0]
    return f"Đi bốn chặng Tin → Phường → Quận → Tỉnh: ở {d['province']}, {top[0]} dẫn đầu với {N(top[1])} tin"


def h_p3_12(c: Ctx) -> str:
    d, g = c.c("P3-12"), c.r("graph_report.json")
    if d is None or g is None:
        return "Modularity và cộng đồng Leiden (chưa chạy demo)"
    cm = g["communities"]
    return f"Modularity toy Q = {D(d['q'], 3)}; Leiden gom {N(cm['n_wards'])} phường thành {N(cm['n_clusters'])} cụm, Q = {D(cm['modularity'], 3)}"


def h_p3_13(c: Ctx) -> str:
    d = c.c("P3-13")
    if d is None:
        return "Một quận nhìn như chùm sao quanh các phường (chưa chạy demo)"
    return f"{d['district']} nhìn như chùm sao: {N(len(d['wards']))} phường quanh một node quận"


NB03_TITLE = """# 03 · Knowledge Graph

Đồ thị tri thức Neo4j dựng từ bảng sạch: schema, entity resolution, nạp hàng loạt, kiểm tra nhất quán, truy vấn Cypher (tra cứu,
tổng hợp, đa chặng) và cộng đồng Leiden của các phường.

**Đầu ra cho notebook sau:** đồ thị Neo4j, `vre.graph.client.run_cypher`, `vre.graph.text2cypher`.
**Lệnh sinh:** `python scripts/vre_pipeline.py graph`. Cell LIVE cần Neo4j đang chạy, không có thì đọc cache cell."""

KG_MERMAID = """
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
  MarketStat -->|STAT_OF| District
```
"""

NB03 = [
    Step("P3-01", "ẨN", "Kết nối Neo4j; đồ thị dựng từ cùng bảng sạch với chỉ mục vector", """
Mục tiêu: kiểm tra Neo4j có trả lời không. Mọi truy vấn của notebook chạy trong giao dịch CHỈ ĐỌC có timeout — cùng hàm
`run_cypher` mà Text2Cypher dùng — nên notebook không thể sửa đồ thị.
""", r'''# P3-01 · ẨN
md("Neo4j: " + ("đang chạy, các cell LIVE chạy thật." if available("neo4j") else "không có trên máy này, các cell LIVE đọc cache cell."))'''),
    Step("P3-02", "LIVE", h_p3_02, """
Mục tiêu: nhìn đồ thị ở quy mô một tin. Tin là node `Listing`; thuộc tính số (giá, diện tích, số phòng) nằm trên node, còn thực thể
dùng chung (loại hình, phường, đường, dự án, tháng, tiện ích) là node riêng để nhiều tin nối vào. Cách đọc: mỗi dòng là một triple
(tin, kiểu cạnh, node đích).
""", r'''# P3-02 · LIVE
def compute():
    from vre.graph.client import run_cypher

    lid = int(run_cypher("MATCH (l:Listing)-[:IN_PROJECT]->() WHERE l.bedrooms IS NOT NULL RETURN l.listing_id AS id LIMIT 1", timeout=10)["id"].iloc[0])
    rows = run_cypher("MATCH (l:Listing {listing_id: $id})-[r]->(n) RETURN l.title AS title, type(r) AS rel, labels(n)[0] AS label, "
                      "coalesce(n.name, n.key) AS name", {"id": lid}, timeout=10)
    return {"listing_id": lid, "title": rows["title"].iloc[0], "triples": rows[["label", "rel", "name"]].values.tolist()}


d = live("P3-02", compute, needs=("neo4j",))
if d:
    md(f"Tin `{d['listing_id']}`: *{d['title']}*")
    table([[f"Listing {d['listing_id']}", rel, f"{label}: {name}"] for label, rel, name in d["triples"]], ["từ", "cạnh", "tới"])'''),
    Step("P3-03", "CACHE", h_p3_03, """
Mục tiêu: schema của đồ thị. Tin gắn vào cấp hành chính chi tiết nhất có được (phường, thiếu thì quận, thiếu nữa thì tỉnh).
`MarketStat` là thống kê giá tính sẵn theo vùng × loại × tháng để Cypher tổng hợp chạy mili-giây thay vì quét mọi tin; `Cluster`
là cộng đồng Leiden của phường. Ràng buộc khoá duy nhất và index nằm trong `graph/post_import.cypher`.
""" + KG_MERMAID, r'''# P3-03 · CACHE
gr = load("graph_report.json")
post = (CODE_ROOT / "graph" / "post_import.cypher").read_text(encoding="utf-8")
md(f"`graph/post_import.cypher`: {num(len(re.findall(r'CREATE CONSTRAINT', post)))} ràng buộc khoá duy nhất, "
   f"{num(len(re.findall(r'CREATE (?:RANGE |FULLTEXT )?INDEX', post)))} index.")
if gr is not None:
    display(pd.DataFrame({"node": pd.Series(gr["nodes"])}).sort_values("node", ascending=False))'''),
    Step("P3-04", "LIVE", h_p3_04, """
Mục tiêu: entity resolution trước khi nạp. Cùng một quận có thể viết `Q.7`, `q7`, `Quận 7`; nếu không chuẩn hoá sẽ thành ba node.
Khoá node gồm cả cấp cha (`tỉnh|quận|phường`) vì nhiều tỉnh có `Phường 1` — khoá chỉ theo tên sẽ gộp nhầm phường của các thành phố.
""", r'''# P3-04 · LIVE
def compute():
    from vre.data.places import normalize_place

    raw = [("Q.7", "district"), ("TP. Hồ Chí Minh", "province"), ("P.5", "ward"), ("Huyện Củ Chi", "district")]
    ex = [[r, k, normalize_place(r, k)] for r, k in raw]
    key = "|".join([normalize_place("TP. Hồ Chí Minh", "province"), normalize_place("Q.7", "district"), normalize_place("P.5", "ward")])
    return {"examples": ex, "ward_key": key}


d = live("P3-04", compute)
if d:
    table(d["examples"], ["gốc", "loại", "chuẩn hoá"])
    md(f"Khoá node Ward: `{d['ward_key']}`")'''),
    Step("P3-05", "CACHE", h_p3_05, """
Mục tiêu: nạp hàng loạt. Với hàng triệu tin, `MERGE` từng dòng mất hàng giờ; pipeline xuất CSV node/cạnh bằng DuckDB rồi dùng
`neo4j-admin database import full` (ghi thẳng file store, offline), sau đó chạy `post_import.cypher` để tạo ràng buộc và index.
Hệ quả: nạp lại là thay cả database, không cập nhật từng phần.
""", r'''# P3-05 · CACHE
if gr is None:
    missing("`results/graph_report.json`", "python scripts/vre_pipeline.py graph")
else:
    md(f"Nạp trong {dec(gr['import_seconds'], 1)} giây; sinh lúc `{gr['generated_at']}`.")'''),
    Step("P3-06", "CACHE", h_p3_06, """
Mục tiêu: quy mô đồ thị theo nhãn và kiểu cạnh. Cách đọc: `IN_WARD` nhiều hơn `IN_DISTRICT` vì cạnh tới quận chỉ có khi tin thiếu
phường; `HAS_AMENITY` vượt số tin vì một tin có nhiều tiện ích. Thang log vì các nhãn chênh nhau nhiều bậc.
""", r'''# P3-06 · CACHE
if gr is not None:
    rels = pd.Series(gr["relationships"]).sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    pd.Series(gr["nodes"]).sort_values(ascending=False).plot.barh(ax=axes[0], logx=True, title="node theo nhãn")
    rels.plot.barh(ax=axes[1], logx=True, title="cạnh theo kiểu")
    for ax in axes:
        ax.invert_yaxis()
    fig.tight_layout()
    plt.show()
    display(rels.to_frame("cạnh"))'''),
    Step("P3-07", "CACHE", h_p3_07, """
Mục tiêu: kiểm tra nhất quán sau import. Hai bất biến: số node `Listing` bằng số tin của bảng sạch và tổng `price_vnd` khớp (sai số
tương đối rất nhỏ). Lần import cũ trên ổ NTFS từng thiếu node mà không báo lỗi; bất biến này bắt được điều đó, và API/QA từ chối
dùng đồ thị lệch manifest chỉ mục.
""", r'''# P3-07 · CACHE
if gr is not None:
    display(pd.Series(gr["invariants"], name="đạt").to_frame())'''),
    Step("P3-08", "CACHE", h_p3_08, """
Mục tiêu: tiện ích (hồ bơi, thang máy, hẻm xe hơi, sổ hồng…) được trích từ tiêu đề và mô tả bằng bảng regex cố định, thành node
`Amenity` để Cypher lọc được. Bản cũ dùng LLM trên vài chục tin; regex chạy được trên mọi tin, không tốn token, đổi lại bỏ sót
cách viết lạ. Số đếm dùng degree của node nên không quét cạnh.
""", r'''# P3-08 · CACHE
def compute():
    from vre.graph.client import run_cypher

    rows = run_cypher("MATCH (a:Amenity) RETURN a.name AS amenity, COUNT { (a)<-[:HAS_AMENITY]-() } AS n ORDER BY n DESC", timeout=30)
    return {"rows": rows.values.tolist()}


d = cached("P3-08", compute, needs=("neo4j",))
if d:
    table(d["rows"], ["tiện ích", "số tin"])'''),
    Step("P3-09", "LIVE", h_p3_09, """
Mục tiêu: Cypher tra cứu: căn hộ hai phòng ngủ ở `Quận 7` dưới mức giá. Mẫu `(tin)-[:IN_WARD]->(phường)-[:IN_DISTRICT]->(quận)` đọc như
câu hỏi; SQL tương đương cần JOIN ba bảng. Cách đọc: thời gian đo phía client, gồm cả mạng và lần đầu Neo4j nạp trang dữ liệu vào
bộ nhớ, nên lần chạy đầu chậm hơn hẳn các lần sau.
""", r'''# P3-09 · LIVE
LOOKUP = """MATCH (l:Listing)-[:IN_WARD]->(:Ward)-[:IN_DISTRICT]->(:District {key: 'Hồ Chí Minh|Quận 7'})
WHERE l.property_type = 'Căn hộ chung cư' AND l.bedrooms = 2 AND l.price_vnd < 4e9 AND l.deal_type = 'sale'
RETURN l.listing_id AS listing_id, l.title AS title, l.price_vnd AS price_vnd, l.area_m2 AS area_m2
ORDER BY l.price_vnd DESC LIMIT 10"""
SQL = """SELECT listing_id, title, price_vnd, area_m2 FROM v_listings
WHERE province = 'Hồ Chí Minh' AND district = 'Quận 7' AND property_type = 'Căn hộ chung cư'
  AND bedrooms = 2 AND price_vnd < 4e9 AND deal_type = 'sale' ORDER BY price_vnd DESC LIMIT 10"""


def compute():
    from vre.graph.client import run_cypher

    t0 = time.perf_counter()
    rows = run_cypher(LOOKUP, timeout=20)
    return {"ms": (time.perf_counter() - t0) * 1000, "rows": rows.values.tolist(), "columns": list(rows.columns)}


md("```cypher\n" + LOOKUP + "\n```\n\nSQL tương đương:\n\n```sql\n" + SQL + "\n```")
d = live("P3-09", compute, needs=("neo4j",))
if d:
    table(d["rows"], d["columns"])'''),
    Step("P3-10", "LIVE", h_p3_10, """
Mục tiêu: Cypher tổng hợp: quận có giá/m² trung vị thấp nhất ở Hồ Chí Minh, đọc từ `MarketStat` (tính sẵn bằng DuckDB lúc import).
Cell tính lại trung vị của quận đó bằng DuckDB trên bảng sạch để chứng minh hai đường cho cùng số. Chỉ xét quận đủ nhiều tin vì
trung vị trên vài tin không ổn định.
""", r'''# P3-10 · LIVE
AGG = """MATCH (m:MarketStat {scope: 'district', property_type: 'ALL', month: 'ALL', deal_type: 'sale'})-[:STAT_OF]->(d:District)
      -[:IN_PROVINCE]->(:Province {key: 'Hồ Chí Minh'})
WHERE m.n >= 1000
RETURN d.name AS district, m.median_price_m2_mil AS median_price_m2_mil, m.n AS n ORDER BY median_price_m2_mil ASC LIMIT 5"""


def compute():
    from vre.data.duck import connect
    from vre.graph.client import run_cypher

    rows = run_cypher(AGG, timeout=20)
    top = rows.iloc[0]
    con = connect(S)
    val = con.execute("SELECT median(price_m2_mil) FROM v_listings WHERE deal_type = 'sale' AND province = 'Hồ Chí Minh' "
                      "AND district = ? AND price_m2_mil IS NOT NULL", [top["district"]]).fetchone()[0]
    con.close()
    return {"rows": rows.values.tolist(), "columns": list(rows.columns), "district": top["district"],
            "cypher": float(top["median_price_m2_mil"]), "duckdb": float(val)}


d = live("P3-10", compute, needs=("neo4j", "data"))
if d:
    table(d["rows"], d["columns"])
    md(f"Kiểm chéo: Cypher {dec(d['cypher'], 4)} so với DuckDB {dec(d['duckdb'], 4)} triệu/m² — "
       + ("khớp." if abs(d["cypher"] - d["duckdb"]) < 1e-6 * max(1, abs(d["duckdb"])) else "LỆCH."))'''),
    Step("P3-11", "LIVE", h_p3_11, """
Mục tiêu: truy vấn đa chặng đi theo đường Tin → Phường → Quận → Tỉnh để đếm tin theo quận của một tỉnh. Trong SQL đây là chuỗi JOIN;
trong Cypher là một mẫu đường. Cách đọc: chỉ đếm tin có phường; tin thiếu phường nối thẳng vào quận nên không nằm trong đếm này.
""", r'''# P3-11 · LIVE
PROVINCE = "Đà Nẵng"
MULTIHOP = """MATCH (l:Listing)-[:IN_WARD]->(:Ward)-[:IN_DISTRICT]->(d:District)-[:IN_PROVINCE]->(:Province {key: $p})
RETURN d.name AS district, count(l) AS n ORDER BY n DESC"""


def compute():
    from vre.graph.client import run_cypher

    return {"province": PROVINCE, "rows": run_cypher(MULTIHOP, {"p": PROVINCE}, timeout=60).values.tolist()}


d = live("P3-11", compute, needs=("neo4j",))
if d:
    table(d["rows"], ["quận", "số tin"])'''),
    Step("P3-12", "LIVE", h_p3_12, """
Mục tiêu: modularity `Q = Σ [L_c / m − (d_c / 2m)²]` đo mức cạnh nằm trong cộng đồng nhiều hơn ngẫu nhiên. Toy: hai tam giác nối nhau
bằng một cạnh. Trên dữ liệu thật, pipeline dựng đồ thị kNN giữa các phường theo hồ sơ giá rồi chạy Leiden; mỗi cụm là một "khu vực
có mặt bằng giá giống nhau" dùng cho câu hỏi "khu vực tương tự".
""", r'''# P3-12 · LIVE
def compute():
    from vre.graph.communities import modularity

    edges = [(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6), (3, 4)]
    return {"q": modularity(edges, [{1, 2, 3}, {4, 5, 6}]), "edges": edges}


d = live("P3-12", compute)
if d and gr is not None:
    md(f"Toy: Q = {dec(d['q'], 4)}. Leiden thật: `{json.dumps(gr['communities'], ensure_ascii=False)}`")'''),
    Step("P3-13", "LIVE", h_p3_13, """
Mục tiêu: trực quan hoá một quận: node quận ở giữa, các phường xung quanh; kích thước theo số tin, màu theo giá/m² trung vị. Cách đọc:
phường đậm màu là nơi đắt hơn trong cùng quận. Vẽ bằng matplotlib từ kết quả Cypher (không cần thư viện đồ thị riêng).
""", r'''# P3-13 · LIVE
DISTRICT_KEY = "Hồ Chí Minh|Quận 7"


def compute():
    from vre.graph.client import run_cypher

    rows = run_cypher("MATCH (w:Ward)-[:IN_DISTRICT]->(d:District {key: $k}) RETURN d.name AS district, w.name AS ward, "
                      "w.n_listings AS n, w.median_price_m2_mil AS price ORDER BY ward", {"k": DISTRICT_KEY}, timeout=20)
    return {"district": rows["district"].iloc[0], "wards": rows[["ward", "n", "price"]].values.tolist()}


d = live("P3-13", compute, needs=("neo4j",))
if d:
    w = pd.DataFrame(d["wards"], columns=["ward", "n", "price"])
    angles = np.linspace(0, 2 * np.pi, len(w), endpoint=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    for a in angles:
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color="#ccc", zorder=1)
    sc = ax.scatter(np.cos(angles), np.sin(angles), s=20 + 400 * w["n"] / w["n"].max(), c=w["price"], cmap="viridis", zorder=2)
    for a, name in zip(angles, w["ward"], strict=True):
        ax.annotate(name, (1.12 * np.cos(a), 1.12 * np.sin(a)), ha="center", fontsize=7)
    ax.scatter([0], [0], s=500, color="#d62728", zorder=3)
    ax.annotate(d["district"], (0, -0.15), ha="center")
    fig.colorbar(sc, label="giá/m² trung vị (triệu)")
    ax.set_axis_off()
    plt.show()'''),
]


# =============================================================================================== 04_graphrag_eval
def _questions() -> list[dict]:
    path = CODE_ROOT / "benchmark" / "questions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def h_p4_01(c: Ctx) -> str:
    qs = _questions()
    if not qs:
        return "Bốn hệ thống chạy trên cùng dữ liệu và cùng bộ câu hỏi"
    return f"Bốn hệ thống, gồm baseline LLM only, chạy trên cùng {N(len(qs))} câu hỏi thuộc {N(len({q['category'] for q in qs}))} nhóm"


def h_p4_02(c: Ctx) -> str:
    g = c.r("benchmark_gt.json")
    if g is None:
        return "Đáp án chuẩn tính bằng SQL, không gõ tay (chưa có benchmark_gt)"
    return (f"Đáp án chuẩn của {N(len(g['items']))} câu tính bằng SQL DuckDB trên toàn bộ bảng sạch trong "
            f"{D(sum(i['sql_seconds'] for i in g['items']), 1)} giây")


def h_p4_03(c: Ctx) -> str:
    d = c.c("P4-03")
    if d is None or d.get("t2c") is None:
        return "Text2Cypher sinh truy vấn chỉ đọc, lệnh ghi bị chặn hai lớp"
    t = d["t2c"]
    return f"Text2Cypher sinh truy vấn chỉ đọc ({N(t['attempts'])} lượt, {N(len(t['rows']))} dòng); lệnh ghi bị chặn hai lớp"


def h_p4_04(c: Ctx) -> str:
    d = c.c("P4-04")
    if d is None:
        return "Local retrieval mở rộng láng giềng quanh một thực thể (chưa chạy demo)"
    return f"Local retrieval: phường {d['ward']} có {N(len(d['neighbors']))} phường cùng cụm giá"


def h_p4_06(c: Ctx) -> str:
    e = c.r("eval_summary.json")
    if e is not None:
        return f"Router chọn đúng nguồn cho {P(e['router']['accuracy'])} câu hỏi (n = {N(e['n_questions'])})"
    d = c.c("P4-06")
    if d is None or not d["n"]:
        return "Router chọn nguồn trả lời (chưa có quyết định router trong cache)"
    return f"Router chọn đúng nguồn cho {N(d['correct'])}/{N(d['n'])} câu đã có trong cache LLM"


def h_p4_08(c: Ctx) -> str:
    d = c.c("P4-08")
    if d is None:
        return "Recall@k và MRR (chưa chạy toy)"
    e = c.r("eval_summary.json")
    real = (f"; thật (Basic RAG): Recall@5 = {D(e['retrieval']['basic_rag']['recall_at_5'], 2)}" if e and "basic_rag" in e["retrieval"]
            else "; số thật chờ eval")
    return f"Toy: Recall@5 = {D(d['recall'], 2)}, MRR = {D(d['mrr'], 2)}{real}"


def h_p4_09(c: Ctx) -> str:
    d = c.c("P4-09")
    if d is None:
        return "Faithfulness và các chỉ số kiểu RAGAS (chưa chạy toy)"
    e = c.r("eval_summary.json")
    real = (f"; thật (Basic RAG) = {D(e['ragas']['basic_rag']['faithfulness'], 2)}" if e and "basic_rag" in e["ragas"]
            else "; số thật chờ eval")
    return f"Faithfulness toy = {D(d['faithfulness'], 2)}{real}"


def h_p4_10(c: Ctx) -> str:
    e = c.r("eval_summary.json")
    if e is not None:
        total = len(_questions())
        scope = "toàn bộ benchmark" if e["n_questions"] >= total else f"mới {N(e['n_questions'])}/{N(total)} câu (chưa đủ benchmark)"
        return f"Benchmark chạy {N(len(e['systems']))} hệ thống trên {scope}"
    d = c.c("P4-10")
    if d is None:
        return "Benchmark bốn hệ thống (chưa chạy eval)"
    return f"Benchmark đang chạy: {N(d['n_rows'])} câu trả lời của {N(d['n_questions'])} câu hỏi (eval chưa xong)"


def h_p4_11(c: Ctx) -> str:
    e = c.r("eval_summary.json")
    if e is None:
        return "Độ chính xác theo nhóm câu hỏi và hệ thống (chưa có eval_summary)"
    best = max(e["overall"], key=lambda s: e["overall"][s]["accuracy"])
    v = e["overall"][best]
    return f"`{best}` đúng nhiều nhất: {P(v['accuracy'])} (CI95 {P(v['ci95'][0])} – {P(v['ci95'][1])}, n = {N(e['n_questions'])})"


def h_p4_12(c: Ctx) -> str:
    e = c.r("eval_summary.json")
    if e is None:
        return "Chi phí và độ trễ theo hệ thống (chưa có eval_summary)"
    co = e["cost"]
    if "hybrid" not in co or "basic_rag" not in co:
        return "Chi phí và độ trễ theo hệ thống"
    return (f"Hybrid tốn {N(co['hybrid']['input_tokens_mean'] - co['basic_rag']['input_tokens_mean'])} token vào và "
            f"{D((co['hybrid']['latency_ms_p50'] - co['basic_rag']['latency_ms_p50']) / 1000, 1)} giây (p50) mỗi câu so với Basic RAG")


def h_p4_13(c: Ctx) -> str:
    e = c.r("error_analysis.json")
    if e is None:
        return "Phân tích lỗi theo nguyên nhân (chưa có error_analysis)"
    total: dict[str, int] = {}
    for by in e["counts"].values():
        for cause, n in by.items():
            total[cause] = total.get(cause, 0) + n
    if not total:
        return "Không có câu sai nào để phân tích"
    top = max(total, key=total.get)
    return f"Nguyên nhân lỗi nhiều nhất là `{top}` ({N(total[top])} lần trên mọi hệ)"


def h_p4_14(c: Ctx) -> str:
    e = c.r("eval_summary.json")
    if e is None:
        return "So sánh cặp bằng bootstrap ghép cặp (chưa có eval_summary)"
    pw = e["pairwise"]
    sig = [p for p in pw if p["ci95"][0] > 0 or p["ci95"][1] < 0]
    return f"{N(len(sig))}/{N(len(pw))} so sánh cặp có CI95 của hiệu không chứa 0"


NB04_TITLE = """# 04 · GraphRAG, Hybrid và Đánh giá

Ghép Basic RAG với đồ thị bằng Text2Cypher và router, đo bằng Recall@k, MRR, nDCG, chỉ số kiểu RAGAS, rồi so bốn hệ thống
(LLM only, Basic RAG, Graph, Hybrid) trên cùng benchmark, có khoảng tin cậy bootstrap.

**Lệnh sinh:** `python scripts/vre_pipeline.py eval` → `results/eval_summary.json`, `results/error_analysis.json`. Khi eval chưa
xong, các cell kết quả hiện "chưa có" hoặc bảng tạm và được chạy lại sau eval."""

NB04 = [
    Step("P4-01", "ẨN", h_p4_01, """
Mục tiêu: nạp benchmark `benchmark/questions.jsonl` (SSOT câu hỏi, code không bao giờ ghi đè) và bốn hệ thống qua facade `vre.qa`.
LLM only dùng cùng mô hình nhưng không có ngữ cảnh — baseline để đo truy xuất thêm được gì. Mỗi câu có tuyến đúng cho router.
""", r'''# P4-01 · ẨN
from vre.eval.questions import load_questions

QUESTIONS = load_questions()
by_cat = pd.Series([q.category for q in QUESTIONS]).value_counts()
display(by_cat.to_frame("số câu"))'''),
    Step("P4-02", "CACHE", h_p4_02, """
Mục tiêu: đáp án chuẩn không gõ tay. Mỗi câu có một câu SQL trên view `v_listings` của toàn bộ bảng sạch; kết quả (id tin, giá trị,
hoặc từ chối cho câu không có đáp án) ghi vào `results/benchmark_gt.json` kèm dấu vân tay dữ liệu. Tin gần trùng được mở rộng theo
`dup_cluster_id` để hệ trả bản sao không bị chấm sai.
""", r'''# P4-02 · CACHE
gt = load("benchmark_gt.json")
if gt is None:
    missing("`results/benchmark_gt.json`", "python scripts/vre_pipeline.py eval gt")
else:
    q = next(x for x in QUESTIONS if x.gt_sql)
    item = next(i for i in gt["items"] if i["id"] == q.id)
    md(f"Ví dụ `{q.id}` ({q.category}): *{q.question}*\n\n```sql\n{q.gt_sql}\n```\n\nĐáp án: "
       f"`{item['gt_value'] if item['gt_value'] is not None else item['gt_ids']}`")
    display(pd.Series(gt["data_fingerprint"], name="fingerprint").to_frame())'''),
    Step("P4-03", "LIVE", h_p4_03, """
Mục tiêu: Text2Cypher: prompt chứa schema đồ thị và ví dụ; LLM sinh Cypher; lớp một là regex chặn lệnh ghi/quản trị, thêm `LIMIT`;
lớp hai là giao dịch chỉ đọc của Neo4j có timeout; lỗi thì gửi lỗi lại cho LLM để sửa. Lời gọi LLM chỉ đọc cache: câu chưa có
trong cache thì cell chỉ chạy phần chặn lệnh ghi.
""", r'''# P4-03 · LIVE
def compute():
    from vre.graph.guard import ensure_limit, is_read_only

    guard = [[q, is_read_only(q)] for q in ("MATCH (n:District) RETURN n.name", "MATCH (n) DETACH DELETE n", "CALL db.labels()")]
    out = {"guard": guard, "limit": ensure_limit("MATCH (n:District) RETURN n.name"), "t2c": None}
    graph_q = next((x for x in QUESTIONS if x.route_gold == "graph"), None)
    if graph_q is not None and available("neo4j"):
        from vre.graph.text2cypher import cypher_result, text2cypher
        try:
            res = cypher_result(text2cypher(graph_q.question))
            out["t2c"] = {"question": graph_q.question, **res}
        except LLMCacheMiss:
            pass
    return out


d = live("P4-03", compute)
if d:
    table(d["guard"], ["câu lệnh", "chỉ đọc?"])
    md(f"`ensure_limit` thêm giới hạn: `{d['limit']}`")
    t = d["t2c"]
    if t is None:
        md("Câu Text2Cypher mẫu chưa có trong cache LLM hoặc Neo4j không chạy: bỏ qua phần sinh Cypher.")
    else:
        md(f"*{t['question']}*\n\n```cypher\n{t['cypher']}\n```\n\nLỗi: `{t['error']}`")
        table(t["rows"], t["columns"])'''),
    Step("P4-04", "LIVE", h_p4_04, """
Mục tiêu: local retrieval kiểu GraphRAG: từ một thực thể (một phường), mở rộng láng giềng trên đồ thị — ở đây là các phường cùng cụm
Leiden — để trả lời "khu vực nào tương tự". Cách đọc: phường cùng cụm có hồ sơ giá giống nhau, không nhất thiết liền kề địa lý.
""", r'''# P4-04 · LIVE
def compute():
    from vre.graph.client import run_cypher

    w = run_cypher("MATCH (w:Ward)-[:IN_DISTRICT]->(:District {key: 'Hồ Chí Minh|Quận 7'}) MATCH (w)-[:IN_CLUSTER]->(c:Cluster) "
                   "RETURN w.key AS key, w.name AS name, c.cluster_id AS cid, c.label AS label ORDER BY w.n_listings DESC LIMIT 1", timeout=10).iloc[0]
    nb = run_cypher("MATCH (c:Cluster {cluster_id: $cid})<-[:IN_CLUSTER]-(o:Ward)-[:IN_DISTRICT]->(d:District) WHERE o.key <> $key "
                    "RETURN o.name AS ward, d.key AS district, o.median_price_m2_mil AS price, o.n_listings AS n "
                    "ORDER BY n DESC LIMIT 10", {"cid": int(w["cid"]), "key": w["key"]}, timeout=10)
    return {"ward": w["name"], "cluster": w["label"], "neighbors": nb.values.tolist()}


d = live("P4-04", compute, needs=("neo4j",))
if d:
    md(f"Phường {d['ward']}, cụm *{d['cluster']}* — mười láng giềng nhiều tin nhất:")
    table(d["neighbors"], ["phường", "quận", "giá/m² trung vị", "số tin"])'''),
    Step("P4-05", "ẨN", "Global search chỉ giải thích, không chạy: câu tổng hợp đã có Cypher chính xác", """
Mục tiêu: global search của GraphRAG tóm tắt từng cộng đồng bằng LLM rồi trả lời câu hỏi toàn cục từ các bản tóm tắt. Với dữ liệu
có cấu trúc như ở đây, câu toàn cục (trung vị, xếp hạng, xu hướng) được Cypher trên `MarketStat` trả chính xác và rẻ hơn nhiều, nên
dự án không dựng bản tóm tắt cộng đồng; cụm Leiden chỉ dùng cho local retrieval.
""", r'''# P4-05 · ẨN
gr = load("graph_report.json")
if gr is not None:
    md(f"Cộng đồng có sẵn: `{json.dumps(gr['communities'], ensure_ascii=False)}`")'''),
    Step("P4-06", "LIVE", h_p4_06, """
Mục tiêu: router phân loại câu hỏi thành `basic_rag`, `graph`, `hybrid` hoặc `valuation` bằng structured output và vài ví dụ mẫu.
Cell chạy router trên mọi câu benchmark nhưng chỉ lấy câu đã có trong cache LLM (từ lượt eval), rồi so với tuyến đúng. Cỡ mẫu nhỏ thì
tỷ lệ chưa nói được nhiều; số chính thức là `eval_summary.router`.
""", r'''# P4-06 · LIVE
def compute():
    from vre.hybrid.router import router

    rows = []
    for q in QUESTIONS:
        try:
            rows.append([q.id, q.route_gold, router(q.question)["route"]])
        except LLMCacheMiss:
            continue
    return {"n": len(rows), "correct": sum(g == p for _, g, p in rows), "rows": rows}


d = live("P4-06", compute, needs=("llmcache",))
if d:
    if d["rows"]:
        conf = pd.DataFrame(d["rows"], columns=["id", "đúng", "router"])
        display(pd.crosstab(conf["đúng"], conf["router"]))
    else:
        md("Chưa có quyết định router nào trong cache LLM.")
summary = load("eval_summary.json")
if summary is not None:
    md(f"Chính thức (`eval_summary.router`): {pct(summary['router']['accuracy'])}.")
    display(pd.DataFrame(summary["router"]["confusion"]).T.fillna(0).astype(int))'''),
    Step("P4-07", "ẨN", "Hybrid gộp ngữ cảnh hai nguồn và có đường lui về Basic RAG", """
Mục tiêu: chế độ hybrid chạy truy xuất RAG song song với router; tuyến graph/hybrid gọi Text2Cypher, đồ thị lỗi hoặc rỗng thì lui về
Basic RAG (cờ `fallback`); tuyến valuation gọi model định giá. Cell in docstring của bước chuẩn bị hybrid.
""", r'''# P4-07 · ẨN
import inspect

from vre.qa import pipeline as qa_pipeline

md(inspect.getdoc(qa_pipeline.prepare_hybrid))'''),
    Step("P4-08", "LIVE", h_p4_08, """
Mục tiêu: Recall@k = tỷ lệ tin đúng nằm trong top-k; RR = `1 / hạng` của tin đúng đầu tiên, MRR là trung bình RR qua các câu; nDCG
thưởng tin đúng ở hạng cao hơn. Câu ràng buộc có hàng chục tin thoả điều kiện dùng recall "capped" (chia cho min(số tin đúng, k)).
Toy chạy trước để kiểm công thức; số thật đọc từ `eval_summary.retrieval`.
""", r'''# P4-08 · LIVE
def compute():
    from vre.eval.metrics import mrr, recall_at_k

    runs = [(["A", "B", "C", "D", "E"], {"A", "C", "F"}), (["B", "A", "D"], {"A"}), (["X", "Y"], {"Z"})]
    return {"recall": recall_at_k(runs[0][0], runs[0][1], 5), "mrr": mrr(runs)}


d = live("P4-08", compute)
if d:
    md(f"Toy: Recall@5 = {dec(d['recall'], 3)}, MRR = {dec(d['mrr'], 3)}.")
if summary is not None:
    display(pd.DataFrame(summary["retrieval"]).T)'''),
    Step("P4-09", "LIVE", h_p4_09, """
Mục tiêu: Faithfulness = tỷ lệ ý trong câu trả lời được ngữ cảnh ủng hộ (LLM giám khảo tách ý và kiểm từng ý); Answer Relevancy và
Context Precision theo định nghĩa của RAGAS. Giám khảo là mô hình khác mô hình sinh để tránh tự chấm thiên vị. Toy: bốn ý, ba ý có
trong ngữ cảnh.
""", r'''# P4-09 · LIVE
def compute():
    from vre.eval.metrics import faithfulness

    return {"faithfulness": faithfulness([True, True, True, False])}


d = live("P4-09", compute)
if d:
    md(f"Toy: Faithfulness = {dec(d['faithfulness'], 2)}.")
if summary is not None:
    display(pd.DataFrame(summary["ragas"]).T)'''),
    Step("P4-10", "CACHE", h_p4_10, """
Mục tiêu: lượt benchmark chạy mọi câu qua bốn hệ, ghi câu trả lời, nguồn, Cypher, lỗi, độ trễ và token vào
`results/eval/raw/answers.jsonl` (resume theo dòng). Cell tóm tắt file đó: số câu trả lời mỗi hệ, lỗi khi chạy, độ trễ trung vị.
""", r'''# P4-10 · CACHE
def compute():
    ans = pd.DataFrame(jsonl("eval/raw/answers.jsonl"))
    by = ans.groupby("system").agg(n=("id", "size"), errors=("error", lambda s: int(s.notna().sum())),
                                   latency_ms_p50=("latency_ms", "median"), input_tokens=("input_tokens", "mean"))
    return {"n_rows": len(ans), "n_questions": ans["id"].nunique(), "by_system": by.round(1).reset_index().to_dict("records")}


d = live("P4-10", compute, needs=("answers",))
if d:
    table(d["by_system"])'''),
    Step("P4-11", "CACHE", h_p4_11, """
Mục tiêu: độ chính xác theo nhóm × hệ thống, thang điểm `0–2` (đúng, đúng một phần, sai), kèm khoảng tin cậy bootstrap theo câu hỏi.
Cách đọc: thanh lỗi chồng nhau nghĩa là chưa đủ bằng chứng hệ nào tốt hơn ở nhóm đó; câu không có đáp án đo tỷ lệ từ chối đúng.
""", r'''# P4-11 · CACHE
if summary is None:
    missing("`results/eval_summary.json`", "python scripts/vre_pipeline.py eval")
else:
    rows = [{"category": c, "system": s, **{k: v[k] for k in ("n", "accuracy", "mean_points")}, "lo": (v["ci95"] or [None, None])[0],
             "hi": (v["ci95"] or [None, None])[1]}
            for c, by in summary["by_category"].items() for s, v in by.items()]
    cat = pd.DataFrame(rows)
    acc = cat.pivot(index="category", columns="system", values="accuracy")
    display(acc)
    width = 0.8 / acc.shape[1]
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, s in enumerate(acc.columns):
        sub = cat[cat["system"] == s].set_index("category").reindex(acc.index)
        ax.bar(np.arange(len(acc)) + i * width, sub["accuracy"], width, yerr=[sub["accuracy"] - sub["lo"], sub["hi"] - sub["accuracy"]],
               capsize=2, label=s)
    ax.set_xticks(np.arange(len(acc)) + width * (acc.shape[1] - 1) / 2)
    ax.set_xticklabels(acc.index, rotation=30, ha="right")
    ax.set_title("Độ chính xác theo nhóm (thanh lỗi = CI95 bootstrap)")
    ax.legend()
    fig.tight_layout()
    plt.show()
    display(pd.DataFrame(summary["overall"]).T)'''),
    Step("P4-12", "CACHE", h_p4_12, """
Mục tiêu: chi phí và độ trễ theo hệ thống: token vào/ra trung bình, USD mỗi câu theo bảng giá cấu hình, độ trễ p50/p95 và tỷ lệ câu
trả lời trích nguồn không có trong ngữ cảnh (`bad_citation_rate`). Graph và Hybrid tốn thêm lượt LLM cho router và Text2Cypher.
""", r'''# P4-12 · CACHE
if summary is not None:
    display(pd.DataFrame(summary["cost"]).T)'''),
    Step("P4-13", "CACHE", h_p4_13, """
Mục tiêu: phân tích lỗi theo nguyên nhân (router nhầm, Cypher lỗi hoặc sai kết quả, bộ lọc sai, truy xuất trượt, LLM bịa, không từ
chối khi phải từ chối…) và vài ví dụ cụ thể. Lỗi khi chạy (exception) được tách riêng, không bị coi là bịa.
""", r'''# P4-13 · CACHE
err = load("error_analysis.json")
if err is None:
    missing("`results/error_analysis.json`", "python scripts/vre_pipeline.py eval")
else:
    display(pd.DataFrame(err["counts"]).fillna(0).astype(int))
    display(pd.DataFrame(err["examples"]))'''),
    Step("P4-14", "CACHE", h_p4_14, """
Mục tiêu: so sánh cặp hệ thống bằng bootstrap ghép cặp theo câu hỏi (lấy mẫu lại cùng tập câu cho cả hai hệ). Hiệu có CI95 không chứa
không thì có ý nghĩa; `p_value` là tỷ lệ mẫu có dấu ngược. Không có ý nghĩa thống kê thì báo đúng như vậy, không làm tròn thành "thắng".
""", r'''# P4-14 · CACHE
if summary is not None:
    display(pd.DataFrame(summary["pairwise"]))'''),
]


# =============================================================================================== 05_valuation
def _vm(c: Ctx) -> dict | None:
    return c.r("valuation_metrics.json")


def h_p5_01(c: Ctx) -> str:
    v = _vm(c)
    if v is None:
        return "Định giá là module phụ (chưa có valuation_metrics)"
    s = v["split"]
    return f"Chia theo thời gian: train `{s['train_months'][0]}`–`{s['train_months'][-1]}`, test `{s['test_months'][0]}`–`{s['test_months'][-1]}`"


def h_p5_02(c: Ctx) -> str:
    v = _vm(c)
    if v is None:
        return "Đặc trưng của model định giá"
    tab = [f["feature"] for f in v["variants"]["tabular"]["shap_top"]]
    full = v["variants"].get("full")
    extra = "" if full is None else f", `full` thêm {N(sum(f['feature'].startswith('pca_') for f in full['shap_top']))} thành phần PCA của mô tả"
    return f"`tabular` dùng {N(len(tab))} đặc trưng{extra}"


def h_p5_03(c: Ctx) -> str:
    v = _vm(c)
    if v is None:
        return "So sánh biến thể định giá"
    t, b, f = v["variants"]["tabular"], v["variants"]["baseline_ward_median"], v["variants"].get("full")
    if f is None:
        return f"`tabular` MAPE {P(t['mape'])} so với baseline trung vị phường {P(b['mape'])}"
    return f"`full` giảm MAPE xuống {P(f['mape'])} (tabular {P(t['mape'])}, baseline trung vị phường {P(b['mape'])})"


def h_p5_04(c: Ctx) -> str:
    v = _vm(c)
    if v is None:
        return "Sai số theo loại bất động sản"
    rows = v["variants"]["tabular"]["by_property_type"]
    worst = max(rows, key=lambda r: r["mape"])
    return f"Sai số lớn nhất ở loại '{worst['property_type']}' (MAPE tabular {P(worst['mape'])})"


def h_p5_05(c: Ctx) -> str:
    v = _vm(c)
    if v is None:
        return "Sai số theo tỉnh"
    rows = v["variants"]["tabular"]["by_province_top10"]
    best = min(rows, key=lambda r: r["mape"])
    return f"Trong mười tỉnh nhiều tin nhất, {best['province']} dự đoán tốt nhất (MAPE tabular {P(best['mape'])})"


def h_p5_06(c: Ctx) -> str:
    v = _vm(c)
    if v is None:
        return "Đặc trưng quan trọng theo SHAP"
    top = v["variants"]["tabular"]["shap_top"][:2]
    return f"Theo SHAP, `{top[0]['feature']}` và `{top[1]['feature']}` đóng góp nhiều nhất"


def h_p5_07(c: Ctx) -> str:
    d = c.c("P5-07")
    if d is None:
        return "Định giá một căn hộ giả định (chưa chạy demo)"
    r = d["result"]
    return (f"Căn hộ giả định được định giá {D(r['predicted_price_m2_mil'], 1)} triệu/m², giá chào nằm ở mức "
            f"`{r['verdict']}`")


def h_p5_08(c: Ctx) -> str:
    v = _vm(c)
    if v is None:
        return "Model thuê"
    rent = v["rent_model"]
    return "Có model định giá thuê" if rent["trained"] else f"Không huấn luyện model thuê: chỉ có {N(rent['n_rent'])} tin thuê"


NB05_TITLE = """# 05 · Định giá (module phụ)

XGBoost dự đoán log giá/m² của tin bán từ đặc trưng cấu trúc (`tabular`), có thêm vector mô tả nén PCA (`full`); so với baseline
trung vị phường. Định giá là tool `valuate()` cho router (tuyến `valuation`) và endpoint `/api/valuate`.

**Lệnh sinh:** `python scripts/vre_pipeline.py price --variant tabular|full` → `results/valuation_metrics.json`, `models/valuation/`."""

NB05 = [
    Step("P5-01", "ẨN", h_p5_01, """
Mục tiêu: chia train/test theo thời gian (tháng cuối làm test) thay vì ngẫu nhiên, vì tin gần trùng đăng lại nhiều lần sẽ rò rỉ giữa
hai tập nếu chia ngẫu nhiên, làm sai số trông thấp giả. Cell kiểm model có trên máy không (CI không có `models/`).
""", r'''# P5-01 · ẨN
vm = load("valuation_metrics.json")
md("Model tabular: " + ("có trên máy." if available("models") else "không có trên máy này, cell LIVE đọc cache cell."))
if vm is not None:
    display(pd.Series({**vm["split"], **vm["rent_model"]}, name="giá trị").to_frame())'''),
    Step("P5-02", "CACHE", h_p5_02, """
Mục tiêu: đặc trưng. Biến số (diện tích, số phòng, tầng, mặt tiền, đường), biến hạng mục (loại, hướng), tiện ích dạng cờ, và target
encoding của tỉnh/quận/phường/đường/dự án (trung bình log giá tính out-of-fold để không rò nhãn). `full` thêm các thành phần PCA của
vector e5 trung bình theo tin — thông tin trong mô tả mà cột cấu trúc không có.
""", r'''# P5-02 · CACHE
if vm is None:
    missing("`results/valuation_metrics.json`", "python scripts/vre_pipeline.py price --variant tabular")
else:
    feats = [f["feature"] for f in vm["variants"]["tabular"]["shap_top"]]
    groups = {"target encoding": [f for f in feats if f.startswith("te_")], "tiện ích": [f for f in feats if f.startswith("am_")],
              "còn lại": [f for f in feats if not f.startswith(("te_", "am_"))]}
    display(pd.Series({k: ", ".join(v) for k, v in groups.items()}, name="đặc trưng").to_frame())'''),
    Step("P5-03", "CACHE", h_p5_03, """
Mục tiêu: so ba biến thể trên cùng tập test: MAE (triệu/m²), MAPE và R². Baseline trung vị phường là cách định giá "theo khu vực"
mà người mua thường làm; model phải thắng nó mới có ích. Lưu ý: `full` chỉ dùng cho tin thật có mô tả; BĐS giả định nhập tay không
có vector mô tả nên API định giá bằng `tabular`.
""", r'''# P5-03 · CACHE
if vm is not None:
    cmp = pd.DataFrame({k: {m: v[m] for m in ("n_train", "n_test", "mae_mil_m2", "mape", "r2", "train_seconds")}
                        for k, v in vm["variants"].items() if v is not None}).T
    display(cmp)'''),
    Step("P5-04", "CACHE", h_p5_04, """
Mục tiêu: sai số theo loại bất động sản. Đất có MAPE cao vì giá phụ thuộc pháp lý, quy hoạch, mặt tiền — thông tin thường chỉ có
trong mô tả; căn hộ đồng nhất hơn nên dễ đoán hơn. Cách đọc: so cột `full` với `tabular` để thấy mô tả giúp loại nào nhiều nhất.
""", r'''# P5-04 · CACHE
if vm is not None:
    parts = [pd.DataFrame(v["by_property_type"]).set_index("property_type")["mape"].rename(k)
             for k, v in vm["variants"].items() if v is not None and v["by_property_type"]]
    by_type = pd.concat(parts, axis=1)
    display(by_type)
    by_type.plot.barh(figsize=(8, 3.5), title="MAPE theo loại").invert_yaxis()
    plt.tight_layout()
    plt.show()'''),
    Step("P5-05", "CACHE", h_p5_05, """
Mục tiêu: sai số theo tỉnh trong nhóm nhiều tin nhất. Tỉnh có thị trường đồng đều hoặc nhiều tin huấn luyện dự đoán tốt hơn; thành
phố lớn có chênh lệch giá rất lớn giữa các phố nên sai số tuyệt đối cao.
""", r'''# P5-05 · CACHE
if vm is not None:
    parts = [pd.DataFrame(v["by_province_top10"]).set_index("province")["mape"].rename(k)
             for k, v in vm["variants"].items() if v is not None and v["by_province_top10"]]
    display(pd.concat(parts, axis=1))'''),
    Step("P5-06", "CACHE", h_p5_06, """
Mục tiêu: SHAP phân rã từng dự đoán thành đóng góp của từng đặc trưng (trên thang log giá); trung bình trị tuyệt đối cho tầm quan
trọng toàn cục. Cách đọc: vị trí (target encoding quận/phường) và loại hình chiếm phần lớn; ở `full`, thành phần PCA của mô tả chen vào
nhóm đầu — mô tả mang thông tin giá thật.
""", r'''# P5-06 · CACHE
if vm is not None:
    shap = {k: v for k, v in vm["variants"].items() if v is not None and v["shap_top"]}
    fig, axes = plt.subplots(1, len(shap), figsize=(6 * len(shap), 4.5), squeeze=False)
    for ax, (k, v) in zip(axes[0], shap.items(), strict=True):
        pd.DataFrame(v["shap_top"]).head(15).set_index("feature")["mean_abs_contribution"].plot.barh(ax=ax, title=k)
        ax.invert_yaxis()
    fig.tight_layout()
    plt.show()'''),
    Step("P5-07", "LIVE", h_p5_07, """
Mục tiêu: một lần `valuate()` thật cho căn hộ giả định có giá chào. Kết quả gồm giá/m² và giá tổng dự đoán, khoảng giá (từ phân vị sai
số trên tập test theo loại), trung vị phường để đối chiếu, phán quyết so với giá chào và đóng góp SHAP của chính căn này.
""", r'''# P5-07 · LIVE
PAYLOAD = {"property_type": "Căn hộ chung cư", "province": "Hồ Chí Minh", "district": "Quận 7", "area_m2": 70, "bedrooms": 2,
           "bathrooms": 2, "asking_price_vnd": 4.5e9}


def compute():
    from vre.valuation.predict import valuate

    return {"input": PAYLOAD, "result": valuate(PAYLOAD, variant="tabular")}


d = live("P5-07", compute, needs=("models", "data"))
if d:
    r = d["result"]
    md(f"Dự đoán {dec(r['predicted_price_m2_mil'], 1)} triệu/m² ⇒ {dec(r['predicted_price_vnd'] / 1e9, 2)} tỷ "
       f"(khoảng {dec(r['interval_price_vnd']['low'] / 1e9, 2)} – {dec(r['interval_price_vnd']['high'] / 1e9, 2)} tỷ); "
       f"giá chào {dec(PAYLOAD['asking_price_vnd'] / 1e9, 2)} tỷ ⇒ `{r['verdict']}`; model `{r['model']['variant']}`.")
    table(r["shap"])'''),
    Step("P5-08", "CACHE", h_p5_08, """
Mục tiêu: phạm vi của module. Dataset gần như toàn tin bán; số tin thuê dưới ngưỡng tối thiểu nên không huấn luyện model thuê, và
`/api/valuate` trả lỗi rõ cho `deal_type = rent` thay vì đoán. Giá là giá chào do người đăng nhập, không phải giá giao dịch.
""", r'''# P5-08 · CACHE
if vm is not None:
    display(pd.Series(vm["rent_model"], name="rent_model").to_frame())'''),
]


NOTEBOOKS: dict[str, tuple[str, list[Step]]] = {
    "00_setup.ipynb": (NB00_TITLE, NB00),
    "01_rag_indexing.ipynb": (NB01_TITLE, NB01),
    "02_retrieval_generation.ipynb": (NB02_TITLE, NB02),
    "03_knowledge_graph.ipynb": (NB03_TITLE, NB03),
    "04_graphrag_eval.ipynb": (NB04_TITLE, NB04),
    "05_valuation.ipynb": (NB05_TITLE, NB05),
}


def render(name: str, ctx: Ctx | None = None) -> nbformat.NotebookNode:
    """Notebook CHƯA chạy (nguồn) cho `name`; heading điền từ kết quả + cache cell hiện có."""
    ctx = ctx or Ctx()
    title, steps = NOTEBOOKS[name]
    cells = [nbformat.v4.new_markdown_cell(title)]
    if name != "00_setup.ipynb":
        cells.append(nbformat.v4.new_code_cell(RUN_SETUP))
    for step in steps:
        assert step.kind in KINDS and step.code.startswith(f"# {step.id} · {step.kind}\n"), step.id
        cells.append(nbformat.v4.new_markdown_cell(_head(step, ctx)))
        cells.append(nbformat.v4.new_code_cell(step.code))
    for i, cell in enumerate(cells):
        cell.id = f"c{i:03d}"  # id cố định: diff ổn định giữa các lần sinh
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": KERNEL, "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def sources(nb: nbformat.NotebookNode) -> list[tuple[str, str]]:
    return [(c.cell_type, c.source) for c in nb.cells]


def execute(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    from nbclient import NotebookClient

    os.environ.setdefault("VRE_NB_STRICT", "1")
    NotebookClient(nb, timeout=TIMEOUT_S, kernel_name=KERNEL, resources={"metadata": {"path": str(NB_DIR)}}).execute()
    for cell in nb.cells:  # bỏ dấu thời gian thực thi để diff ổn định
        cell.get("metadata", {}).pop("execution", None)
    return nb


def build(name: str) -> Path:
    """Sinh nguồn → chạy (cell LIVE ghi cache cell) → điền lại ô markdown từ cache vừa ghi (ô code không đổi nên không chạy lại)."""
    nb = execute(render(name))
    for cell, fresh in zip(nb.cells, render(name).cells, strict=True):
        if cell.cell_type == "markdown":
            cell.source = fresh.source
    path = NB_DIR / name
    nbformat.write(nb, path)
    return path


def main(argv: list[str]) -> int:
    selected = [n for n in NOTEBOOKS if not argv or any(n.startswith(a) for a in argv)]
    if not selected:
        print(f"không có notebook khớp {argv}; có: {list(NOTEBOOKS)}", file=sys.stderr)
        return 2
    for name in selected:
        print("đã chạy và lưu", build(name), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
