"""Ranh giới hành chính cho bản đồ web + centroid cho API (data-contract §8).

Nguồn: geoBoundaries gbOpen VNM ADM1 (63 tỉnh + Côn Đảo, vintage 2008) và ADM2 (708 quận/huyện, vintage 2020).
Đây là địa giới TRƯỚC sáp nhập 07/2025 — khớp với dữ liệu tin đăng (63 tỉnh cũ).

Các bước:
1. Kiểm tra URL bằng HTTP HEAD (fail-fast nếu không 200), tải GeoJSON gốc vào `data/geo/raw/` (bỏ qua nếu đã có).
2. Đơn giản hoá bằng `npx mapshaper -simplify 5% keep-shapes` (mapshaper là devDependency của `web/`).
3. Gán mỗi quận/huyện vào tỉnh bằng point-in-polygon (ADM2 không mang tên tỉnh), khớp tên với địa danh trong
   `listings.parquet` qua `vre.data.places.normalize_place` + gấp dấu (ADM2 viết không dấu).
4. Ghi `web/public/geo/{provinces,districts}.json` (thuộc tính `key`, `name`) và `data/geo/centroids.json`.

Chạy: `& $VRE_PY scripts\\build_geo.py` (sau `. scripts\\vre_env.ps1`). Không cần mạng nếu `data/geo/raw/` đã có file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import duckdb

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from vre.data.places import normalize_place  # noqa: E402

COMMIT = "9469f09"  # bản phát hành geoBoundaries cố định (API current trỏ về commit này ngày 2026-10-15)
BASE_URL = f"https://github.com/wmgeolab/geoBoundaries/raw/{COMMIT}/releaseData/gbOpen/VNM"
LAYERS = {"ADM1": f"{BASE_URL}/ADM1/geoBoundaries-VNM-ADM1.geojson",
          "ADM2": f"{BASE_URL}/ADM2/geoBoundaries-VNM-ADM2.geojson"}
SOURCE = ("geoBoundaries gbOpen VNM ADM1 (Public Domain, vintage 2008) / "
          "ADM2 (OCHA ROAP, CC BY 3.0 IGO, vintage 2020) — wmgeolab/geoBoundaries@" + COMMIT)
SOURCE_VINTAGE = "ADM1 2008; ADM2 2020"
SIMPLIFY = "5%"
PRECISION = "0.0001"

# Tên trong geoBoundaries khác cách viết của dữ liệu (sau normalize_place). Khoá = tên đã gấp dấu.
PROVINCE_ALIASES = {"thuathienhue": "Huế", "condao": "Bà Rịa - Vũng Tàu"}
# Quận/huyện đã sáp nhập sau vintage 2020: nhiều polygon ADM2 → 1 khoá dữ liệu. (tỉnh, tên ADM2 gấp dấu) → tên dữ liệu.
DISTRICT_ALIASES = {
    ("Hồ Chí Minh", "quan2"): "Thủ Đức", ("Hồ Chí Minh", "quan9"): "Thủ Đức",
    ("Bà Rịa - Vũng Tàu", "longdien"): "Long Đất", ("Bà Rịa - Vũng Tàu", "datdo"): "Long Đất",
    ("Bà Rịa - Vũng Tàu", "tanthanh"): "Phú Mỹ",
    ("Cao Bằng", "quanguyen"): "Quảng Hòa", ("Cao Bằng", "phuchoa"): "Quảng Hòa",
    ("Thanh Hóa", "tinhgia"): "Nghi Sơn",
}
_SUFFIX = re.compile(r"\b(district|city|town)\b", re.I)
_ASCII_PREFIX = re.compile(r"^(huyen|thi xa|thanh pho)\s+", re.I)  # ADM2 không dấu: "Huyen Cai Lay", "Thi xa Hong Ngu"


def fold(name: str | None) -> str:
    """Bỏ dấu, chữ thường, chỉ giữ chữ + số (ADM2 viết không dấu, ADM1 dùng en dash)."""
    if not name:
        return ""
    text = unicodedata.normalize("NFD", _SUFFIX.sub("", name).replace("đ", "d").replace("Đ", "D"))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]", "", text.lower())


def head_ok(url: str) -> None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 — URL cố định https
        if response.status != 200:
            raise RuntimeError(f"HEAD {url} → {response.status}")


def download(raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for layer, url in LAYERS.items():
        path = raw_dir / url.rsplit("/", 1)[1]
        if not path.exists():
            head_ok(url)
            tmp = path.with_suffix(".tmp")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310
            os.replace(tmp, path)
        out[layer] = path
    return out


def simplify(src: Path, dst: Path, ratio: str) -> None:
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if npx is None:
        raise RuntimeError("không tìm thấy npx — cần Node 24 + `npm --prefix web ci`")
    cmd = [npx, "--no-install", "mapshaper", "-i", str(src), "-simplify", ratio, "keep-shapes",
           "-o", "format=geojson", f"precision={PRECISION}", str(dst)]
    subprocess.run(cmd, check=True, cwd=CODE_ROOT / "web")


# ---------- hình học phẳng trên (lon, lat): đủ cho centroid hiển thị ----------

def polygons(geometry: dict) -> list[list[list[list[float]]]]:
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    raise ValueError(f"geometry không hỗ trợ: {geometry['type']}")


def ring_area_centroid(ring: list[list[float]]) -> tuple[float, float, float]:
    area = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1], strict=True):
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area /= 2
    if area == 0:
        return 0.0, ring[0][0], ring[0][1]
    return area, cx / (6 * area), cy / (6 * area)


def centroid(geometries: list[dict]) -> tuple[float, float]:
    """Centroid có trọng số diện tích của mọi vòng ngoài (trừ lỗ)."""
    total = sx = sy = 0.0
    for geometry in geometries:
        for poly in polygons(geometry):
            for index, ring in enumerate(poly):
                area, cx, cy = ring_area_centroid(ring)
                weight = abs(area) * (1 if index == 0 else -1)
                total += weight
                sx += cx * weight
                sy += cy * weight
    return sx / total, sy / total


def point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    inside = False
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1], strict=True):
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def contains(geometry: dict, x: float, y: float) -> bool:
    return any(point_in_ring(x, y, poly[0]) and not any(point_in_ring(x, y, h) for h in poly[1:])
               for poly in polygons(geometry))


def probe_points(geometry: dict, grid: int = 9) -> list[tuple[float, float]]:
    """Điểm lưới nằm TRONG polygon (để bỏ phiếu tỉnh); rỗng ⇒ dùng centroid. Ranh ADM1 thô nên 1 điểm không đủ."""
    xs = [x for poly in polygons(geometry) for x, _ in poly[0]]
    ys = [y for poly in polygons(geometry) for _, y in poly[0]]
    points = [(min(xs) + (max(xs) - min(xs)) * (i + 0.5) / grid, min(ys) + (max(ys) - min(ys)) * (j + 0.5) / grid)
              for i in range(grid) for j in range(grid)]
    inside = [(x, y) for x, y in points if contains(geometry, x, y)]
    return inside or [centroid([geometry])]


# ---------- khớp với dữ liệu ----------

def data_places(clean_path: Path) -> tuple[set[str], set[tuple[str, str]]]:
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'; SET threads=4")
    rows = con.execute("SELECT DISTINCT province, district FROM read_parquet(?) WHERE province IS NOT NULL",
                       [str(clean_path)]).fetchall()
    provinces = {p for p, _ in rows}
    districts = {(p, d) for p, d in rows if d is not None}
    return provinces, districts


def match_provinces(features: list[dict], provinces: set[str]) -> None:
    by_fold = {fold(p): p for p in provinces}
    for feature in features:
        raw = feature["properties"]["shapeName"].strip()
        folded = fold(normalize_place(raw, "province"))
        key = PROVINCE_ALIASES.get(folded) or by_fold.get(folded)
        feature["properties"] = {"key": key, "name": key or raw}


def vertex_distance(geometry: dict, x: float, y: float) -> float:
    return min(((px - x) ** 2 + (py - y) ** 2) ** 0.5 for poly in polygons(geometry) for px, py in poly[0])


def province_votes(geometry: dict, provinces: list[tuple[str, dict]]) -> dict[str, int]:
    """Số điểm lưới của quận nằm trong từng polygon tỉnh (ADM1 thô và chồng lấn nhau ⇒ đếm mọi tỉnh chứa điểm)."""
    votes: dict[str, int] = {}
    for x, y in probe_points(geometry):
        for key, province_geometry in provinces:
            if contains(province_geometry, x, y):
                votes[key] = votes.get(key, 0) + 1
    return votes


def match_districts(features: list[dict], provinces: list[tuple[str, dict]],
                    districts: set[tuple[str, str]]) -> None:
    """Tỉnh của quận: ưu tiên TÊN (quận có trong dữ liệu ở đúng 1 tỉnh lân cận), hình học ADM1 chỉ để phân xử.

    ADM1 (2008, nguồn Wikipedia) lệch tới ~0,1° nên point-in-polygon thuần gán nhầm quận sát ranh giới
    (vd Từ Sơn → Hà Nội). Ứng viên tên chỉ được nhận khi có phiếu hình học hoặc tỉnh cách polygon ≤ NEAR_DEG,
    tránh gán một huyện vùng núi trùng tên vào tỉnh ở đầu kia đất nước.
    """
    near_deg = 0.3
    by_name: dict[str, set[str]] = {}
    exact = {(p, fold(d)): d for p, d in districts}
    for province, district in districts:
        by_name.setdefault(fold(district), set()).add(province)
    for province, folded in DISTRICT_ALIASES:
        by_name.setdefault(folded, set()).add(province)
    shapes = dict(provinces)
    for feature in features:
        raw = feature["properties"]["shapeName"].strip()
        folded = fold(normalize_place(_ASCII_PREFIX.sub("", raw), "district"))
        votes = province_votes(feature["geometry"], provinces)
        x, y = centroid([feature["geometry"]])
        candidates = [p for p in by_name.get(folded, set())
                      if votes.get(p) or (p in shapes and vertex_distance(shapes[p], x, y) <= near_deg)]
        if candidates:
            province = max(candidates, key=lambda p: (votes.get(p, 0), -vertex_distance(shapes[p], x, y)))
            name = DISTRICT_ALIASES.get((province, folded)) or exact[(province, folded)]
        else:
            province = max(votes, key=lambda k: votes[k]) if votes else min(
                shapes, key=lambda p: vertex_distance(shapes[p], x, y))
            # Đảo xa (Trường Sa, Hoàng Sa) bị lọc theo khoảng cách nhưng vẫn khớp tên ở tỉnh gán bằng hình học.
            name = DISTRICT_ALIASES.get((province, folded)) or exact.get((province, folded))
        key = f"{province}|{name}" if name else None
        feature["properties"] = {"key": key, "name": name or raw, "province": province}


def dissolve_provinces(districts_path: Path, out_path: Path) -> list[dict]:
    """Tỉnh = hợp các quận đã gán tỉnh (ranh tỉnh khớp tuyệt đối với ranh quận; ADM1 gốc quá thô để hiển thị)."""
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if npx is None:
        raise RuntimeError("không tìm thấy npx — cần Node 24 + `npm --prefix web ci`")
    subprocess.run([npx, "--no-install", "mapshaper", "-i", str(districts_path), "-dissolve", "province",
                    "-o", "format=geojson", f"precision={PRECISION}", str(out_path)], check=True, cwd=CODE_ROOT / "web")
    features = json.loads(out_path.read_text(encoding="utf-8"))["features"]
    for feature in features:
        province = feature["properties"]["province"]
        feature["properties"] = {"key": province, "name": province}
    return features


def items_for(level: str, features: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for feature in features:
        if feature["properties"]["key"]:
            groups.setdefault(feature["properties"]["key"], []).append(feature)
    out = []
    for key, group in sorted(groups.items()):
        lon, lat = centroid([f["geometry"] for f in group])
        name = key.split("|")[-1]
        out.append({"level": level, "key": key, "name": name, "lat": round(lat, 5), "lon": round(lon, 5)})
    return out


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    data_root = Path(os.environ.get("VRE_DATA_ROOT", r"D:\Vietnam-Real-Estates"))
    parser.add_argument("--data-root", type=Path, default=data_root)
    args = parser.parse_args()
    geo_dir = args.data_root / "data" / "geo"
    raw = download(geo_dir / "raw")
    adm2_simplified = geo_dir / "raw" / "ADM2_simplified.geojson"
    simplify(raw["ADM2"], adm2_simplified, SIMPLIFY)
    adm2 = json.loads(adm2_simplified.read_text(encoding="utf-8"))["features"]
    # ADM1 gốc (không đơn giản hoá) chỉ dùng để phân xử tỉnh của quận; không xuất ra web.
    adm1_raw = json.loads(raw["ADM1"].read_text(encoding="utf-8"))["features"]

    provinces, districts = data_places(args.data_root / "data" / "clean" / "listings.parquet")
    match_provinces(adm1_raw, provinces)
    province_shapes = [(f["properties"]["key"], f["geometry"]) for f in adm1_raw if f["properties"]["key"]]
    match_districts(adm2, province_shapes, districts)

    public = CODE_ROOT / "web" / "public" / "geo"
    write_json(public / "districts.json", {"type": "FeatureCollection", "features": adm2})
    adm1 = dissolve_provinces(public / "districts.json", geo_dir / "raw" / "provinces_dissolved.geojson")
    write_json(public / "provinces.json", {"type": "FeatureCollection", "features": adm1})

    items = items_for("province", adm1) + items_for("district", adm2)
    found = {(i["level"], i["key"]) for i in items}
    wanted = [("province", p) for p in sorted(provinces)] + [("district", f"{p}|{d}") for p, d in sorted(districts)]
    wanted_set = set(wanted)
    unmatched = [{"level": level, "key": key} for level, key in wanted if (level, key) not in found]
    coverage = {level: round(sum(1 for lv, k in wanted if lv == level and (lv, k) in found)
                             / max(1, sum(1 for lv, _ in wanted if lv == level)), 4)
                for level in ("province", "district")}
    write_json(geo_dir / "centroids.json", {
        "schema_version": 1, "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": SOURCE, "source_vintage": SOURCE_VINTAGE,
        "items": [i for i in items if (i["level"], i["key"]) in wanted_set],
        "unmatched": unmatched, "coverage": coverage})
    print(json.dumps({"coverage": coverage, "n_unmatched": len(unmatched),
                      "sizes_kb": {n: (public / f"{n}.json").stat().st_size // 1024 for n in ("provinces", "districts")},
                      "unmatched_sample": unmatched[:25]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
