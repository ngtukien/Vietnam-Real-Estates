"""Schema prompt + few-shot Text2Cypher cho KG mới (data-contract §6).

Quy tắc chính: câu hỏi tổng hợp (trung vị, số tin, xếp hạng, xu hướng tháng) đọc node `MarketStat` tính sẵn,
KHÔNG quét `Listing` (2,5M node ⇒ timeout). Mọi ví dụ tổng hợp đều lọc `m.n >= 20` để không xếp hạng trên vài tin.
"""

from __future__ import annotations

from vre.data.amenities import amenity_table

PROPERTY_TYPES = ("Nhà", "Đất", "Căn hộ chung cư", "Biệt thự/Nhà liền kề", "Shophouse")

SCHEMA_TEMPLATE = """Nút (khoá duy nhất in đậm bằng dấu *) và thuộc tính:
- (:Listing {{listing_id*: int, title, property_type, deal_type: 'sale'|'rent', price_vnd: VND, area_m2: m²,
            price_m2_mil: triệu đồng/m² (null nếu giá bất thường), bedrooms, bathrooms, floors, frontage_m,
            road_width_m, direction, published_at, month: 'YYYY-MM', dup_cluster_id}})
- (:PropertyType {{name*}})  giá trị: {types}
- (:Province {{key* = name, name, name_2025: tỉnh sau sáp nhập 07/2025, n_listings, median_price_m2_mil}})
- (:District {{key* = 'Tỉnh|Quận', name, n_listings, median_price_m2_mil}})  ví dụ key: 'Hồ Chí Minh|Quận 7', 'Hà Nội|Cầu Giấy'
- (:Ward {{key* = 'Tỉnh|Quận|Phường', name, n_listings, median_price_m2_mil}})
- (:Street {{key* = 'Tỉnh|Quận|Đường', name, n_listings}})
- (:Project {{key* = 'Tỉnh|tên dự án viết thường', name, n_listings, median_price_m2_mil}})
- (:Month {{key* = 'YYYY-MM', year, month}})  dữ liệu từ 2025-06 đến 2026-03
- (:Amenity {{key*, name}})  key: {amenities}
- (:Cluster {{cluster_id*, n_wards, median_price_m2_mil, label}})  cụm phường có hồ sơ giá/loại hình giống nhau
- (:MarketStat {{key*, scope: 'province'|'district'|'ward'|'project', area_key: key của khu vực,
               property_type: tên loại hoặc 'ALL', month: 'YYYY-MM' hoặc 'ALL', deal_type: 'sale',
               n: số tin có giá, median_price_m2_mil, p25_price_m2_mil, p75_price_m2_mil, median_price_vnd, median_area_m2}})
  thống kê TÍNH SẴN (chỉ tin bán có giá; chỉ có khi n >= 5).
Quan hệ:
- (:Listing)-[:OF_TYPE]->(:PropertyType); (:Listing)-[:POSTED_IN]->(:Month); (:Listing)-[:HAS_AMENITY]->(:Amenity)
- (:Listing)-[:IN_WARD]->(:Ward)-[:IN_DISTRICT]->(:District)-[:IN_PROVINCE]->(:Province)
- Tin thiếu phường: (:Listing)-[:IN_DISTRICT]->(:District); thiếu cả quận: (:Listing)-[:IN_PROVINCE]->(:Province).
  Mọi tin của một quận: MATCH (d:District {{key: 'Tỉnh|Quận'}}) WITH d MATCH (l:Listing)-[:IN_WARD|IN_DISTRICT*1..2]->(d)
  (WITH d bắt Neo4j bắt đầu từ quận; thiếu nó planner quét chỉ mục giá trên mọi Listing và quá thời gian).
- (:Listing)-[:ON_STREET]->(:Street)-[:IN_DISTRICT]->(:District); (:Listing)-[:IN_PROJECT]->(:Project)-[:IN_DISTRICT]->(:District)
- (:Ward)-[:IN_CLUSTER]->(:Cluster)
- (:MarketStat)-[:STAT_OF]->(:Province|District|Ward|Project)"""

RULES = """Bạn viết MỘT câu Cypher chỉ đọc (MATCH/WHERE/WITH/RETURN/ORDER BY/LIMIT) cho Neo4j để trả lời câu hỏi.
Không dùng CREATE, MERGE, DELETE, SET, REMOVE, DROP, CALL, LOAD CSV, FOREACH.
- Trung vị giá, giá/m², số tin, xếp hạng khu vực, xu hướng theo tháng: đọc MarketStat (month: 'ALL' khi không hỏi
  tháng; property_type: 'ALL' khi không hỏi loại hình; deal_type: 'sale'). KHÔNG tính avg/median/count trên Listing.
- Khi xếp hạng hoặc so sánh khu vực, lọc m.n >= 20 để kết quả có ý nghĩa.
- Quận/phường trùng tên ở nhiều tỉnh: khi biết tỉnh dùng key ('Tỉnh|Quận'); chỉ dùng name khi không biết tỉnh.
- Tên tỉnh: 'Hồ Chí Minh' (không 'TP.HCM'), 'Hà Nội'. Quận số: 'Quận 7'. Dự án: so khớp toLower(p.name) CONTAINS '<tên viết thường>'.
- "Khu vực tương tự" một phường = các phường cùng Cluster.
- Khi trả về tin cụ thể, luôn trả l.listing_id AS listing_id. Tối đa 20 dòng. Dùng đúng tên thuộc tính và đơn vị."""

FEW_SHOT: tuple[tuple[str, str], ...] = (
    ("Ở Hồ Chí Minh, quận nào có giá/m² nhà trung vị thấp nhất (ít nhất 20 tin)?",
     """MATCH (m:MarketStat {scope: 'district', property_type: 'Nhà', month: 'ALL', deal_type: 'sale'})-[:STAT_OF]->(d:District)
      -[:IN_PROVINCE]->(:Province {key: 'Hồ Chí Minh'})
WHERE m.n >= 20
RETURN d.name AS district, round(m.median_price_m2_mil, 2) AS median_price_m2_mil, m.n AS n
ORDER BY median_price_m2_mil ASC LIMIT 1"""),
    ("So sánh giá/m² trung vị căn hộ giữa Cầu Giấy và Hoàng Mai ở Hà Nội.",
     """MATCH (m:MarketStat {scope: 'district', property_type: 'Căn hộ chung cư', month: 'ALL', deal_type: 'sale'})
WHERE m.area_key IN ['Hà Nội|Cầu Giấy', 'Hà Nội|Hoàng Mai'] AND m.n >= 20
RETURN m.area_key AS area, round(m.median_price_m2_mil, 2) AS median_price_m2_mil, m.n AS n
ORDER BY median_price_m2_mil DESC"""),
    ("Giá/m² trung vị đất ở Thủ Đức thay đổi thế nào qua các tháng?",
     """MATCH (m:MarketStat {scope: 'district', area_key: 'Hồ Chí Minh|Thủ Đức', property_type: 'Đất', deal_type: 'sale'})
WHERE m.month <> 'ALL' AND m.n >= 20
RETURN m.month AS month, round(m.median_price_m2_mil, 2) AS median_price_m2_mil, m.n AS n
ORDER BY month"""),
    ("Tỉnh nào có nhiều tin nhà đất có giá nhất trong tháng 2026-02?",
     """MATCH (m:MarketStat {scope: 'province', property_type: 'ALL', month: '2026-02', deal_type: 'sale'})
RETURN m.area_key AS province, m.n AS n, round(m.median_price_m2_mil, 2) AS median_price_m2_mil
ORDER BY n DESC LIMIT 5"""),
    ("Căn hộ ở dự án Vinhomes Grand Park đắt hơn hay rẻ hơn trung vị của quận chứa dự án?",
     """MATCH (mp:MarketStat {scope: 'project', property_type: 'Căn hộ chung cư', month: 'ALL', deal_type: 'sale'})
      -[:STAT_OF]->(p:Project)-[:IN_DISTRICT]->(d:District)
WHERE toLower(p.name) CONTAINS 'vinhomes grand park' AND mp.n >= 20
MATCH (md:MarketStat {scope: 'district', property_type: 'Căn hộ chung cư', month: 'ALL', deal_type: 'sale'})-[:STAT_OF]->(d)
RETURN p.name AS project, d.name AS district, round(mp.median_price_m2_mil, 2) AS project_median_m2_mil,
       round(md.median_price_m2_mil, 2) AS district_median_m2_mil,
       round(mp.median_price_m2_mil / md.median_price_m2_mil - 1, 3) AS relative_diff, mp.n AS n_project
ORDER BY n_project DESC LIMIT 3"""),
    ("Trung vị giá/m² nhà ở quận có dự án Ecopark so với trung vị toàn tỉnh?",
     """MATCH (p:Project)-[:IN_DISTRICT]->(d:District)-[:IN_PROVINCE]->(pr:Province)
WHERE toLower(p.name) CONTAINS 'ecopark'
MATCH (md:MarketStat {scope: 'district', property_type: 'Nhà', month: 'ALL', deal_type: 'sale'})-[:STAT_OF]->(d)
MATCH (mv:MarketStat {scope: 'province', property_type: 'Nhà', month: 'ALL', deal_type: 'sale'})-[:STAT_OF]->(pr)
WHERE md.n >= 20
RETURN p.name AS project, d.name AS district, pr.name AS province, round(md.median_price_m2_mil, 2) AS district_median_m2_mil,
       round(mv.median_price_m2_mil, 2) AS province_median_m2_mil, md.n AS n_district
ORDER BY n_district DESC LIMIT 3"""),
    ("Liệt kê căn hộ 2 phòng ngủ ở Quận 7, Hồ Chí Minh dưới 4 tỷ.",
     """MATCH (d:District {key: 'Hồ Chí Minh|Quận 7'})
WITH d
MATCH (l:Listing)-[:IN_WARD|IN_DISTRICT*1..2]->(d)
WHERE l.property_type = 'Căn hộ chung cư' AND l.bedrooms = 2 AND l.price_vnd <= 4e9 AND l.deal_type = 'sale'
RETURN l.listing_id AS listing_id, l.title AS title, l.price_vnd AS price_vnd, l.area_m2 AS area_m2
ORDER BY l.price_vnd ASC LIMIT 20"""),
    ("Đường nào ở Cầu Giấy, Hà Nội có nhiều tin đăng nhất?",
     """MATCH (s:Street)-[:IN_DISTRICT]->(:District {key: 'Hà Nội|Cầu Giấy'})
RETURN s.name AS street, s.n_listings AS n_listings
ORDER BY n_listings DESC LIMIT 5"""),
    ("Có bao nhiêu nhà có thang máy ở Ba Đình, Hà Nội?",
     """MATCH (d:District {key: 'Hà Nội|Ba Đình'})
WITH d
MATCH (l:Listing)-[:IN_WARD|IN_DISTRICT*1..2]->(d)
MATCH (l)-[:HAS_AMENITY]->(:Amenity {key: 'thang_may'})
WHERE l.property_type = 'Nhà'
RETURN count(DISTINCT l) AS n_listings"""),
    ("Những khu vực nào tương tự phường Dịch Vọng (Cầu Giấy, Hà Nội)?",
     """MATCH (:Ward {key: 'Hà Nội|Cầu Giấy|Dịch Vọng'})-[:IN_CLUSTER]->(c:Cluster)<-[:IN_CLUSTER]-(w:Ward)-[:IN_DISTRICT]->(d:District)
WHERE w.key <> 'Hà Nội|Cầu Giấy|Dịch Vọng' AND w.n_listings >= 20
RETURN w.name AS ward, d.name AS district, round(w.median_price_m2_mil, 2) AS median_price_m2_mil, c.label AS cluster
ORDER BY w.n_listings DESC LIMIT 10"""),
)


def schema_text(types: tuple[str, ...] | list[str] = PROPERTY_TYPES) -> str:
    return SCHEMA_TEMPLATE.format(types=", ".join(f"'{t}'" for t in types),
                                  amenities=", ".join(f"'{k}' ({v['name']})" for k, v in amenity_table().items()))


def few_shot_text() -> str:
    return "\n\n".join(f"Ví dụ {i}. Câu hỏi: {q}\nCypher:\n{c}" for i, (q, c) in enumerate(FEW_SHOT, start=1))


def build_system(types: tuple[str, ...] | list[str] = PROPERTY_TYPES) -> str:
    """Prompt hệ thống Text2Cypher: quy tắc + schema + few-shot."""
    return f"{RULES}\n\n{schema_text(types)}\n\n{few_shot_text()}"
