# P3 — Why LLM filters match 0 Qdrant points (`filter_dropped`)

Read-only investigation, 2026-10-15. Code: worktree `D:\Vietnam-Real-Estates-wt\ablation` @ 202db99.
Method: `parse_filters` (LLM disk cache; 6 hits + 15 new calls = 21 `gt_kind=ids` questions of 27), compile with
`conditions`/`to_qdrant`, `count(exact=True)` on alias `vre_chunks` for the full filter and for each condition alone,
then `scroll` the GT listing payloads (`results/benchmark_gt.json`). Not run (LLM budget): CS10, CS11, CS12, PS08, PS09, PS10.
Script: `C:\Users\Nghaiz\AppData\Local\Temp\claude\d--Vietnam-Real-Estates\78375574-07f7-4335-811e-4c4bc63b3baf\scratchpad\fdrop.py`, raw output `...\scratchpad\fdrop.json`.

## Result: 4 of 21 drop (full count 0)

| id | extracted filter (non-null) | zero-match field | extracted value | GT payload value |
|---|---|---|---|---|
| LK01 | district=Cầu Giấy, Căn hộ, sale, bedrooms=3, price 8000–8000, area 80–80, project | project | `Sun Feliza Suites` | `Sun Feliza Suites Cầu Giấy` |
| LK05 | Biệt thự/Nhà liền kề, price 18500–18500, area ≥105, project | project | `Thung Lũng Mùa Xuân Ecopark` | `Khu đô thị Ecopark` |
| LK07 | price 10450–10450, project | project | `Aqua City Biên Hòa` | `Aqua City` |
| LK09 | district=Quận 1, Nhà, deal_type, area 160–160, street | deal_type (alone: 335 pts, but GT fails it) | `rent` | `sale` |

Non-dropping but lossy: CS09 `bedrooms=5` exact while question says "từ 5 phòng ngủ trở lên" (GT bedrooms 5,5,8 → one GT excluded).
LK10 (no-diacritic question) extracted only Nhà + max price → under-filtered (204k pts), not a drop.

Everything else compiled correctly and matched the GT payload: province/district/ward (already pass through
`normalize_place`, `filters.py:118-121` — "Q7"/"q1" → "Quận 7"/"Quận 1" correct), price units (triệu × 1e6 vs `price_vnd`),
exact price/area equality, bedrooms Range, month MatchAny (CS09 `2026-03`). The ablation smoke's 2/5 drops = LK01 + LK05.

## Root cause

1. **Project is exact keyword match on a free-text, non-canonical vocabulary** (3 of 4 drops).
   `filters.py:123-126` only `normalize_text`s `project`; `filters.py:216` puts `project` (and `street`) in `_EQ_FIELDS`;
   `filters.py:259-260` compiles it to `MatchValue`. Payload project names carry district suffixes, umbrella vs
   sub-project names, "Chung cư/Khu đô thị/KDC" prefixes and parenthesised aliases (Qdrant facet, district=Biên Hòa:
   `Aqua City`, `Khu đô thị Izumi City Đồng Nai`, `Khu dân cư Kara Garden Biên Hòa`; Văn Giang: `Khu đô thị Ecopark`,
   `Swanlake Residences The Onsen Ecopark` vs `Swan Lake Onsen Residences`; Cầu Giấy: `Sun Feliza Suites Cầu Giấy`,
   `Hà Nội Paragon (Paragon Tower)`). A user/LLM spelling equals the stored string only by luck.
   `normalize_place` would NOT fix this (it handles province/district/ward prefixes and numbers only).
2. **deal_type misread** (1 drop): "đang cho thuê bao nhiêu" (asks rent income of a 55 tỷ sale listing) → `rent`.
   Prompt `filters.py:192-196` does not say deal_type is the type of the listing sought. Amplifier: the collection holds
   only **335 `rent` points of 2,685,246** (sale 2,684,911), so any `rent` filter is near-certain to drop.
3. **All-or-nothing fallback** amplifies both: `retrieve.py:137-138` (and `ablation.py:145-146`) drop the WHOLE filter
   when 0 hits, discarding the correct district/type/price constraints with the one bad field.

## Fix proposal

1. `filters.py` — new `resolve_project(name, district=None, province=None) -> list[str]`, used by `conditions()` instead
   of the `eq` on `project` (same for `street`, lower priority): fold both sides (NFC, lower, strip diacritics,
   strip prefixes `chung cư|khu đô thị|khu dân cư|kđt|kdc|dự án|phân khu`, strip parentheses, strip trailing
   district/province name), candidates = vocabulary values whose folded token set contains the query's or is contained
   by it; emit `Condition("project", "in", candidates[:20])`; no candidate ⇒ drop only the project condition (logged).
   Vocabulary: `SELECT project, district, count(*) FROM v_listings GROUP BY ALL` materialised once to a small JSON at
   index build (or Qdrant `facet(key="project", filter=district)` at query time). Covers LK01 (query ⊂ value),
   LK07 (value ⊂ query after stripping "Biên Hòa"), LK05 ("ecopark" ⊂ query). Same list feeds `to_sql`/`mask` (B4 stays one source).
2. `retrieve.py:137-138` + `ablation.py:145-146` — progressive relaxation before dropping all: 0 hits ⇒ drop
   `project`/`street` ⇒ drop exact numeric equalities (price/area/bedrooms) ⇒ drop all (`FILTER_DROPPED`); report which
   fields were relaxed in `note`.
3. `FILTER_PROMPT` (`filters.py:192`) — "deal_type = loại tin cần tìm; giá tổng 'X tỷ' ⇒ sale; chỉ 'rent' khi tìm tin cho thuê (giá/tháng)".
4. (contract change, api-contract §2) `min_bedrooms` for "từ N trở lên" — recall, not drop.

## Scope across the 100 questions

Measured: 4/21 ids questions drop (19%); 3/7 project-bearing lookups drop. 23 of 100 questions have `project` in `gt_sql`
(LK01-07, MH01-04, MH08-10, MH12, CP04, CP10, PS01, PS02, PS05, PS07, PS08, PS10); at the measured ~40-45% project
mismatch rate expect **~10 (range 8-14) of 100** to hit `filter_dropped` from project, plus ~1-2 from deal_type/street.
Refusal/value questions without a project or street are largely unaffected.
