"""Dedup tầng 1+2 (data-contract §2.2, §3.3): chọn đại diện, lý do dup_map, n_raw_rows."""

import duckdb
import pandas as pd

from vre.data.dedup import build_dup_map

COLS = ["source_shard", "source_row", "listing_id", "prefix_key", "drop_reason", "published_at", "n_nonnull"]


def _run(rows):
    con = duckdb.connect()
    con.register("frame", pd.DataFrame(rows, columns=COLS))
    con.execute("CREATE TEMP VIEW interim AS SELECT * FROM frame")
    build_dup_map(con)
    reasons = {(s, r): (reason, kept) for s, r, reason, kept in
               con.execute("SELECT source_shard, source_row, reason, kept_listing_id FROM dup_map").fetchall()}
    kept = dict(con.execute("SELECT listing_id, n_raw_rows FROM kept").fetchall())
    return reasons, kept


def test_exact_and_prefix_dups_choose_newest_representative():
    t_old, t_new = pd.Timestamp("2025-06-01"), pd.Timestamp("2025-09-01")
    rows = [
        (0, 0, 10, 100, None, t_old, 5),   # id 10, nhóm prefix 100
        (0, 1, 10, 100, None, t_new, 5),   # trùng chính xác id 10, mới hơn ⇒ đại diện id 10
        (0, 2, 11, 100, None, t_old, 9),   # cùng prefix, cũ hơn ⇒ prefix_dup
        (1, 0, 12, 200, None, None, 3),    # đơn lẻ
        (1, 1, 13, 300, "dropped_no_text", None, 0),
    ]
    reasons, kept = _run(rows)
    assert reasons[(0, 1)] == ("kept", 10)
    assert reasons[(0, 0)] == ("exact_dup", 10)
    assert reasons[(0, 2)] == ("prefix_dup", 10)
    assert reasons[(1, 0)] == ("kept", 12)
    assert reasons[(1, 1)] == ("dropped_no_text", None)
    assert kept == {10: 3, 12: 1}


def test_tie_breaks_on_nonnull_then_listing_id():
    t = pd.Timestamp("2025-07-01")
    rows = [(0, 0, 21, 5, None, t, 4), (0, 1, 20, 5, None, t, 4), (0, 2, 30, 6, None, t, 2), (0, 3, 31, 6, None, t, 7)]
    reasons, _ = _run(rows)
    assert reasons[(0, 1)][0] == "kept" and reasons[(0, 0)] == ("prefix_dup", 20)
    assert reasons[(0, 3)][0] == "kept" and reasons[(0, 2)] == ("prefix_dup", 31)


def test_every_row_has_a_reason_and_kept_count_matches():
    rows = [(0, i, 100 + i % 3, i % 2, None, None, 1) for i in range(12)]
    reasons, kept = _run(rows)
    assert len(reasons) == 12
    assert sum(1 for r, _ in reasons.values() if r == "kept") == len(kept)
    assert sum(kept.values()) == 12
