"""listing_id (data-contract §2.1): tất định, ≤ 2⁵³−1, trùng chính xác cùng id, dò va chạm."""

import duckdb
import pandas as pd
import pytest

from vre.data.clean import ID_MASK, listing_ids, norm, prefix_keys
from vre.data.dedup import ListingIdCollision, check_collisions


def test_listing_id_deterministic_and_53_bit():
    names = pd.Series(["Bán nhà", "Bán  NHÀ ", "Khác"])
    descs = pd.Series(["Mô tả", "mô tả", "x"])
    ids, check = listing_ids(norm(names), norm(descs))
    again, _ = listing_ids(norm(names), norm(descs))
    assert (ids == again).all()
    assert ids[0] == ids[1] != ids[2]  # norm gộp hoa/thường + khoảng trắng
    assert (ids >= 0).all() and (ids <= ID_MASK).all() and ID_MASK == 2**53 - 1
    assert check[0] == check[1]


def test_prefix_key_uses_first_300_chars():
    base = "a" * 400
    keys = prefix_keys(pd.Series(["t", "t"]), pd.Series([base + "X", base + "Y"]))
    assert keys[0] == keys[1]


def test_collision_detected():
    con = duckdb.connect()
    con.register("frame", pd.DataFrame({"listing_id": [1, 1, 2], "key_hash2": [10, 11, 12],
                                        "source_shard": [0, 0, 0], "source_row": [0, 1, 2], "title": ["a", "b", "c"]}))
    con.execute("CREATE TEMP VIEW interim AS SELECT * FROM frame")
    with pytest.raises(ListingIdCollision, match="1 listing_id"):
        check_collisions(con)


def test_no_collision_for_exact_duplicates():
    con = duckdb.connect()
    con.register("frame", pd.DataFrame({"listing_id": [1, 1], "key_hash2": [10, 10],
                                        "source_shard": [0, 1], "source_row": [0, 0], "title": ["a", "a"]}))
    con.execute("CREATE TEMP VIEW interim AS SELECT * FROM frame")
    assert check_collisions(con) == 0
