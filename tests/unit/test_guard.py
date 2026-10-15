"""vre.graph.guard: chặn Cypher ghi/quản trị và ép LIMIT (port từ tests/test_presentation_notebooks.py)."""

import pytest

from vre.config import get_settings
from vre.graph import guard


def test_read_only_query_passes():
    assert guard.is_read_only("MATCH (l:Listing) RETURN count(l) AS so_tin")


@pytest.mark.parametrize("bad", [
    "MATCH (l) DETACH DELETE l", "MATCH (l) SET l.x = 1", "CALL db.labels()",
    "LOAD CSV FROM 'x' AS r RETURN r", "merge (n:X)",
    "SHOW USERS", "USE system MATCH (n) RETURN n", "MATCH (n) FOREACH (x IN [1] | SET n.a = x)",
])
def test_write_and_admin_cypher_is_blocked(bad):
    assert not guard.is_read_only(bad)
    assert guard.FORBIDDEN.search(bad)


def test_ensure_limit_appends_when_missing():
    assert guard.ensure_limit("MATCH (n) RETURN n;", 50) == "MATCH (n) RETURN n\nLIMIT 50"


def test_ensure_limit_keeps_existing_limit():
    assert guard.ensure_limit("MATCH (n) RETURN n LIMIT 5", 50) == "MATCH (n) RETURN n LIMIT 5"


def test_limit_inside_with_does_not_cap_final_return():
    assert guard.ensure_limit("MATCH (n) WITH n LIMIT 3 RETURN n", 50).endswith("LIMIT 50")


def test_ensure_limit_defaults_to_settings_max_rows():
    expected = f"MATCH (n) RETURN n\nLIMIT {get_settings().cypher_max_rows}"
    assert guard.ensure_limit("MATCH (n) RETURN n") == expected
