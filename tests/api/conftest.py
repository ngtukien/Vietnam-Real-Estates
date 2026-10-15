"""Fixture test API (`VRE_FAKE_BACKENDS=1`, không cần dịch vụ). Tiện ích dùng chung: `api_helpers.py`."""

from __future__ import annotations

from typing import Any

import pytest
from api_helpers import make_app


@pytest.fixture
def app() -> Any:
    return make_app()
