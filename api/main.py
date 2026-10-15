"""App factory FastAPI (api-contract §0). Chạy:

    & $VRE_PY -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --timeout-keep-alive 5

Khi chưa đặt `VRE_API_TOKEN`, chỉ client loopback được gọi `/api/*` (trừ khi `VRE_API_ALLOW_REMOTE_WITHOUT_TOKEN=1`).
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import TypeAdapter
from pydantic.json_schema import models_json_schema
from starlette.middleware.cors import CORSMiddleware

from api import API_VERSION
from api.deps import ApiSettings, ChatSlots, RateLimiter, get_api_settings
from api.errors import install_handlers
from api.middleware import GuardMiddleware, RequestIdMiddleware
from api.routers import chat, compare, geo, graph, listings, meta, stats, valuate
from api.schemas import EXTRA_SCHEMA_ENUMS, EXTRA_SCHEMA_MODELS
from api.services.backend import Backend, make_backend

log = logging.getLogger("vre.api")
REF_TEMPLATE = "#/components/schemas/{model}"
for _key, _value in {"TQDM_DISABLE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(_key, _value)


def _startup_checks(settings: ApiSettings) -> None:
    """Kiểm manifest chỉ mục (real mode). Không nạp encoder: facade lõi tự nạp ở câu hỏi đầu (api-contract §7)."""
    if settings.fake_backends:
        log.info("VRE_FAKE_BACKENDS=1 — dữ liệu từ api/fixtures, không gọi Qdrant/Neo4j/LLM")
        return
    from vre.config import load_settings

    vs = load_settings()
    manifest = vs.index_dir / f"manifest_{vs.collection}.json"
    if not manifest.is_file():
        log.warning("Thiếu %s — /api/info sẽ trả 503", manifest.name)
    if not settings.api_token:
        log.warning("Chưa đặt VRE_API_TOKEN: chỉ nhận kết nối loopback%s",
                    " (đã tắt bằng VRE_API_ALLOW_REMOTE_WITHOUT_TOKEN)" if settings.api_allow_remote_without_token else "")


def create_app(settings: ApiSettings | None = None, backend: Backend | None = None) -> FastAPI:
    settings = settings or get_api_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _startup_checks(settings)
        yield

    app = FastAPI(title="Vietnam Real Estates API", version=API_VERSION, lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url="/api/openapi.json", separate_input_output_schemas=False)
    app.state.settings = settings
    app.state.backend = backend or make_backend(settings)
    app.state.slots = ChatSlots(settings.chat_max_concurrent, settings.chat_queue_timeout_s)
    app.state.chat_limiter = RateLimiter(settings.chat_rate_per_min)
    app.state.api_limiter = RateLimiter(settings.api_rate_per_min)
    install_handlers(app)
    for module in (meta, chat, compare, valuate, stats, geo, graph, listings):
        app.include_router(module.router)

    # add_middleware chèn vào ĐẦU ngăn xếp ⇒ thêm theo thứ tự ngược: middleware thêm sau cùng chạy trước nhất.
    app.add_middleware(GuardMiddleware, settings=settings, chat_limiter=app.state.chat_limiter,
                       api_limiter=app.state.api_limiter)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-API-Key"], expose_headers=["X-Request-Id", "Retry-After"])
    app.add_middleware(RequestIdMiddleware)

    def openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(title=app.title, version=app.version, routes=app.routes,
                             separate_input_output_schemas=False)
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        _, extra = models_json_schema([(m, "validation") for m in EXTRA_SCHEMA_MODELS], ref_template=REF_TEMPLATE)
        for name, sub in extra.get("$defs", {}).items():
            components.setdefault(name, sub)
        for enum in EXTRA_SCHEMA_ENUMS:
            components.setdefault(enum.__name__, TypeAdapter(enum).json_schema(ref_template=REF_TEMPLATE))
        # 422 của FastAPI được map về 400 (api-contract §5) ⇒ bỏ khỏi tài liệu.
        for path in schema.get("paths", {}).values():
            for op in path.values():
                op.get("responses", {}).pop("422", None)
        for name in ("HTTPValidationError", "ValidationError"):
            components.pop(name, None)
        app.openapi_schema = schema
        return schema

    app.openapi = openapi  # type: ignore[method-assign]
    return app


app = create_app()
