"""C1 (khung ngữ cảnh bị chép vào câu trả lời đồ thị), C2 (model dự phòng ≠ giám khảo), C3 (router)."""

from unittest import mock

import pytest

from vre.config import load_settings
from vre.graph.text2cypher import GRAPH_ANSWER_SYSTEM
from vre.hybrid import router as R
from vre.llm.client import LLMResult
from vre.rag import generate


def test_graph_answer_prompts_forbid_echoing_the_wrapper():
    for system in (GRAPH_ANSWER_SYSTEM, R.GRAPH_CONTEXT_SYSTEM):
        assert generate.NO_ECHO_RULE in system
        assert "KHÔNG chép lại thẻ <tai_lieu>" in system and "KẾT QUẢ TRUY VẤN" in system


@pytest.mark.parametrize(("leaked", "expected"), [
    ("<tai_lieu>\nKẾT QUẢ TRUY VẤN:\nbathrooms: 2\n[Tin#30]\n</tai_lieu>", "bathrooms: 2\n[Tin#30]"),  # LK01
    ("KẾT QUẢ TRUY VẤN:\n<tai_lieu>\nmedian_curr: 86.32 *_m2_mil\n</tai_lieu>", "median_curr: 86.32 triệu đồng/m²"),  # TM01
    ("<tai_lieu>\nKết quả: 91.26\u202f*_m2_mil\n</tai_lieu>", "Kết quả: 91.26 triệu đồng/m²"),  # AG01
    ("KẾT QUẢ TRUY VẤN (3 dòng):\n[tai_lieu]Quận 1 cao nhất[/tai_lieu]", "Quận 1 cao nhất"),
    ("Giá trung vị là 86,3 triệu đồng/m² [Cypher].", "Giá trung vị là 86,3 triệu đồng/m² [Cypher]."),  # sạch: giữ nguyên
])
def test_strip_context_echo(leaked, expected):
    assert generate.strip_context_echo(leaked) == expected


def test_generate_sanitizes_answer():
    with mock.patch.object(generate, "ask_llm", return_value=LLMResult("<tai_lieu>\nĐáp [Tin#7]\n</tai_lieu>", None, 1, 1, True)):
        assert generate.generate("q", ["x"]).text == "Đáp [Tin#7]"


def test_generator_never_falls_back_to_judge(monkeypatch):
    s = load_settings(llm_provider="groq", llm_model="openai/gpt-oss-120b", judge_model="openai/gpt-oss-20b",
                      llm_fallback_models="openai/gpt-oss-20b,qwen/qwen3-32b")
    assert s.fallbacks_for(s.llm_model) == ["qwen/qwen3-32b"]
    assert s.fallbacks_for(s.judge_model) == ["qwen/qwen3-32b"]  # giám khảo cũng không lui về model sinh
    groq_default = load_settings(llm_provider="groq", llm_model="", judge_model="", llm_fallback_models=None)
    assert groq_default.judge_model in groq_default.llm_fallback_list
    assert groq_default.fallbacks_for(groq_default.llm_model) == []


def test_judge_config_records_effective_fallbacks(monkeypatch):
    from vre.eval import judge

    s = load_settings(llm_provider="groq", llm_model="openai/gpt-oss-120b", judge_model="openai/gpt-oss-20b",
                      llm_fallback_models="openai/gpt-oss-20b")
    info = judge.check_judge_config(s)
    assert info["generator_fallbacks"] == [] and info["configured_fallbacks"] == ["openai/gpt-oss-20b"]
    assert "bị bỏ" in info["note"]


def test_client_uses_filtered_fallbacks(monkeypatch):
    from vre.llm import client

    s = load_settings(llm_provider="groq", llm_model="gen", judge_model="judge", llm_fallback_models="judge,other")
    monkeypatch.setattr(client, "get_settings", lambda: s)
    tried = []

    class Fake:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(model, **_):
                    tried.append(model)
                    return "ok"
    monkeypatch.setattr(client, "client", lambda: Fake)
    assert client._create_with_fallback("gen", {}) == "ok" and tried == ["gen"]
    assert s.fallbacks_for("gen") == ["other"]


ROUTER_CASES = [  # câu kiểu benchmark (diễn đạt khác) → route vàng theo quy ước benchmark/questions.jsonl
    ("Mình thấy tin bán căn hộ 2PN 70m2 ở Masteri Thảo Điền giá 6 tỷ, căn đó có ban công không?", "basic_rag"),
    ("Nhà ở Bình Thạnh 4x15m, giá dưới 8 tỷ, đăng tháng 1/2026", "basic_rag"),
    ("Tỉnh nào có giá/m² đất trung vị thấp nhất?", "graph"),
    ("Tìm căn hộ 3PN ở dự án Vinhomes Ocean Park dưới 5 tỷ", "hybrid"),
    ("Căn hộ 75m2 2PN ở Thủ Đức nên rao bán bao nhiêu?", "valuation"),
]


def test_router_prompt_rules():
    p = R.ROUTER_SYSTEM
    assert "Lọc theo cột rồi liệt kê tin vẫn là basic_rag" in p
    assert "Liệt kê căn hộ 2 phòng ngủ ở Quận 7 dưới 4 tỷ\" -> graph" not in p  # few-shot cũ dạy sai
    for route in R.ROUTES:
        assert f"-> {route}" in p and f"- {route}:" in p


@pytest.mark.parametrize(("question", "route"), ROUTER_CASES)
def test_router_parses_schema(question, route):
    data = {"route": route, "reason": "r"}
    with mock.patch.object(R, "ask_llm", return_value=LLMResult("", data, 3, 2, True)) as ask:
        out = R.router(question)
    assert out["route"] == route and out["input_tokens"] == 3
    args, kwargs = ask.call_args
    assert question in args[0] and kwargs["system"] == R.ROUTER_SYSTEM and kwargs["output"] is R.Route
    assert set(R.Route.model_json_schema()["properties"]["route"]["enum"]) == set(R.ROUTES)
