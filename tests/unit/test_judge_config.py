"""Bug B2: eval từ chối chạy khi giám khảo = mô hình sinh, trừ khi VRE_ALLOW_SAME_JUDGE=1 (ghi vào eval_summary.llm)."""

import pytest

from vre.config import load_settings
from vre.eval import judge


def test_same_judge_is_refused(monkeypatch):
    monkeypatch.delenv(judge.ALLOW_SAME_JUDGE_ENV, raising=False)
    settings = load_settings(llm_provider="openai", llm_model="gpt-x", judge_model="")  # openai: không có judge mặc định
    assert settings.judge_model == settings.llm_model
    with pytest.raises(RuntimeError, match="JUDGE_MODEL"):
        judge.check_judge_config(settings)


def test_same_judge_allowed_by_env_is_recorded(monkeypatch):
    monkeypatch.setenv(judge.ALLOW_SAME_JUDGE_ENV, "1")
    info = judge.check_judge_config(load_settings(llm_provider="openai", llm_model="gpt-x", judge_model="gpt-x"))
    assert info["same_judge_allowed"] is True and info["generator"] == info["judge"] == "gpt-x"


def test_distinct_judge_passes(monkeypatch):
    monkeypatch.setenv(judge.ALLOW_SAME_JUDGE_ENV, "1")
    info = judge.check_judge_config(load_settings(llm_provider="groq", llm_model="", judge_model=""))
    assert info["generator"] == "openai/gpt-oss-120b" and info["judge"] == "openai/gpt-oss-20b"
    assert info["same_judge_allowed"] is False
    assert "note" in info  # judge nằm trong danh sách fallback của generator ⇒ ghi chú


def test_judge_system_uses_question_tolerance():
    assert "15%" in judge.judge_system(0.15) and "30%" in judge.judge_system(0.15)
    assert "5%" in judge.JUDGE_SYSTEM and "20%" in judge.JUDGE_SYSTEM
