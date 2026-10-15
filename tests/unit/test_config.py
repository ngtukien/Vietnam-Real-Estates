"""vre.config + vre.envguard: khoá §9.1 đủ và đúng mặc định, override bằng env, envguard chặn đường dẫn ngoài ổ D."""

from pathlib import Path

import pytest

from vre import envguard
from vre.config import EMBED_MODEL_SLUG, EMBED_MODELS, load_settings

EXPECTED_DEFAULTS = {
    "duckdb_threads": 12, "duckdb_memory_gb": 12, "ingest_workers": 8,
    "chunk_size": 400, "chunk_overlap": 100, "max_chunks_per_listing": 64, "header_max_tokens": 96, "part_size": 50000,
    "embed_max_length": 512, "embed_token_budget": 32768, "embed_max_batch": 256, "embed_min_token_budget": 4096,
    "tokenize_workers": 2, "tokenize_prefetch_parts": 3, "bm25_workers": 4,
    "qdrant_url": "http://127.0.0.1:6333", "qdrant_grpc_port": 6334, "qdrant_upload_parallel": 1, "qdrant_upload_batch": 256,
    "qdrant_max_indexing_threads": 12, "qdrant_indexing_threshold": 20000, "qdrant_green_timeout_s": 10800,
    "collection": "vre_chunks_e5_large", "collection_alias": "vre_chunks", "dryrun_collection": "vre_dryrun", "dryrun_parts": 2,
    "xgb_nthread": 12, "xgb_device_tabular": "cpu", "xgb_device_full": "cuda", "rent_min_rows": 20000,
    "pca_dim": 32, "pca_fit_sample": 300000, "min_free_disk_gb_start": 45, "min_free_disk_gb_part": 5,
    "night_max_attempts": 3, "night_retry_wait_s": 120, "handoff_poll_s": 300,
}
PATH_PROPS = {
    "raw_dir": "data/raw/vietnam-real-estates", "clean_path": "data/clean/listings.parquet",
    "dup_map_path": "data/clean/dup_map.parquet", "index_dir": "index", "results_dir": "results", "models_dir": "models",
    "logs_dir": "logs", "handoff_dir": "results/handoff", "features_dir": "data/features", "cache_dir": ".cache",
    "compose_file": "docker-compose.yaml",
}


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("VRE_DUCKDB_THREADS", "VRE_INGEST_WORKERS", "VRE_DUCKDB_MEMORY_GB", "LLM_MODEL", "VRE_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VRE_LOAD_DOTENV", "0")
    return monkeypatch


def test_every_contract_key_has_its_default(clean_env):
    settings = load_settings()
    for key, value in EXPECTED_DEFAULTS.items():
        assert getattr(settings, key) == value, key


def test_paths_are_absolute_under_data_root(clean_env, tmp_path):
    settings = load_settings(data_root=tmp_path)
    for prop, rel in PATH_PROPS.items():
        path = getattr(settings, prop)
        assert path.is_absolute() and path == tmp_path / Path(rel), prop


def test_env_override_with_prefix(clean_env):
    clean_env.setenv("VRE_DUCKDB_THREADS", "4")
    clean_env.setenv("LLM_MODEL", "x/y")
    settings = load_settings()
    assert settings.duckdb_threads == 4
    assert settings.llm_model == "x/y"


def test_llm_defaults_follow_provider(clean_env):
    settings = load_settings(llm_provider="groq")
    assert settings.llm_model == "openai/gpt-oss-120b"
    assert settings.judge_model == "openai/gpt-oss-20b"
    assert settings.llm_fallback_list == ["openai/gpt-oss-20b"]


def test_single_embedding_model():
    assert EMBED_MODEL_SLUG == "e5_large"
    assert EMBED_MODELS == {"e5_large": ("intfloat/multilingual-e5-large", 1024, "mean", "passage: ", "query: ")}


def test_envguard_fills_defaults_on_d():
    env: dict[str, str] = {}
    envguard.apply(env)
    assert all(env[name].startswith("D:") for name in envguard.CACHE_VARS)
    assert env["PYTHONUTF8"] == "1"


def test_envguard_raises_when_a_path_is_on_c():
    env = {"TMP": r"C:\Users\x\AppData\Local\Temp"}
    with pytest.raises(envguard.EnvPathError, match="TMP"):
        envguard.apply(env)


def test_envguard_violations_lists_only_bad_vars():
    assert envguard.violations({"HF_HOME": r"D:\x", "TORCH_HOME": r"C:\y", "OTHER": r"C:\z"}) == {"TORCH_HOME": r"C:\y"}
