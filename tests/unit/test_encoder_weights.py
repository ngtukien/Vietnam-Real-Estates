"""`vre.index.weights` là loader e5-large DUY NHẤT và không bao giờ đi đường mmap/`from_file` (p3-encoder-crash.md):
safetensors mmap COW trên Windows commit thấp trả storage base NULL ⇒ access violation thay vì MemoryError."""

import mmap
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "src" / "vre" / "index" / "weights.py"
ENCODERS = (REPO_ROOT / "src" / "vre" / "index" / "embed.py", REPO_ROOT / "src" / "vre" / "rag" / "query_encoder.py")
SCAN_DIRS = ("src", "tests", "scripts", "notebooks")


def _py_files():
    for name in SCAN_DIRS:
        yield from (REPO_ROOT / name).rglob("*.py")


@pytest.mark.parametrize("func", ["load_state_fp16", "build_model_fp16", "_weights_path"])
def test_weights_module_is_single_loader(func):
    pattern = re.compile(rf"^\s*def {func}\(", re.MULTILINE)
    defined = sorted(str(p.relative_to(REPO_ROOT)) for p in _py_files() if pattern.search(p.read_text(encoding="utf-8")))
    assert defined == [str(WEIGHTS.relative_to(REPO_ROOT))], f"{func} phải chỉ định nghĩa ở weights.py, thấy: {defined}"


@pytest.mark.parametrize("path", ENCODERS, ids=lambda p: p.name)
def test_encoders_use_shared_loader_not_from_pretrained(path):
    source = path.read_text(encoding="utf-8")
    assert "build_model_fp16" in source
    for banned in ("AutoModel.from_pretrained", "from_file", "safe_open", "load_file"):
        assert banned not in source, f"{path.name} không được nạp trọng số qua {banned} (mmap COW)"


@pytest.fixture
def tiny_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from safetensors.torch import save_file

    config = transformers.XLMRobertaConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                           intermediate_size=32, max_position_embeddings=40)
    torch.manual_seed(0)
    reference = transformers.XLMRobertaModel(config).eval()
    state = {k: v.detach().contiguous() for k, v in reference.state_dict().items()}
    path = tmp_path / "model.safetensors"
    save_file(state, str(path))
    return config, state, path


def test_build_model_fp16_never_maps_the_file(tiny_checkpoint, monkeypatch):
    import safetensors
    import safetensors.torch
    import torch
    import transformers

    from vre.index import weights

    config, state, path = tiny_checkpoint

    def forbidden(*_args, **_kwargs):
        raise AssertionError("loader e5-large không được dùng mmap / from_file / safe_open / from_pretrained")

    monkeypatch.setattr(weights, "_weights_path", lambda: str(path))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: config)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", forbidden)
    monkeypatch.setattr(torch.UntypedStorage, "from_file", forbidden)
    monkeypatch.setattr(safetensors, "safe_open", forbidden)
    monkeypatch.setattr(safetensors.torch, "safe_open", forbidden)
    monkeypatch.setattr(safetensors.torch, "load_file", forbidden)
    monkeypatch.setattr(mmap, "mmap", forbidden)

    model = weights.build_model_fp16()

    assert not model.training
    params = dict(model.named_parameters())
    assert params and all(p.dtype == torch.float16 and not p.is_meta for p in params.values())
    for name, value in state.items():
        if name in params:
            assert torch.equal(params[name], value.half()), name
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([[0, 5, 7, 2]]), attention_mask=torch.ones(1, 4, dtype=torch.long))
    assert out.last_hidden_state.shape == (1, 4, 16)
