# p3: e5-large encoder crash fix

**Status:** BLOCKED on verification. The code is done, committed and pushed. The heavy-test suite has not run yet.
**Worktree / branch:** D:\Vietnam-Real-Estates-wt\encoderfix, `feat/p3-encoderfix`. Commit `d0c45fac8aa14bc36f08fa3b67e84aca132fcbff`, pushed to origin. No PR opened (not requested). Working tree clean.

## What changed (fix proposals 1–3 from p3-encoder-crash.md)
- **New `src/vre/index/weights.py`** (86 lines). Holds `_weights_path`, `load_state_fp16`, `SLICE_BYTES` and `_NUMPY_DTYPES`, moved verbatim from query_encoder. Adds `build_model_fp16()`: the old `QueryEncoder.__init__` body, from the pooling check and AutoConfig through the is_meta check. It returns `model.eval()`. It is now the only e5-large loader.
- **`src/vre/rag/query_encoder.py`**: `self.model = build_model_fp16()`. Removed the json, struct and typing.Any imports and the AutoConfig/AutoModel imports. Updated the docstring line. `dim` and `prefix` still come from EMBED_MODELS. The pooling check moved into `build_model_fp16`.
- **`src/vre/index/embed.py`**: `self.model = build_model_fp16().to(device).eval()`, with a comment citing the crash report. Dropped the `AutoModel` import. The CUDA cap (`set_per_process_memory_fraction`, VRAM_CAP_GB=4.0) is unchanged and still set right after the model moves to the device.
- **`tests/gpu/test_embed_parity.py`**: the reference stays an independent transformers loader and now passes `model_kwargs={"disable_mmap": True}`.
- **`scripts/vre_env.ps1`**: `$env:OPENBLAS_NUM_THREADS = '4'`.
- **`tests/conftest.py`**: `os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")`. Deviation: it sits next to the existing `VRE_LOAD_DOTENV` line, after `import pytest`, not literally at the top of the file. Reason: placing it at the top would trigger ruff E402 on the imports below it. Nothing above that line imports numpy, so the effect is the same.
- **New `tests/unit/test_encoder_weights.py`** (6 tests):
  - Grep test: `load_state_fp16`, `build_model_fp16` and `_weights_path` are each defined only in weights.py, scanning src/, tests/, scripts/ and notebooks/.
  - embed.py and query_encoder.py use `build_model_fp16` and contain no `AutoModel.from_pretrained`, `from_file`, `safe_open` or `load_file`.
  - Mock-level test: builds a tiny XLM-R safetensors checkpoint, then patches each of these to raise: `torch.UntypedStorage.from_file`, `safetensors.safe_open`, `safetensors.torch.safe_open`, `safetensors.torch.load_file`, `mmap.mmap` and `AutoModel.from_pretrained`. `build_model_fp16` must still load the model in eval mode, with fp16 non-meta parameters equal to `ref.half()`, and run a forward pass.

## Verification
| Check | Result |
|---|---|
| ruff (all 7 touched files) | PASS: "All checks passed!" |
| mypy on weights.py, embed.py, query_encoder.py | 1 error at `embed.py:106` (`prefix + t`, `str \| int`). **Pre-existing:** the same error is at HEAD `embed.py:105` (I ran mypy on `git show HEAD:` and got it). The line is untouched, so I left it. weights.py and query_encoder.py are clean. |
| `pytest -q tests/unit` | **NOT RUN.** The run_heavy queue was blocked (see below). |
| integration `test_index_complete.py` ×3 | NOT RUN |
| gpu `test_embed_parity.py` ×3 (cosine ≥ 0.999) | NOT RUN |
| `pytest -q tests/api` | NOT RUN |

## Blocker
- For about 40 minutes (23:2x to about 00:05), free commit was flat at 4.7–4.9 GB, and run_heavy needs at least 6 GB.
- No project python was running. vmmemWSL held 15.2 GB of commit; the commit limit is 44.4 GB.
- The filterfix lane's run_heavy (PID 4192, `tests/unit tests/api`) holds `Global\vre_heavy_job` while it waits for commit, so my job queued behind it.
- I tried the documented runtime reclaim (`docker run --rm --privileged alpine … drop_caches`). The permission classifier denied it, and I did not work around that.
- I stopped my own queued waiter (PID 39100). I did not touch PID 4192, which belongs to the other lane.

## Remaining steps for whoever resumes
1. Free commit to at least 6 GB. That needs coordinator or user action: drop_caches or a Docker restart, or run run_heavy with a lower `-MinFreeCommitGb` (for example 4.5) once filterfix's job finishes.
2. From the worktree, run each of these through run_heavy:
   - `-m pytest -q tests/unit`
   - `-X faulthandler -m pytest -q tests/integration/test_index_complete.py`, 3 times
   - `-X faulthandler -m pytest -q -m gpu tests/gpu/test_embed_parity.py`, 3 times
   - `-m pytest -q tests/api`
3. Parity risk to watch: the encoder now casts fp32 weights to fp16 element by element, instead of going through transformers' `dtype=fp16` path. The values should be bit-identical, since both cast fp32 to fp16, but that is unverified until the gpu test runs.

Helper I used: C:\Users\Nghaiz\AppData\Local\Temp\claude\d--Vietnam-Real-Estates\78375574-07f7-4335-811e-4c4bc63b3baf\scratchpad\rh.ps1 `-Args1 "<python args>"`. It runs `cd` into the worktree, then `& $VRE_PY <args>` under run_heavy.
