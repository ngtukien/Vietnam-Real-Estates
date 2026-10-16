# p3 — e5-large encoder load crash: root cause

**Status:** DONE (root cause proven; fix proposed, not applied; no tracked file edited)
**Worktree:** D:\Vietnam-Real-Estates-wt\int (feat/p3-integration). Scratch scripts and logs: C:\Users\Nghaiz\AppData\Local\Temp\claude\d--Vietnam-Real-Estates\78375574-07f7-4335-811e-4c4bc63b3baf\scratchpad\

## Root cause
When system free commit is below about 2.24 GB (the size of model.safetensors), Windows refuses torch's copy-on-write file mapping with ERROR_COMMITMENT_LIMIT (1455). torch never checks that MapViewOfFile returned NULL and hands back a 2.24 GB storage with a NULL base. safetensors 0.8 slices every tensor at NULL + data_offset, and the first read is an access violation at 0xc1e8.

## Causal chain, with evidence
1. **Load path.** src/vre/index/embed.py:32 calls AutoModel.from_pretrained. In transformers 5.14.1, modeling_utils.py:4577-4578 opens the file with safe_open(file, framework="pt", device="cpu", backend="mmap"). The safetensors 0.8.0 "mmap" backend for pt is torch.UntypedStorage.from_file(path, shared=False, nbytes=2239611368). Each tensor is then a slice storage[0xc1e8+begin:...], where 0xc1e8 = 8 + header bytes, i.e. the start of the data section. Trace (r4.log): from_file -> ptr=0x271fec80000, slices ptr=0x271fec8c1e8, ...
2. **torch on Windows** (aten/src/ATen/MapAllocator.cpp, _WIN32 regular-file branch, flags==0, which is shared=False): CreateFileMappingW(hfile, NULL, PAGE_WRITECOPY, ...), then base_ptr_ = MapViewOfFile(hmfile, FILE_MAP_COPY, 0, 0, 0) with **no NULL check**. The SHAREDMEM branch does check.
3. **Commit exhaustion gives a NULL storage, with no exception** (ballast.py, r6.log). Untouched VirtualAlloc(MEM_COMMIT) ballast lowered free commit in steps. from_file stayed OK down to about 2.1 GB, then: ballastGB=14.25 sysFreeCommitGB=2.03 -> INVALID(data=NULL) GetLastError=1455. Free commit kept bouncing back while the pagefile grew on demand; the failure only comes once it cannot grow, which is why the crash is intermittent. The same NULL storage also appeared naturally on the 2nd load in one process (r4.log: "Attempted to access the data pointer on an invalid python storage." raised by from_file).
4. **The crash** (ctypes vectored exception handler, identical in 2 of 2 runs, r3.log): code=0xc0000005 at torch_cpu.dll+0x8e4a279, access=read, target=0xc1e8. Disassembly at that address: call 0x18a156bf0 (data ptr), then movzx edx, BYTE PTR [rax], inside a ScalarType switch. That is the NULL base + 0xc1e8. Python stack (faulthandler, real pytest run, r1.log): torch/storage.py:471 __getitem__ <- transformers/core_model_loading.py:1215 _materialize_copy <- :1239 _job (ThreadPool worker) <- embed.py:32 <- embed.py:119 get_encoder <- tests/integration/test_index_complete.py:56.
5. **Why commit is short even though run_heavy saw >= 6 GB free.** Each process pre-commits a lot before loading a single weight:
   - numpy import takes 0.79 GB at default OpenBLAS (32 logical CPUs), versus 0.12 GB with OPENBLAS_NUM_THREADS=4.
   - After import torch the process is at 1.55 GB.
   - Machine-wide, vmmemWSL holds about 16.5 GB.
   - In r5, free commit was 1.70 GB right after import torch, and the next from_pretrained crashed.
   - During the pytest crash (mon1.log), free commit went 6.3 -> 2.45 GB while python grew to 4.7 GB.

## Ruled out
- **Library change since Phase 2.** torch 2.13.0+cu126, transformers 5.14.1, safetensors 0.8.0, sentence-transformers 5.7.0 and numpy 2.5.1 were all installed 2026-10-14 23:13-23:37, before the Phase-2 merge (2026-10-15 21:06). In the requested paths, the only diff since e6557dd is the new query_encoder.py. Phase 2 likely had more commit headroom. That part is UNVERIFIED: no Phase-2 commit measurements exist.
- **Two loaders in one process.** Neither test imports query_encoder, and a 10-line script that only calls AutoModel.from_pretrained crashes (r2.log: 1 of 3 async runs).
- **CUDA / set_per_process_memory_fraction.** The crash happens before .to("cuda"), and CPU-only scripts crash.
- **HF_DEACTIVATE_ASYNC_LOAD.** It only changes which thread reads first. from_file runs in the main thread at safe_open. Sync went 0 of 3 here, but crashed in p3-fixups. It is not a fix.
- **Use-after-close of the safe_open handles.** They are closed only after convert_and_load returns, and the main thread was still submitting jobs at crash time.
- **0xC0000409 in tests/gpu.** Not reproduced here. My inference is the same NULL-base storage reaching a fast-fail-checked path in the same load step. UNVERIFIED.

## Fix proposal (minimal, SSOT)
1. **Stop loading the encoder through the transformers/safetensors mmap path.** Reuse the sliced readinto loader that the API already runs in production (query_encoder.load_state_fp16). Its peak is about the fp16 model (1.1 GB) plus one 64 MB slice, it makes no COW view, and a commit shortage raises Python MemoryError instead of an AV. Move it under index so vre.index does not depend on vre.rag:
   - New src/vre/index/weights.py: move _weights_path and load_state_fp16 there, plus a new build_model_fp16(). It is the current QueryEncoder.__init__ body from AutoConfig.from_pretrained through the is_meta check, and returns model.eval().
   - src/vre/rag/query_encoder.py: self.model = build_model_fp16().
   - src/vre/index/embed.py:32-33: replace the AutoModel.from_pretrained(...).to(device).eval() call with `self.model = build_model_fp16().to(device).eval()` and drop the unused AutoModel import.
2. **tests/gpu/test_embed_parity.py:28.** The reference must stay an independent loader, so keep transformers but skip the COW mmap: SentenceTransformer(model_name(), device="cuda", local_files_only=True, model_kwargs={"disable_mmap": True}). With disable_mmap, modeling_utils.py:4569-4571 reads the file into bytes, so low commit raises a clean MemoryError.
3. **Saves 0.67 GB commit per python process:**
   - `$env:OPENBLAS_NUM_THREADS = '4'` in scripts/vre_env.ps1 (same as Dockerfile.api).
   - `os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")` at the top of tests/conftest.py.
4. **Optional upstream torch issue:** the Windows MapAllocator regular-file branch has no NULL check on MapViewOfFile. Not filed.

## Verification plan
1. Log Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit' before and after each run (scratchpad mon.ps1).
2. Under run_heavy.ps1, run 3x each: python -X faulthandler -m pytest -q tests/integration/test_index_complete.py, and -m gpu tests/gpu/test_embed_parity.py. Expect passes, with cosine >= 0.999; this also guards that the sliced loader matches ST fp32.
3. Fail-loud check: run ballast.py (floor 1.2 GB) using build_model_fp16() instead of from_file. It must raise MemoryError/RuntimeError, never exit -1073741819.
4. pytest -q tests/unit, plus an API /api/chat smoke test, because query_encoder changes.

## Note for the coordinator
The ~22:49 machine-wide free-commit dip (1.2-2.0 GB) was most likely my diagnostic ballast run. It committed untouched memory down to a 1.2 GB floor for about 1 minute and then released it (released; sysFreeCommitGB=17.48). It touched no RAM, but other processes could have hit commit failures in that window. I ran no further ballast after it.
