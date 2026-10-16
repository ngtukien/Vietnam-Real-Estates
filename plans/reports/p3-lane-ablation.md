# Phase 3 A4 — retrieval ablation lane report

**Status:** DONE_WITH_CONCERNS
**Branch:** `feat/p3-ablation`, pushed (HEAD `202db99`). Commits `666beb6` (first version), `202db99` (scope cut).
**No PR:** none opened; the lead merges the lanes.

## What changed after the lead's scope cut
The chunk_size 256/400/512 subcorpus sweep is gone. The first commit had that code (subcorpus sampling, re-chunk, re-embed, ablation collections). The second commit deletes it, so no dead paths are left. No collections were ever created and nothing was embedded.

## Files
- `src/vre/eval/ablation.py` (230 lines)
- `tests/unit/test_ablation_corpus.py` (78 lines, 6 tests)

## Contract with the eval lane (CLI)
`run(settings: Settings, limit: int | None = None, *, out: Path | None = None, log=...) -> dict`
- Writes `results/ablation.json` and returns the same dict.
- `limit` caps the number of questions.
- `out` and `log` are keyword-only and meant for smoke runs and tests.

## Design
- **Runs:** all on alias `vre_chunks` (chunk 400, e5_large), retrieval only: dense / bm25 / hybrid × prefilter on/off.
- **Rerank:** gated on `hf_cached(RERANK_MODEL)`, which checks `$HF_HOME/hub/models--…/snapshots/*/config.json` and never touches the network. `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` is not in the cache, so the 6 rerank runs are left out and a `notes` entry says so. If the model gets cached, those 6 runs are added automatically and the total becomes 12.
- **Questions:** `gt_kind=ids` only, 27 in total. Loaded with `load_questions` and checked against `load_gt`.
- **Filters:** `parse_filters`, served from the `vre.llm` disk cache when possible. Calls that miss the cache are counted in `notes` as `filter_llm_calls`. If a filter returns 0 hits, the run drops the filter and searches again, which is the same fallback the live `retrieve` uses; each drop is recorded in `notes`.
- **Search code:** reuses `vre.rag.retrieve.search_dense`, `search_bm25`, `search_hybrid` and `rerank`. The rerank pool is `max(rerank_candidates, 10)`. Top-10 listings per question.
- **Scoring unit:** `dup_cluster_id`, per data-contract §2.2. Retrieved listings become clusters in rank order, deduplicated; the relevant set is `gt_dup_clusters`.
- **Metrics:**
  - recall@5 and recall@10 use `metrics.recall_at_k(capped=True)`, because constraint questions have 21 to 569 GT clusters.
  - MRR uses `metrics.mrr`.
  - nDCG@10 uses binary relevance. It calls `metrics.ndcg_at_k` if that exists, otherwise a local `_ndcg_local`. **Lead:** the local copy should be removed once the eval lane ships `ndcg_at_k`.
  - Latency is per question and covers retrieval only (query encode, Qdrant, rerank). `latency_ms_p50` is reported.
- **`ablation.json` shape:** follows §7.7. `corpus` = `{n_listings` from `index/manifest_vre_chunks_e5_large.json` (2,537,920), `n_gt_listings`, `seed: 42}`. Every run has `embedding: "e5_large"` and `chunk_size: 400`. There is one extra top-level key, `notes: [str]`.
- **Resume:** the file is written atomically after every run. A run is skipped if a run with the same name, same `n_questions` and same `corpus` is already there.
- **Cleanup:** `release_gpu()` runs in a `finally` block and clears the reranker and the CUDA cache.

## Verification (real outputs)
- `pytest -q tests/unit`, run through `run_heavy.ps1`: **266 passed**.
  - One earlier run without the lock had a single failure in `test_valuation_features.py::test_run_tabular_then_full_preserves_entries_and_valuate`. That file passed 9/9 on its own and in the locked full run. It is not my file; it looks flaky or order-dependent.
- `ruff check` on both files: clean.
- `mypy src/vre/eval/ablation.py --follow-imports=silent`: clean. Without that flag mypy reports errors in `vre/index/embed.py` and `vre/index/qdrant_store.py`, which are not mine.
- **Smoke:** `run(Settings(), limit=5, out=<scratch>)` through `run_heavy.ps1`, started with 6.1 GB commit free. Took **27.6 s** in total, including 4 filter LLM calls that missed the cache and loading the CPU query encoder (the first run took 13.9 s).

| run | R@5 | R@10 | MRR | nDCG@10 | p50 ms |
|---|---|---|---|---|---|
| full_dense | 0.933 | 0.933 | 0.900 | 0.879 | 341 |
| full_dense_prefilter | 1.000 | 1.000 | 1.000 | 1.000 | 299 |
| full_bm25 | 0.267 | 0.333 | 0.400 | 0.325 | 31 |
| full_bm25_prefilter | 0.600 | 0.600 | 0.600 | 0.600 | 6 |
| full_hybrid | 0.933 | 0.933 | 0.547 | 0.622 | 374 |
| full_hybrid_prefilter | 1.000 | 1.000 | 0.680 | 0.755 | 306 |

The smoke output went to scratch, not `results/`, so no partial `results/ablation.json` was written. GPU was not used: the query encoder runs on CPU and rerank was absent.

## Extrapolated full run
- Estimate: about **1.5–2.5 min**.
  - Measured query cost is about 1.36 s per question across the 6 runs, so 27 questions take about 37 s.
  - Encoder load is about 12 s.
  - Up to 22 more filter LLM calls are possible if they miss the cache, about 1 min.
- If the reranker is cached later, add 6 runs. Their cost was not measured.
- Command: `vre_pipeline.py ablation`, once the eval lane wires the CLI. Run it through `run_heavy.ps1`.

## Concerns
1. **Prefilter dropped on 2 of 5 smoke questions.** In 2 of the 5 questions, the LLM filter returned 0 hits and was dropped. The prefilter numbers therefore partly measure the "filter dropped, search unfiltered" path. Worth checking how place values like province or district get normalised. This is observational; I did not change filters.py, which is outside my lane.
2. **Rerank cannot be measured.** The model is not cached, so rerank is not part of the ablation.
3. **One key beyond §7.7.** `notes` is a top-level key the contract does not list. The README and notebook readers should ignore unknown keys.

## Next steps
- Eval lane: wire `vre_pipeline.py ablation` to call `ablation.run(settings, limit)`.
- Lead: once the eval lane adds `metrics.ndcg_at_k`, delete `_ndcg_local` (it is used automatically when present).
