"""Lượt chạy dài: chunk + tokenize → BM25 → embed GPU → upload Qdrant → HNSW green → alias → kiểm chứng → PCA.

Điều phối (data-contract §4.1, §7.5): pool tokenize ghi `chunks/` + `tok/` cho mọi part, pool BM25 theo sau; GPU embed ở luồng chính,
một luồng nạp trước tối đa `tokenize_prefetch_parts` part token vào RAM; luồng upload chờ `bm25_avgdl` (một lần, khi mọi part đã `bm25`)
rồi bám theo embed. Checkpoint sau mỗi part mỗi stage; chạy lại cùng lệnh để tiếp tục. Mã thoát: 0 = completed, 20 = lỗi cần người.
"""

import queue
import threading
import time
import traceback
from concurrent.futures import as_completed

from vre.index import bm25 as bm25_mod
from vre.index import qdrant_upload, tokenize
from vre.pipeline import stages
from vre.pipeline.report import Report, finish
from vre.pipeline.state import layout_for, log

EXIT_OK, EXIT_NEEDS_HUMAN = 0, 20
POLL_S = 5


class NightRun:
    def __init__(self, settings, dry_run: bool):
        self.settings, self.dry_run = settings, dry_run
        self.layout = layout_for(settings, dry_run)
        self.report = Report(self.layout, settings)
        self.stop, self.failure = threading.Event(), None
        self.avgdl_ready = threading.Event()
        self.ledger = self.encoder = self.qc = None
        self.n_parts, self.step = 0, "checks"
        self.upload_counters = {"upload_retries": 0}

    def fail(self, step: str, exc: BaseException, part: int | None = None) -> None:
        self.report.error(step, exc, part)
        log(f"LỖI ở {step} (part {part}): {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        self.failure = self.failure or exc
        self.stop.set()

    def _wait(self, predicate) -> bool:
        while not predicate():
            if self.stop.is_set():
                return False
            time.sleep(POLL_S)
        return True

    def cpu_stages(self, tok_pool, bm25_pool, chunked: dict) -> None:
        """Chunk mọi part (ghi đĩa, không giữ RAM), mỗi part chunk xong thì đẩy sang BM25; xong hết ⇒ `bm25_avgdl`."""
        ledger, layout, parts, step = self.ledger, self.layout, range(self.n_parts), "chunk"
        try:
            self.report.begin("chunk", self.n_parts)
            self.report.begin("bm25", self.n_parts)
            pending = []

            def submit_bm25(part):
                pending.append(bm25_pool.submit(stages.bm25_task, (layout, part)))
                pending[-1].add_done_callback(lambda f: f.exception() is None and (
                    ledger.mark(part, "bm25"), self.report.progress("bm25", ledger.count("bm25"))))

            for part in parts:
                if ledger.done(part, "chunked"):
                    chunked[part].set()
                    if not ledger.done(part, "bm25"):
                        submit_bm25(part)
            todo = [(self.settings, layout, p) for p in parts if not ledger.done(p, "chunked")]
            for future in as_completed([tok_pool.submit(stages.chunk_task, args) for args in todo]):
                out = future.result()  # worker chết ⇒ BrokenProcessPool, không treo
                ledger.mark(out["part"], "chunked", n_chunks=out["n_chunks"])
                chunked[out["part"]].set()
                self.report.progress("chunk", ledger.count("chunked"))
                if not ledger.done(out["part"], "bm25"):
                    submit_bm25(out["part"])
            self.report.end("chunk")
            step = "bm25"
            for future in pending:
                future.result()
            if ledger.state["bm25_avgdl"] is None:
                ledger.set(bm25_avgdl=bm25_mod.corpus_avgdl(layout, list(parts)))
            self.report.end("bm25", note=f"avgdl={ledger.state['bm25_avgdl']:.2f}")
            self.avgdl_ready.set()
        except BaseException as exc:
            self.fail(step, exc)

    def feed(self, chunked: dict, tokens: queue.Queue) -> None:
        """Nạp token của part kế tiếp vào RAM; hàng đợi giới hạn ⇒ tối đa `tokenize_prefetch_parts` part đi trước GPU."""
        try:
            for part in range(self.n_parts):
                if self.ledger.done(part, "embedded"):
                    continue
                if not self._wait(chunked[part].is_set):
                    return
                item = (part, tokenize.load_tokens(self.layout.tok(part)))
                while not self.stop.is_set():
                    try:
                        tokens.put(item, timeout=POLL_S)
                        break
                    except queue.Full:
                        pass
            tokens.put(None)
        except BaseException as exc:
            self.fail("embed", exc)

    def upload(self, parts: queue.Queue) -> None:
        part = None
        try:
            if not self._wait(self.avgdl_ready.is_set):
                return
            self.report.begin("upload", self.n_parts)
            avgdl = self.ledger.state["bm25_avgdl"]
            while (part := parts.get()) is not None and not self.stop.is_set():
                if self.ledger.done(part, "uploaded"):
                    continue
                n = qdrant_upload.upload_part(self.settings, self.qc, self.layout.collection, self.layout, part,
                                              avgdl, self.upload_counters, log)
                self.ledger.mark(part, "uploaded")
                self.report.progress("upload", self.ledger.count("uploaded"))
                log(f"part {part:04d}: uploaded {n} point")
            self.report.end("upload")
        except BaseException as exc:
            self.fail("upload", exc, part)

    def embed_all(self, tokens: queue.Queue, uploads: queue.Queue) -> None:
        self.step = "embed"
        self.report.begin("embed", self.n_parts)
        while True:
            try:
                item = tokens.get(timeout=POLL_S)
            except queue.Empty:
                if self.stop.is_set():
                    return
                continue
            if item is None:
                break
            part, toks = item
            stages.require_disk(self.settings, part)
            t0 = time.monotonic()
            n = stages.embed_part(self.encoder, self.layout, part, toks)
            if n != self.ledger.part(part)["n_chunks"]:
                raise RuntimeError(f"part {part:04d}: embed {n} hàng ≠ n_chunks {self.ledger.part(part)['n_chunks']}")
            seconds = time.monotonic() - t0
            self.report.embed_chunks += n
            self.report.embed_seconds += seconds
            self.ledger.mark(part, "embedded")
            self.layout.tok(part).unlink(missing_ok=True)
            uploads.put(part)
            self.report.progress("embed", self.ledger.count("embedded"))
            self.report.sample_peaks()
            log(f"part {part:04d}: embedded {n} chunk ({n / seconds:.0f} chunk/s)")
        self.report.end("embed")


def run(settings, resume: bool = True, dry_run: bool = False) -> int:
    """CLI `night`: chạy (hoặc tiếp tục) lượt chạy dài. Report luôn được ghi; 0 = completed, 20 = lỗi cần người."""
    stages.offline_env()
    stages.keep_awake(True)
    job = NightRun(settings, dry_run)
    # Pool tạo TRƯỚC khi nạp CUDA (spawn: tiến trình con không kế thừa trạng thái GPU).
    tok_pool = bm25_mod.make_pool(settings.tokenize_workers)
    bm25_pool = bm25_mod.make_pool(settings.bm25_workers)
    status = "failed"
    try:
        if not stages.prepare(job, resume, log):
            status = "skipped"
            return EXIT_OK
        chunked = {p: threading.Event() for p in range(job.n_parts)}
        tokens: queue.Queue = queue.Queue(maxsize=settings.tokenize_prefetch_parts)
        uploads: queue.Queue = queue.Queue()
        for part in stages.resume_uploads(job.ledger, job.n_parts, log):
            uploads.put(part)
        threads = [threading.Thread(target=job.cpu_stages, args=(tok_pool, bm25_pool, chunked), daemon=True),
                   threading.Thread(target=job.feed, args=(chunked, tokens), daemon=True),
                   threading.Thread(target=job.upload, args=(uploads,), daemon=True)]
        for t in threads:
            t.start()
        from vre.index.embed import get_encoder

        job.encoder = get_encoder(settings, "cuda")
        job.embed_all(tokens, uploads)
        uploads.put(None)
        for t in threads:
            t.join()
        if job.failure is not None:
            raise job.failure
        stages.finalize(job, log)
        status = "completed"
        log(f"XONG: {job.layout.collection}, {job.ledger.n_chunks()} chunk")
    except BaseException as exc:
        if job.failure is None:
            job.fail(job.step, exc)
    finally:
        job.stop.set()
        for pool in (tok_pool, bm25_pool):
            bm25_mod.terminate_pool(pool)
        if status != "skipped":
            finish(job, status)
            log(f"report → {job.layout.report_path} (status={status})")
        stages.keep_awake(False)
    return EXIT_OK if status == "completed" else EXIT_NEEDS_HUMAN
