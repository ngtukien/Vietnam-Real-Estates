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
from typing import TYPE_CHECKING, Any

from vre.index import bm25 as bm25_mod
from vre.index import qdrant_upload, tokenize
from vre.pipeline import stages
from vre.pipeline.report import Report, finish, require_resources
from vre.pipeline.state import Ledger, layout_for, log

if TYPE_CHECKING:
    from vre.index.embed import Encoder

EXIT_OK, EXIT_NEEDS_HUMAN = 0, 20
POLL_S = 5


class NightRun:
    def __init__(self, settings, dry_run: bool):
        self.settings, self.dry_run = settings, dry_run
        self.layout = layout_for(settings, dry_run)
        self.report = Report(self.layout, settings)
        self.stop = threading.Event()
        self.failure: BaseException | None = None
        self.avgdl_ready = threading.Event()
        self.ledger: Ledger | None = None  # stages.prepare mở; report.finish chấp nhận None (lỗi trước khi mở)
        self.encoder: Encoder | None = None
        self.qc: Any = None  # QdrantClient, do stages.prepare tạo
        self.n_parts, self.step = 0, "checks"
        self.upload_counters = {"upload_retries": 0}

    def require_ledger(self) -> Ledger:
        """Ledger đã mở (stages.prepare chạy trước mọi stage); gọi sớm hơn là lỗi lập trình."""
        if self.ledger is None:
            raise RuntimeError("ledger chưa mở: stages.prepare phải chạy trước")
        return self.ledger

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
        ledger, layout, parts, step = self.require_ledger(), self.layout, range(self.n_parts), "chunk"
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
            for pool in (tok_pool, bm25_pool):  # xong phần CPU: trả RAM của 16 worker cho phần còn lại (~8h)
                bm25_mod.terminate_pool(pool)
        except BaseException as exc:
            self.fail(step, exc)

    def feed(self, chunked: dict, tokens: queue.Queue) -> None:
        try:
            ledger = self.require_ledger()
            for part in range(self.n_parts):
                if ledger.done(part, "embedded"):
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

    def upload(self, parts: queue.Queue, upload_pool) -> None:
        part = None
        try:
            ledger = self.require_ledger()
            if not self._wait(self.avgdl_ready.is_set):
                return
            self.report.begin("upload", self.n_parts)
            avgdl = ledger.state["bm25_avgdl"]
            while (part := parts.get()) is not None and not self.stop.is_set():
                if ledger.done(part, "uploaded"):
                    continue
                out = upload_pool.submit(qdrant_upload.upload_part_task, (self.settings, self.layout, part, avgdl)).result()
                self.upload_counters["upload_retries"] += out["upload_retries"]
                ledger.mark(part, "uploaded")
                self.report.progress("upload", ledger.count("uploaded"))
                log(f"part {part:04d}: uploaded {out['n']} point")
            self.report.end("upload")
        except BaseException as exc:
            self.fail("upload", exc, part)

    def embed_all(self, tokens: queue.Queue, uploads: queue.Queue) -> None:
        self.step = "embed"
        ledger = self.require_ledger()
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
            require_resources(self.settings, part, log)
            t0 = time.monotonic()
            n = stages.embed_part(self.encoder, self.layout, part, toks)
            if n != ledger.part(part)["n_chunks"]:
                raise RuntimeError(f"part {part:04d}: embed {n} hàng ≠ n_chunks {ledger.part(part)['n_chunks']}")
            seconds = time.monotonic() - t0
            self.report.embed_chunks += n
            self.report.embed_seconds += seconds
            ledger.mark(part, "embedded")
            self.layout.tok(part).unlink(missing_ok=True)
            uploads.put(part)
            self.report.progress("embed", ledger.count("embedded"))
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
    upload_pool = bm25_mod.make_pool(1)  # dựng point + gRPC ở tiến trình riêng: không giành GIL với vòng GPU
    status = "failed"
    try:
        if not stages.prepare(job, resume, log):
            status = "skipped"
            return EXIT_OK
        chunked = {p: threading.Event() for p in range(job.n_parts)}
        tokens: queue.Queue[tuple[int, Any] | None] = queue.Queue(maxsize=settings.tokenize_prefetch_parts)
        uploads: queue.Queue[int | None] = queue.Queue()
        for part in stages.resume_uploads(job.ledger, job.n_parts, log):
            uploads.put(part)
        threads = [threading.Thread(target=job.cpu_stages, args=(tok_pool, bm25_pool, chunked), daemon=True),
                   threading.Thread(target=job.feed, args=(chunked, tokens), daemon=True),
                   threading.Thread(target=job.upload, args=(uploads, upload_pool), daemon=True)]
        for thread in threads:
            thread.start()
        ledger = job.require_ledger()
        if not all(ledger.done(p, "embedded") for p in range(job.n_parts)):  # model ~6 GB commit: chỉ nạp khi cần
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
        log(f"XONG: {job.layout.collection}, {ledger.n_chunks()} chunk")
    except BaseException as exc:
        if job.failure is None:
            job.fail(job.step, exc)
    finally:
        job.stop.set()
        for pool in (tok_pool, bm25_pool, upload_pool):
            bm25_mod.terminate_pool(pool)
        if status != "skipped":
            finish(job, status)
            log(f"report → {job.layout.report_path} (status={status})")
        stages.keep_awake(False)
    return EXIT_OK if status == "completed" else EXIT_NEEDS_HUMAN
