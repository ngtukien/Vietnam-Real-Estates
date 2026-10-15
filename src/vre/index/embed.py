"""Encoder e5-large tự viết cho GPU (data-contract §9.1): fp16 + SDPA, lô theo ngân sách token.

Trong một part: sắp theo độ dài, gom lô sao cho `số câu × độ dài dài nhất ≤ embed_token_budget` (tối đa `embed_max_batch` câu),
mean pooling theo attention mask, L2-normalize, trả fp16 theo đúng thứ tự gốc. CUDA OOM ⇒ giải phóng cache, ngân sách ½,
làm lại lô; dưới `embed_min_token_budget` mà vẫn OOM ⇒ raise (lỗi cần người).
"""

import threading

import numpy as np

from vre.config import EMBED_MODEL_SLUG, EMBED_MODELS

_lock = threading.Lock()
# Trần bộ nhớ đệm CUDA của tiến trình. Không có trần, caching allocator giữ khối cho mọi shape lô (đo 2026-10-15: reserved
# 7,5 GB / allocated 1,2 GB sau 20s) ⇒ vượt VRAM 8 GB, WDDM tràn sang bộ nhớ hệ thống và tính vào commit (máy sập 2 lần).
# Benchmark đỉnh 2 GB ở budget 32768; chạm trần thì allocator trả cache và thử lại, vẫn thiếu ⇒ OOM ⇒ ngân sách ½.
VRAM_CAP_GB = 4.0
_encoder = None


class Encoder:
    def __init__(self, settings, device: str = "cuda"):
        import torch

        from vre.index import tokenize
        from vre.index.weights import build_model_fp16

        self.torch, self.device = torch, device
        self.tokenizer = tokenize.load_tokenizer()
        self.special = tokenize.special_ids(self.tokenizer)
        # Không nạp qua transformers: backend mmap COW của safetensors làm sập tiến trình khi commit thấp
        # (plans/reports/p3-encoder-crash.md). Loader dùng chung đọc theo lát, thiếu bộ nhớ thì nổ MemoryError.
        self.model = build_model_fp16().to(device).eval()
        if device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_GB * 1e9 / total))
        self.dim = EMBED_MODELS[EMBED_MODEL_SLUG][1]
        self.max_length = settings.embed_max_length
        self.max_batch = settings.embed_max_batch
        self.min_budget = settings.embed_min_token_budget
        self.budget = settings.embed_token_budget
        self.oom_retries = 0
        self.last_batch_size = 0

    def _groups(self, lengths: np.ndarray) -> list[np.ndarray]:
        order = np.argsort(lengths, kind="stable")
        groups, start = [], 0
        while start < len(order):
            end = start + 1
            while end < len(order) and end - start < self.max_batch and \
                    (end - start + 1) * lengths[order[end]] <= self.budget:
                end += 1
            groups.append(order[start:end])
            start = end
        return groups

    def _forward(self, ids: np.ndarray, offsets: np.ndarray, rows: np.ndarray):
        torch = self.torch
        lengths = offsets[rows + 1] - offsets[rows]
        batch = np.full((len(rows), int(lengths.max())), self.special.pad, dtype=np.int64)
        for j, r in enumerate(rows):
            batch[j, :lengths[j]] = ids[offsets[r]:offsets[r + 1]]
        mask = (np.arange(batch.shape[1])[None, :] < lengths[:, None]).astype(np.int64)
        # Không pin_memory: CachingHostAllocator giữ lại mọi khối pinned theo từng shape lô ⇒ commit của tiến trình tăng
        # dần (dry-run 2026-10-15: 18 → 22 GB, góp phần làm cạn commit của máy). Lô id chỉ vài trăm KB nên copy đồng bộ rẻ.
        input_ids = torch.from_numpy(batch).to(self.device)
        attention = torch.from_numpy(mask).to(self.device)
        hidden = self.model(input_ids=input_ids, attention_mask=attention).last_hidden_state
        m = attention.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * m).sum(1) / m.sum(1)
        return torch.nn.functional.normalize(pooled.float(), dim=-1).half()

    def encode_ids(self, ids: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """Embed các chuỗi id đã có (ids phẳng + offsets) → fp16 `[n, dim]`, L2-normalized, thứ tự gốc."""
        torch = self.torch
        n = len(offsets) - 1
        lengths = offsets[1:] - offsets[:-1]
        if n and int(lengths.max()) > self.max_length:
            raise ValueError(f"chuỗi dài {int(lengths.max())} token > embed_max_length={self.max_length}")
        out = torch.empty((n, self.dim), dtype=torch.float16, device=self.device)
        with torch.inference_mode():
            pending = self._groups(lengths)
            while pending:
                rows = pending.pop(0)
                try:
                    out[torch.from_numpy(rows).to(self.device)] = self._forward(ids, offsets, rows)
                    self.last_batch_size = len(rows)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    self.oom_retries += 1
                    if self.budget // 2 < self.min_budget:
                        raise
                    self.budget //= 2
                    rest = np.concatenate([rows, *pending]) if pending else rows
                    pending = self._groups_of(rest, lengths)
        return out.cpu().numpy()

    def _groups_of(self, rows: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
        """Gom lại các dòng còn lại theo ngân sách mới (giữ chỉ số gốc)."""
        return [rows[g] for g in self._groups(lengths[rows])]

    def encode_texts(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        """Embed văn bản thô (truy vấn, test parity): thêm tiền tố `passage: `/`query: `, cắt ở `max_length`."""
        prefix = str(EMBED_MODELS[EMBED_MODEL_SLUG][3 if kind == "passage" else 4])
        enc = self.tokenizer([prefix + t for t in texts], truncation=True, max_length=self.max_length)["input_ids"]
        offsets = np.zeros(len(enc) + 1, dtype=np.int64)
        np.cumsum([len(e) for e in enc], out=offsets[1:])
        flat = np.fromiter((t for e in enc for t in e), dtype=np.int64, count=int(offsets[-1]))
        return self.encode_ids(flat, offsets)


def get_encoder(settings, device: str | None = None) -> Encoder:
    """Một encoder cho cả tiến trình (lượt chạy dài và smoke_search dùng chung model đã nạp)."""
    global _encoder
    with _lock:
        if _encoder is None:
            import torch

            _encoder = Encoder(settings, device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return _encoder
