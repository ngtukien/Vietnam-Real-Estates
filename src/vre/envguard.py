"""Giữ mọi cache/TMP trên ổ D. Gọi `apply()` ĐẦU TIÊN, trước mọi import torch/HF.

`apply()` đặt mặc định bằng `os.environ.setdefault` rồi raise nếu bất kỳ biến nào trỏ sang ổ khác `D:`
(errors over silent fallbacks: không âm thầm ghi đè biến người dùng đã đặt — dot-source `scripts/vre_env.ps1`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from pathlib import PureWindowsPath

DATA_ROOT_DEFAULT = r"D:\Vietnam-Real-Estates"
REQUIRED_DRIVE = "D:"

# biến -> đường dẫn con dưới <data_root>\.cache
CACHE_VARS: dict[str, str] = {
    "HF_HOME": "huggingface",
    "TORCH_HOME": "torch",
    "PIP_CACHE_DIR": "pip",
    "TMP": "tmp",
    "TEMP": "tmp",
    "CUDA_CACHE_PATH": "nv",
    "MPLCONFIGDIR": "matplotlib",
    "XDG_CACHE_HOME": "xdg",
    "JUPYTER_DATA_DIR": r"jupyter\data",
    "JUPYTER_RUNTIME_DIR": r"jupyter\runtime",
    "IPYTHONDIR": r"jupyter\ipython",
    "npm_config_cache": "npm",
    "PLAYWRIGHT_BROWSERS_PATH": "ms-playwright",
}
FLAG_VARS: dict[str, str] = {"PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false"}


class EnvPathError(RuntimeError):
    """Một biến cache/TMP trỏ ra ngoài ổ D."""


def defaults(data_root: str = DATA_ROOT_DEFAULT) -> dict[str, str]:
    cache = PureWindowsPath(data_root) / ".cache"
    values = {name: str(cache / sub) for name, sub in CACHE_VARS.items()}
    values["VRE_DATA_ROOT"] = data_root
    return values


def violations(environ: Mapping[str, str]) -> dict[str, str]:
    """Biến cache/TMP (và VRE_DATA_ROOT) không nằm trên ổ D. Rỗng ⇒ đạt."""
    names = [*CACHE_VARS, "VRE_DATA_ROOT"]
    return {n: environ[n] for n in names
            if n in environ and not environ[n].upper().startswith(REQUIRED_DRIVE)}


def apply(environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ if environ is None else environ
    root = env.get("VRE_DATA_ROOT", DATA_ROOT_DEFAULT)
    for name, value in {**defaults(root), **FLAG_VARS}.items():
        env.setdefault(name, value)
    bad = violations(env)
    if bad:
        listed = ", ".join(f"{k}={v}" for k, v in sorted(bad.items()))
        raise EnvPathError(f"Biến môi trường trỏ ra ngoài ổ D: {listed}. "
                           r"Chạy '. D:\Vietnam-Real-Estates\scripts\vre_env.ps1' trước.")
    if environ is None:
        for name in CACHE_VARS:
            os.makedirs(env[name], exist_ok=True)
    return {n: env[n] for n in CACHE_VARS}
