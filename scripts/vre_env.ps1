# Biến môi trường ổ D cho mọi lệnh của dự án. Dùng: . D:\Vietnam-Real-Estates\scripts\vre_env.ps1
# Mọi cache/TMP trỏ về checkout chính (dùng chung với worktree p2) để không ghi gì lên ổ C.
$VreRoot = 'D:\Vietnam-Real-Estates'
$VreCache = Join-Path $VreRoot '.cache'
$env:VRE_DATA_ROOT = $VreRoot
$paths = [ordered]@{
    HF_HOME                  = Join-Path $VreCache 'huggingface'
    TORCH_HOME               = Join-Path $VreCache 'torch'
    PIP_CACHE_DIR            = Join-Path $VreCache 'pip'
    TMP                      = Join-Path $VreCache 'tmp'
    TEMP                     = Join-Path $VreCache 'tmp'
    CUDA_CACHE_PATH          = Join-Path $VreCache 'nv'
    MPLCONFIGDIR             = Join-Path $VreCache 'matplotlib'
    XDG_CACHE_HOME           = Join-Path $VreCache 'xdg'
    JUPYTER_DATA_DIR         = Join-Path $VreCache 'jupyter\data'
    JUPYTER_RUNTIME_DIR      = Join-Path $VreCache 'jupyter\runtime'
    IPYTHONDIR               = Join-Path $VreCache 'jupyter\ipython'
    npm_config_cache         = Join-Path $VreCache 'npm'
    PLAYWRIGHT_BROWSERS_PATH = Join-Path $VreCache 'ms-playwright'
}
foreach ($name in $paths.Keys) {
    New-Item -ItemType Directory -Force -Path $paths[$name] | Out-Null
    Set-Item -Path "env:$name" -Value $paths[$name]
}
$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$VRE_PY = Join-Path $VreRoot '.venv\Scripts\python.exe'
$global:VRE_PY = $VRE_PY
