# Lượt chạy dài chỉ mục e5-large (phase-02 Bước 5). Không cần người: tự chạy lại khi lỗi, tiếp tục từ checkpoint.
#   powershell -NoProfile -ExecutionPolicy Bypass -File D:\Vietnam-Real-Estates-p2\scripts\night.ps1
# Mã thoát: 0 = completed; 20 = vẫn lỗi sau night_max_attempts lần (đọc errors[] trong night_report.json + log).
$ErrorActionPreference = 'Continue'
. 'D:\Vietnam-Real-Estates\scripts\vre_env.ps1'
$env:PYTHONUNBUFFERED = '1'
# CR-04 (chờ p1 đổi mặc định trong vre.config): ít worker CPU hơn để giữ commit memory; GPU vẫn là nút cổ chai.
$env:VRE_TOKENIZE_WORKERS = '4'
$env:VRE_BM25_WORKERS = '6'

$Root = 'D:\Vietnam-Real-Estates'
$Compose = Join-Path $Root 'docker-compose.yaml'
$Cli = Join-Path $PSScriptRoot 'vre_pipeline.py'
$LogDir = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir ("night-{0}.log" -f (Get-Date -Format 'yyMMdd-HHmm'))

function Write-Log([string]$Message) {
    $line = "[{0}] night.ps1: {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    $line | Tee-Object -FilePath $Log -Append
}

$MaxAttempts = [int](& $VRE_PY $Cli config-get night_max_attempts)
$RetryWait = [int](& $VRE_PY $Cli config-get night_retry_wait_s)
Write-Log "log=$Log max_attempts=$MaxAttempts retry_wait_s=$RetryWait"

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    Write-Log "lần ${attempt}/${MaxAttempts}: docker compose up -d --wait qdrant"
    & docker compose -f $Compose up -d --wait qdrant 2>&1 | Tee-Object -FilePath $Log -Append
    Write-Log "lần ${attempt}/${MaxAttempts}: vre_pipeline.py night --resume"
    & $VRE_PY $Cli night --resume 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $Log -Append
    $code = $LASTEXITCODE
    Write-Log "lần ${attempt}/${MaxAttempts}: mã thoát $code"
    if ($code -eq 0) { exit 0 }
    if ($attempt -lt $MaxAttempts) {
        Write-Log "chờ $RetryWait s rồi chạy lại cùng lệnh (tiếp tục từ checkpoint)"
        Start-Sleep -Seconds $RetryWait
    }
}
Write-Log "hết $MaxAttempts lần, vẫn lỗi: cần người (xem errors[] trong night_report.json)"
exit 20
