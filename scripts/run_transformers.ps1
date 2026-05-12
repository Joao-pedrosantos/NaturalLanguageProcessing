# Roda apenas os transformers (tiny, medium, 10M) em sequência,
# seguidos de clean_log, compare_runs e probe_classify.
#
# Uso:
#   pwsh -File scripts\run_transformers.ps1
#
# Loga tudo em logs\run_transformers_<timestamp>.log.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory $logDir | Out-Null }
$logFile = Join-Path $logDir "run_transformers_$timestamp.log"

function Log {
    param([string]$msg)
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Add-Content -Path $logFile -Value $line -Encoding utf8
    Write-Output $line
}

# Pega o python: prefere venv local, senão miniconda.
$py = $null
if (Test-Path "venv\Scripts\python.exe") { $py = "venv\Scripts\python.exe" }
elseif (Test-Path "venv\bin\python")     { $py = "venv\bin\python" }
else { $py = (Get-Command python).Source }
Log "python: $py"

function Run-Step {
    param([string]$label, [string[]]$args)
    Log "=== $label ==="
    & $py @args 2>&1 | Tee-Object -FilePath $logFile -Append
    if ($LASTEXITCODE -ne 0) {
        Log "FAIL: $label (exit $LASTEXITCODE)"
        throw "step failed: $label"
    }
}

# --- Treinos ---
$trainConfigs = @(
    @{ label = "Transformer tiny";   cfg = "configs/train_tiny_transformer.yaml" },
    @{ label = "Transformer medium"; cfg = "configs/train_medium_transformer.yaml" },
    @{ label = "Transformer 10M";    cfg = "configs/train_10m_transformer.yaml" }
)

foreach ($t in $trainConfigs) {
    Run-Step $t.label @(
        "-m", "scripts.train_local",
        "--config", $t.cfg,
        "--train-file", "artifacts/train_books",
        "--dev-file", "artifacts/dev/pt_dev.txt"
    )
}

# --- Clean logs (xLSTM + Transformer existentes) ---
$logsToClean = @(
    "runs/xlstm_tiny/log.jsonl",
    "runs/xlstm_medium/log.jsonl",
    "runs/xlstm_10m/log.jsonl",
    "runs/transformer_tiny/log.jsonl",
    "runs/transformer_medium/log.jsonl",
    "runs/transformer_10m/log.jsonl"
) | Where-Object { Test-Path $_ }

if ($logsToClean.Count -gt 0) {
    Run-Step "clean_log" (@("-m", "scripts.clean_log") + $logsToClean)
}

# --- Scaling curve com os 6 runs ---
$compareArgs = @("-m", "scripts.compare_runs", "--runs")
$labels = @()
$runDirs = @(
    @{ dir = "runs/xlstm_tiny";        label = "xLSTM tiny" },
    @{ dir = "runs/xlstm_medium";      label = "xLSTM medium" },
    @{ dir = "runs/xlstm_10m";         label = "xLSTM 10M" },
    @{ dir = "runs/transformer_tiny";  label = "Transformer tiny" },
    @{ dir = "runs/transformer_medium";label = "Transformer medium" },
    @{ dir = "runs/transformer_10m";   label = "Transformer 10M" }
)
$existing = $runDirs | Where-Object { Test-Path (Join-Path $_.dir "log.jsonl") }
$compareArgs += ($existing | ForEach-Object { $_.dir })
$compareArgs += "--labels"
$compareArgs += ($existing | ForEach-Object { $_.label })
$compareArgs += @("--out", "artifacts/scaling_curve.png")
Run-Step "compare_runs" $compareArgs

if (Test-Path "artifacts/scaling_curve.png") {
    Copy-Item "artifacts/scaling_curve.png" "scaling_curve.png" -Force
}

# --- Author-classification probe nos medium/10M (xLSTM + Transformer) ---
$probeTargets = @(
    @{ name = "xlstm_medium";       arch = "xlstm" },
    @{ name = "xlstm_10m";          arch = "xlstm" },
    @{ name = "transformer_medium"; arch = "transformer" },
    @{ name = "transformer_10m";    arch = "transformer" }
)
foreach ($p in $probeTargets) {
    $ckpt = "runs/$($p.name)/best.pt"
    $cfg  = "configs/$($p.name).yaml"
    $out  = "artifacts/probe_$($p.name).json"
    if (Test-Path $ckpt) {
        Run-Step "probe $($p.name)" @(
            "-m", "scripts.probe_classify",
            "--ckpt", $ckpt, "--config", $cfg, "--arch", $p.arch,
            "--out", $out
        )
    } else {
        Log "(skip probe $($p.name): $ckpt ausente)"
    }
}

Log "feito."
