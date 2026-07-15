param(
    [string]$Config = "configs/train_single.yaml",
    [string]$ModelConfig = "configs/model_default.yaml",
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [string]$Resume = "",
    [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { "python" })
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Arguments = @("scripts/train.py", "--config", $Config, "--model-config", $ModelConfig, "--device", $Device)
if ($Resume) { $Arguments += @("--resume", $Resume) }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
