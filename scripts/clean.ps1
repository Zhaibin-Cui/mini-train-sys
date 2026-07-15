$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Targets = @(
    ".pytest_cache",
    ".triton_cache",
    "build",
    "dist",
    "mini_train_sys.egg-info",
    "minitrain/kernels/cuda_ext/build",
    "tests/benchmark_results",
    "checkpoints",
    "runs",
    "logs",
    "outputs",
    "profiles"
)

foreach ($Relative in $Targets) {
    $Path = Join-Path $Root $Relative
    if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Recurse -Force }
}
Get-ChildItem -LiteralPath $Root -Directory -Force -Filter ".pytest_tmp*" |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $Root -Directory -Recurse -Force -Filter "__pycache__" |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $Root -File -Recurse -Force |
    Where-Object { $_.Extension -in ".pyc", ".pyo" } |
    Remove-Item -Force
Get-ChildItem -LiteralPath $Root -File -Force |
    Where-Object { $_.Extension -in ".obj", ".lib", ".exp" } |
    Remove-Item -Force

Write-Host "Generated build, test, cache, and run outputs removed."
