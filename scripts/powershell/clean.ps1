$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$PytestCache = Join-Path $Root ".pytest_cache"
if (Test-Path -LiteralPath $PytestCache) {
    Remove-Item -LiteralPath $PytestCache -Recurse -Force
}
Get-ChildItem -LiteralPath $Root -Directory -Recurse -Force -Filter "__pycache__" |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $Root -File -Recurse -Force |
    Where-Object { $_.Extension -in ".pyc", ".pyo" } |
    Remove-Item -Force
Write-Host "Python and pytest temporary caches removed; CUDA/Triton caches preserved."
