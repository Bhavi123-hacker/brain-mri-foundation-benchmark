<#
.SYNOPSIS
    SHIFT-MRI Research Web Application Launcher (PowerShell)
    Fair Benchmarking of Brain-MRI Foundation Models
    Version 3.0.0-final | October 2026
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\ml project"
Set-Location -Path $ProjectRoot

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  SHIFT-MRI: BRAIN MRI FOUNDATION BENCHMARK & FAILURE STUDIO" -ForegroundColor Green
Write-Host "  Source-Locked Cross-Cohort Reliability Benchmark" -ForegroundColor Gray
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "[ERROR] Virtual environment python executable not found at: $PythonExe"
    exit 1
}

$pyVersion = & $PythonExe --version
Write-Host "[OK] Python Environment Verified: $pyVersion" -ForegroundColor Green
Write-Host ""

$port = 8000
$url = "http://localhost:$port"

Write-Host "Starting SHIFT-MRI Web Application Server..." -ForegroundColor Yellow
Write-Host "Access URL: $url" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server.`n" -ForegroundColor Gray

Start-Process $url

& $PythonExe run_frontend.py --port $port
