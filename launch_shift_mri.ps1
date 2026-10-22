<#
.SYNOPSIS
    SHIFT-MRI Research Platform Launcher for Windows PowerShell.
.DESCRIPTION
    Validates virtual environment, ensures drive compatibility, launches the
    SHIFT-MRI audited HTTP backend server, and launches the browser workstation.
#>

[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$HostAddress = "127.0.0.1"
)

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "  SHIFT-MRI: Source-Locked Cross-Cohort Reliability Research Platform" -ForegroundColor White
Write-Host "  Investigator: Bhavy Garg | School of Computer Science & Engineering, VIT Chennai" -ForegroundColor Gray
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Drive D: Verification / Mapping
if (-not (Test-Path "D:\")) {
    Write-Host "[*] Drive D:\ not found. Checking if E:\ exists..." -ForegroundColor Yellow
    if (Test-Path "E:\ml project") {
        Write-Host "[*] Mapping E:\ to D:\ via subst..." -ForegroundColor Yellow
        Start-Process -FilePath "subst" -ArgumentList "D: E:\" -NoNewWindow -Wait
        if (Test-Path "D:\") {
            Write-Host "[OK] Virtual drive D:\ mapped successfully to E:\" -ForegroundColor Green
        }
    }
}

# Resolve directory
$projectRoot = if (Test-Path "D:\ml project") { "D:\ml project" } else { $PSScriptRoot }
Set-Location $projectRoot
Write-Host "[*] Project Root: $projectRoot" -ForegroundColor Gray

# 2. Virtual Environment Verification
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "[FATAL] Python virtual environment not found at $pythonExe."
    exit 1
}

# 3. Check if server already listening on target port
$active = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }
if ($active) {
    Write-Host "[*] Server is already actively listening on port $Port." -ForegroundColor Green
} else {
    Write-Host "[*] Starting SHIFT-MRI Server daemon on port $Port..." -ForegroundColor Cyan
    $serverScript = Join-Path $projectRoot "backend\server.py"
    $proc = Start-Process -FilePath $pythonExe -ArgumentList "$serverScript --port $Port" -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 3
    Write-Host "[OK] Server process spawned (PID: $($proc.Id))." -ForegroundColor Green
}

# 4. Verification Check
$url = "http://${HostAddress}:${Port}"
try {
    $health = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 5
    Write-Host "[OK] Health Check Passed: Status=$($health.status), Version=$($health.version), Benchmark Models=$($health.benchmarked_models)" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Health check did not respond immediately, proceeding to browser launch..." -ForegroundColor Yellow
}

# 5. Open Web Application
Write-Host "[*] Launching default web browser to $url ..." -ForegroundColor Cyan
Start-Process $url

Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "  Platform Active at: $url" -ForegroundColor White
Write-Host "  API Health: $url/api/health" -ForegroundColor Gray
Write-Host "  Registry:   $url/api/registry" -ForegroundColor Gray
Write-Host "===============================================================================" -ForegroundColor Cyan
