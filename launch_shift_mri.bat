@echo off
REM ============================================================================
REM SHIFT-MRI Research Web Application Launcher (Windows Batch)
REM Fair Benchmarking of Brain-MRI Foundation Models
REM Version 3.0.0-final | October 2026
REM ============================================================================

setlocal enabledelayedexpansion
title SHIFT-MRI Research Studio & Inference Server

cd /d "D:\ml project"

echo ============================================================================
echo   SHIFT-MRI: BRAIN MRI FOUNDATION BENCHMARK & FAILURE STUDIO
echo   Source-Locked Cross-Cohort Reliability Benchmark
echo ============================================================================
echo.

set PYTHON_EXE=D:\ml project\.venv\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment python executable not found at:
    echo   %PYTHON_EXE%
    echo Please ensure the Python 3.12 virtual environment is configured.
    pause
    exit /b 1
)

echo [OK] Python Environment Verified:
"%PYTHON_EXE%" --version
echo.

echo Launching SHIFT-MRI Web Application...
echo Access URL: http://localhost:8000
echo.

start "" "http://localhost:8000"

"%PYTHON_EXE%" run_frontend.py --port 8000

pause
