@echo off
setlocal enabledelayedexpansion
title SHIFT-MRI Research Platform Launcher

echo ===============================================================================
echo   SHIFT-MRI: Source-Locked Cross-Cohort Reliability Research Platform
echo   Investigator: Bhavy Garg ^| VIT Chennai
echo ===============================================================================
echo.

:: 1. Check/Ensure Drive D: Mapping
if not exist "D:\" (
    echo [*] Virtual drive D: not found. Checking if E: exists...
    if exist "E:\ml project" (
        echo [*] Mapping E:\ to D:\ via subst for path compatibility...
        subst D: E:\ >nul 2>&1
        if exist "D:\" (
            echo [OK] Virtual drive D: mapped successfully to E:\.
        ) else (
            echo [WARN] Could not automatically map D:. Proceeding in current directory.
        )
    )
)

:: Set working directory
if exist "D:\ml project" (
    cd /d "D:\ml project"
) else (
    cd /d "%~dp0"
)

:: 2. Verify Python Virtual Environment
set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment python not found at %PYTHON_EXE%!
    echo Please ensure the .venv directory is present.
    pause
    exit /b 1
)

:: 3. Verify Server Script
if not exist "backend\server.py" (
    echo [ERROR] backend\server.py not found in current directory: %CD%
    pause
    exit /b 1
)

:: 4. Check if Server is already running on Port 8000
netstat -ano | findstr /R /C:":8000.*LISTENING" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [*] SHIFT-MRI Server is already active and listening on port 8000.
) else (
    echo [*] Launching SHIFT-MRI Background Server on http://127.0.0.1:8000 ...
    start "" /b "%PYTHON_EXE%" backend\server.py --port 8000
    timeout /t 3 /nobreak >nul
)

:: 5. Open Web Application in Default Browser
echo [*] Opening research workstation in default browser...
start http://127.0.0.1:8000

echo.
echo ===============================================================================
echo   [OK] SHIFT-MRI Platform is Running!
echo   URL: http://127.0.0.1:8000
echo   Health Check: http://127.0.0.1:8000/api/health
echo.
echo   Press any key to close this launcher window (Server will remain active).
echo ===============================================================================
pause >nul
