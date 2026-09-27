@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo .venv is missing. Run setup_windows.bat first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m pytest -q
pause
