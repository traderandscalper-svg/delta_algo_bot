@echo off
setlocal
cd /d "%~dp0"

echo Creating Python virtual environment...
py -m venv .venv
if errorlevel 1 goto :error

echo Installing dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error

.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error

if not exist .env (
    copy /Y .env.example .env >nul
)

echo.
echo Setup complete.
echo Edit .env with your Delta DEMO/TESTNET API credentials.
echo Then run:
echo   .venv\Scripts\python.exe -m app.main
echo.
pause
exit /b 0

:error
echo.
echo Setup failed. Check the error above.
pause
exit /b 1
