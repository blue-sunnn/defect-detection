@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo Defect Detection Platform - Source Runner
echo ============================================================
echo.

set "PYTHON_CMD="

where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
    )
)

if not defined PYTHON_CMD (
    echo ERROR: Python was not found.
    echo.
    echo Install Python 3 from https://www.python.org/downloads/
    echo During installation, enable "Add python.exe to PATH".
    echo Then run start_app.bat again.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating local virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :error
) else (
    echo [1/4] Using existing local virtual environment.
)

echo.
echo [2/4] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :error

echo.
echo [3/4] Installing runtime dependencies...
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo [4/4] Starting application...
python run_app.py
if errorlevel 1 goto :error

exit /b 0

:error
echo.
echo ============================================================
echo The application could not start.
echo Read the first error above for details.
echo ============================================================
pause
exit /b 1
