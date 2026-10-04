@echo off
cd /d "%~dp0"
title Cache Creek Game Camera Project (Debug)

echo ============================================================
echo  Cache Creek Game Camera Project - Debug Mode
echo  Starting...
echo ============================================================
echo.

:: Find Python
set PYTHON=
for %%P in (python.exe) do set PYTHON=%%~$PATH:P
if "%PYTHON%"=="" (
    :: Try known install location
    if exist "C:\Users\RJ\AppData\Local\Programs\Python\Python314\python.exe" (
        set PYTHON=C:\Users\RJ\AppData\Local\Programs\Python\Python314\python.exe
    )
)

if "%PYTHON%"=="" (
    echo ERROR: Python not found on PATH.
    echo Please install Python from https://python.org and check "Add to PATH".
    pause
    exit /b 1
)

echo Using Python: %PYTHON%
echo.

:: Create an isolated virtual environment; never install into system Python.
if not exist ".venv\Scripts\python.exe" (
    echo Creating isolated Python environment...
    "%PYTHON%" -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create the virtual environment.
        pause
        exit /b 1
    )
)
set PYTHON=.venv\Scripts\python.exe

:: Install dependencies if not done yet
if not exist ".venv\.deps_installed" (
    echo Installing dependencies - this may take several minutes...
    echo.
    "%PYTHON%" -m pip install --upgrade pip
    "%PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo ERROR: CPU-only PyTorch installation failed.
        pause
        exit /b 1
    )
    "%PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Dependency install failed. See above.
        pause
        exit /b 1
    )
    echo. > .venv\.deps_installed
    echo Dependencies installed successfully.
    echo.
)

:: Run and write log
echo Running application... Output also saved to debug.log
echo.
"%PYTHON%" main.py > debug.log 2>&1

echo.
echo ============================================================
echo  Application output (debug.log):
echo ============================================================
type debug.log
echo.
echo ============================================================
pause
