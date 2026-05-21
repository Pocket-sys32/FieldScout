@echo off
cd /d "%~dp0"
title Cache Creek Game Camera Project

:: Find Python - try PATH first, then known location
set PYTHON=
for %%P in (python.exe) do set PYTHON=%%~$PATH:P
if "%PYTHON%"=="" (
    if exist "C:\Users\RJ\AppData\Local\Programs\Python\Python314\python.exe" (
        set PYTHON=C:\Users\RJ\AppData\Local\Programs\Python\Python314\python.exe
    )
)

if "%PYTHON%"=="" (
    echo ERROR: Python not found.
    echo Please install Python from https://python.org and check "Add to PATH".
    pause
    exit /b 1
)

:: Install dependencies on first run
if not exist ".deps_installed" (
    echo Installing dependencies - one-time setup, may take several minutes...
    "%PYTHON%" -m pip install --upgrade pip
    "%PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        pause
        exit /b 1
    )
    echo. > .deps_installed
)

:: Launch
"%PYTHON%" main.py
if errorlevel 1 (
    echo.
    echo The application exited with an error.
    echo Run run_debug.bat for details.
    pause
)
