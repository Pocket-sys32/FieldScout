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

:: Install dependencies if not done yet
if not exist ".deps_installed" (
    echo Installing dependencies - this may take several minutes...
    echo.
    "%PYTHON%" -m pip install --upgrade pip
    "%PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Dependency install failed. See above.
        pause
        exit /b 1
    )
    echo. > .deps_installed
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
