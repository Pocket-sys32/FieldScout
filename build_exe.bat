@echo off
cd /d "%~dp0"
:: Build a standalone Windows .exe with PyInstaller.
:: Run this ONCE on a developer machine.  Volunteers just need the dist\ folder.

title Cache Creek Game Camera Project - EXE Build

:: Find Python
set PYTHON=
for %%P in (python.exe) do set PYTHON=%%~$PATH:P
if "%PYTHON%"=="" (
    if exist "C:\Users\RJ\AppData\Local\Programs\Python\Python314\python.exe" (
        set PYTHON=C:\Users\RJ\AppData\Local\Programs\Python\Python314\python.exe
    )
)
if "%PYTHON%"=="" (
    echo ERROR: Python not found.
    pause
    exit /b 1
)

echo ============================================================
echo  Cache Creek Game Camera Project - EXE Build
echo  Using Python: %PYTHON%
echo ============================================================
echo.
echo NOTE: This can take 10-20 minutes. Please wait...
echo.

:: Clean previous build artefacts
if exist build  rmdir /s /q build
if exist dist   rmdir /s /q dist

"%PYTHON%" -m PyInstaller ^
    --name "CacheCreek_GameCamera" ^
    --onedir ^
    --windowed ^
    --icon NONE ^
    --add-data "fsvcc_detector;fsvcc_detector" ^
    --add-data "assets;assets" ^
    --collect-all torch ^
    --collect-all torchvision ^
    --collect-all PytorchWildlife ^
    --collect-all transformers ^
    --collect-all customtkinter ^
    --collect-all cv2 ^
    --collect-all gspread ^
    --collect-all google ^
    --hidden-import "PIL._tkinter_finder" ^
    --collect-all easyocr ^
    --hidden-import "easyocr" ^
    --hidden-import "pymediainfo" ^
    main.py

if errorlevel 1 (
    echo.
    echo BUILD FAILED.  Check the output above for errors.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Build complete!
echo  Distribute the entire  dist\CacheCreek_GameCamera\  folder.
echo  Volunteers run:  CacheCreek_GameCamera.exe
echo ============================================================
pause
