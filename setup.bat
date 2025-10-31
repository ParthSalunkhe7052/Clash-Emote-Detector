@echo off
REM Clash Royale Emote Detector - Setup Script
REM Author: Parth Salunkhe
REM This script sets up the Python environment and installs all dependencies

echo ========================================
echo  Clash Emote Detector - Setup v2.2
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo [2/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo [2/4] Virtual environment already exists, skipping...
)
echo.

REM Activate virtual environment and upgrade pip
echo [3/4] Upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo [4/4] Installing dependencies from requirements.txt...
echo.
if exist "requirements.txt" (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo ERROR: requirements.txt not found!
    pause
    exit /b 1
)
echo.

echo Installing PyTorch (CPU version - smaller download)...
echo This may take 2-3 minutes...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo.
    echo WARNING: PyTorch installation failed!
    echo This is OK - it will be installed automatically when you run the app.
    echo.
)
echo.

echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo Your Clash Emote Detector is ready to run!
echo.
echo To start the application:
echo   - Double-click 'run.bat' 
echo   - OR run: python webapp\app.py
echo.
echo To collect training data:
echo   - Open browser at http://localhost:5000/collect
echo.
pause
