@echo off
REM Clash Royale Emote Detector - Web App Launcher
REM Author: Parth Salunkhe

echo ========================================
echo  Clash Emote Detector v2.2
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run 'setup.bat' first to install dependencies.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if PyTorch is installed
echo Checking dependencies...
python -c "import torch" 2>nul
if errorlevel 1 (
    echo.
    echo ⚠️  PyTorch not found! Installing now...
    echo This is a one-time setup. May take 2-3 minutes.
    echo.
    
    REM Install PyTorch (CPU version for faster download)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
    if errorlevel 1 (
        echo.
        echo ❌ Failed to install PyTorch!
        echo Please check your internet connection and try again.
        pause
        exit /b 1
    )
    
    echo.
    echo ✅ PyTorch installed successfully!
    echo.
)



echo.
echo ========================================
echo  Ready to Launch!
echo ========================================
echo.
echo Features:
echo   - Real-time emote detection
echo   - 7 emotes supported + custom emotes
echo   - Switch between multiple AI models
echo   - Manage emotes (upload custom ones)
echo   - Enhanced data collection
echo.
echo Camera Setup:
echo   - Camera 0: Laptop/Built-in Camera
echo   - Camera 1: DroidCam (if running)
echo   - Switch cameras in Settings page
echo.
echo Starting Flask web server...
echo.
echo Open your browser at: http://localhost:5000
echo.
echo Available Pages:
echo   /           - Live Detection
echo   /capture    - Enhanced Data Collection
echo   /manage     - Manage Emotes
echo   /settings   - Settings ^& Configuration
echo.
echo Press Ctrl+C to stop server
echo ========================================
echo.

python webapp\app.py

call venv\Scripts\deactivate.bat

echo.
echo Server stopped.
pause
