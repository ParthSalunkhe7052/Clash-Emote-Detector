@echo off
REM Retrain Neural Network Model with Improvements
REM Author: Parth

echo ========================================
echo  Retraining Improved Neural Network
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run 'setup.bat' first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Checking training data...
if not exist "pose_data\pose_features_latest.npy" (
    echo ERROR: Training data not found!
    echo Please run collect_training_data.py first to collect samples.
    pause
    exit /b 1
)

echo.
echo Starting training with improvements:
echo  - Data augmentation (noise, scaling, flipping)
echo  - Class balancing for fair training
echo  - 150 epochs with early stopping
echo  - Temporal smoothing enabled
echo.
echo This may take 10-20 minutes...
echo.

python training\train_neural_model.py

if errorlevel 1 (
    echo.
    echo ❌ Training failed!
    pause
    exit /b 1
)

echo.
echo ✅ Training complete!
echo.
echo Copying trained model to root directory...
copy trained_model\pose_neural_classifier.pth pose_neural_classifier.pth

echo.
echo ========================================
echo  🎉 Model retrained successfully!
echo ========================================
echo.
echo The improved model is now ready to use.
echo Run 'run.bat' to start the web application.
echo.
pause
