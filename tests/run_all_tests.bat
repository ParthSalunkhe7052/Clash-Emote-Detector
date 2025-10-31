@echo off
REM Run all test suites

echo ========================================
echo Running Test Suite
echo ========================================

echo.
echo [1/2] Running Audio Manager Unit Tests...
python tests\test_audio_manager.py
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: Audio Manager tests
    exit /b 1
)

echo.
echo [2/2] Running Integration Tests...
python tests\test_integration.py
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: Integration tests
    exit /b 1
)

echo.
echo ========================================
echo All Tests Passed!
echo ========================================
