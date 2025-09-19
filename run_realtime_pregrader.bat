@echo off
echo.
echo  =====================================
echo   GemCheck - Real-Time PSA Pre-grader
echo  =====================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if opencv is installed
python -c "import cv2; print('OpenCV version:', cv2.__version__)" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing OpenCV...
    pip install opencv-python
)

REM Check if PIL is installed
python -c "import PIL; print('PIL installed')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing Pillow...
    pip install Pillow
)

echo.
echo Starting GemCheck Real-Time Analysis...
echo.
echo Features:
echo - Live computational photography overlays
echo - Real-time centering analysis
echo - Professional PSA-style grading
echo - Capture and export functionality
echo.
echo Controls:
echo - Start Camera: Begin live analysis
echo - Stop Camera: Stop camera feed  
echo - Capture & Analyze: Save current analysis
echo.
echo Press Ctrl+C or close window to exit
echo.

python real_time_pregrader.py

pause