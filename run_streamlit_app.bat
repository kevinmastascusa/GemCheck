@echo off
echo.
echo  ========================================
echo   GemCheck - PSA Card Pre-grader Web App
echo  ========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if streamlit is installed
python -c "import streamlit; print('Streamlit installed')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing Streamlit...
    pip install streamlit
)

echo.
echo Starting GemCheck Streamlit Web Interface...
echo.
echo Features:
echo - Upload and analyze card images
echo - Detailed PSA-style grading reports
echo - Advanced configuration options
echo - Professional analysis visualizations
echo.
echo The app will open in your web browser automatically.
echo Use Ctrl+C to stop the server.
echo.

REM Change to app directory and run main.py
cd /d "%~dp0"
python -m streamlit run app/main.py

pause