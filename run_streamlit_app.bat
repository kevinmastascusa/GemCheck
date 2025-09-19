@echo off
echo Starting PSA Card Pre-grader Streamlit App...
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
echo Starting Streamlit PSA Pre-grader Web App...
echo.
echo The app will open in your web browser automatically.
echo Use Ctrl+C to stop the server.
echo.

REM Change to app directory and run main.py
cd /d "%~dp0"
python -m streamlit run app/main.py

pause