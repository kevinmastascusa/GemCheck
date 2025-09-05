@echo off
echo Starting GemCheck PSA Card Pre-Grader...
cd /d "%~dp0app"
python -m streamlit run main.py
pause