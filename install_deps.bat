@echo off
cd /d "C:\Users\dlaev\Personal Projects\PSA PREGRADER"
call venv\Scripts\activate.bat
pip install opencv-python pyautogui mss pytesseract requests beautifulsoup4 Pillow numpy
echo Dependencies installed!
pause