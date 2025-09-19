@echo off
cd /d "C:\Users\dlaev\Personal Projects\PSA PREGRADER"
call venv\Scripts\activate.bat
python test_scanner.py
echo Test finished with exit code: %ERRORLEVEL%
pause