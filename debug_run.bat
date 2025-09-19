@echo off
cd /d "C:\Users\dlaev\Personal Projects\PSA PREGRADER"
call venv\Scripts\activate.bat
echo Running scanner...
python pokemon_card_price_scanner.py 2>&1
echo.
echo Exit code: %ERRORLEVEL%
echo Press any key to see errors...
pause