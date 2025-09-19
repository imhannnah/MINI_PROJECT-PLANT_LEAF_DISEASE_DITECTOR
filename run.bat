@echo off
echo Activating virtual environment...
call .\.venv\Scripts\activate

echo Starting Flask app...
start http://127.0.0.1:5000/
python app.py

pause
