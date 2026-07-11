@echo off
REM One-click launcher for the demo. Uses the project's own environment so it
REM cannot hit "No module named ..." errors from the system Python.
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
    echo .venv missing - run:  python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
".venv\Scripts\streamlit.exe" run app\streamlit_app.py
pause
