@echo off
REM Double-click to start the VGM crowding app (needs Python with: pip install -r requirements.txt)
cd /d "%~dp0"
python -m streamlit run app.py
pause
