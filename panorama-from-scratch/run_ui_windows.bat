@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Hay chay setup_windows.bat truoc.
 pause
 exit /b 1
)
.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
if errorlevel 1 (
 echo Cai dat Streamlit that bai.
 pause
 exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
pause
