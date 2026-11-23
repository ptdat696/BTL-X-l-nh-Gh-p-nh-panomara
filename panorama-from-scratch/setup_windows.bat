@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 py -3.12 -m venv .venv
  if errorlevel 1 py -m venv .venv
  if errorlevel 1 (
    echo Khong tao duoc virtual environment. Hay cai Python 3.11/3.12.
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Cai dat that bai.
  pause
  exit /b 1
)
echo Da cai xong. Chay run_demo_windows.bat de demo.
pause
