@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Hay chay setup_windows.bat truoc.
  pause
  exit /b 1
)
if not exist "data\demo\4_images\img_1.png" (
  .venv\Scripts\python.exe scripts\generate_demo_data.py
)
.venv\Scripts\python.exe main.py data\demo\4_images\img_1.png data\demo\4_images\img_2.png data\demo\4_images\img_3.png data\demo\4_images\img_4.png -o outputs\demo_4_images.png --save-debug
if errorlevel 1 (
 echo Ghép ảnh that bai.
) else (
 echo Hoan thanh: outputs\demo_4_images.png
)
pause
