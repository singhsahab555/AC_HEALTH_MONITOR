@echo off
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501') do (
    taskkill /PID %%a /F
)
if not errorlevel 1 echo Streamlit on port 8501 stopped.
if errorlevel 1 echo No process was running on port 8501.
