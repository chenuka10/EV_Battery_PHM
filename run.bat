@echo off
echo ==============================================================================
echo  EV Battery Prognostics ^& Health Management (PHM) - Production Server
echo  SLIIT IT3051 Mini Project 2026 ^| Group Necrons
echo ==============================================================================
echo.
echo Activating virtual environment and starting FastAPI + Vanilla Frontend...
echo.
echo [1] Production Dashboard: http://localhost:8000
echo [2] Interactive OpenAPI:  http://localhost:8000/docs
echo [3] Streamlit Fallback:   streamlit run app.py
echo.
echo Press Ctrl+C to terminate the server.
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
) else (
    python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
)
pause
