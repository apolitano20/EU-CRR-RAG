@echo off
cd /d "%~dp0"

echo Starting CRR Eval Dashboard...

start "CRR Eval Dashboard" cmd /k ".venv\Scripts\streamlit.exe run evals\dashboard.py --server.port 8501 --server.headless true"

timeout /t 4 /nobreak >nul
start http://localhost:8501
