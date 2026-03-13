@echo off
cd /d "%~dp0"

echo Clearing Next.js cache...
if exist "%~dp0frontend\.next" rmdir /s /q "%~dp0frontend\.next"

echo Starting EU CRR RAG...

:: Start API server on port 8080 (matches frontend/.env.local)
start "CRR API" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate && uvicorn api.main:app --reload --port 8080"

:: Start frontend on port 3001 (3000 is occupied by a stuck process)
start "CRR Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev -- --port 3001"

:: Wait for servers to start, then open browser
timeout /t 8 /nobreak >nul
start http://localhost:3001
