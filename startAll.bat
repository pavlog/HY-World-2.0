@echo off
REM ============================================================
REM  AIWorldStudio / WorldStereo  -  start all servers
REM  Single resident server: _ws_step_server.py (port 5005)
REM   -> brings WorldStereo (fp8 transformer) + MoGe + the
REM      disk prompt-cache (_ws_prompt_cache, built into _ws_serve).
REM  Close this window to stop the server.
REM ============================================================
title WS Step Server  ::  http://127.0.0.1:5005

REM ---- force UTF-8 stdout (HW2 prints emoji; Windows cp1252 -> UnicodeEncodeError 500) ----
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM ---- WorldStereo working recipe (fp8 + sage + model-offload) ----
set WS_SAGE=1
set WS_FP8_UMT5=1
set WS_OFFLOAD=1
set WS_FP8_TRANSFORMER=1
set WS_FP8_FILE=D:/HF_MODELS/WorldStereo/worldstereo-memory-dmd/model_fp8.safetensors

set PYEXE=%USERPROFILE%\miniconda3\envs\worldmirror\python.exe
set SRVDIR=D:\HY-World-2.0

REM ---- free port 5005 if a previous instance is still running ----
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":5005 " ^| findstr LISTENING') do (
    echo Killing previous server on :5005 PID %%P ...
    taskkill /F /PID %%P >nul 2>&1
)

cd /d %SRVDIR%
echo.
echo Starting WorldStereo step server (fp8, ~1.5 min to load) ...
echo Open the client at  http://127.0.0.1:5005  once it prints "Running on ...".
echo.
"%PYEXE%" -u _ws_step_server.py

echo.
echo [server exited]  -  press a key to close.
pause >nul
