@echo off
setlocal
cd /d "%~dp0.."

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] venv not found. Create it first:
  echo         py -3.12 -m venv .venv
  echo         .venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
  pause
  exit /b 1
)

if not exist "build\engine\gomoku_nnue.exe" (
  echo [ERROR] engine not built. Build with the venv cmake + Ninja toolchain first ^(see README.md^).
  pause
  exit /b 1
)

"%PY%" "gui\gomoku_gui.py" %*
