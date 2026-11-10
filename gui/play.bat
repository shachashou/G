@echo off
setlocal
cd /d "%~dp0.."

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] venv not found. Create it first:
  echo         py -3.12 -m venv .venv
  echo         Select requirements-amd.txt, requirements-nvidia.txt or requirements-cpu.txt; see README.md.
  pause
  exit /b 1
)

if not exist "build\engine\gomoku_nnue.exe" (
  echo [ERROR] engine not built. Build with the venv cmake + Ninja toolchain first ^(see README.md^).
  pause
  exit /b 1
)

"%PY%" "gui\gomoku_gui.py" %*
