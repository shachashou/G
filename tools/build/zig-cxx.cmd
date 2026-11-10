@echo off
rem C++ compiler for the venv-contained build (see zig-cc.cmd). clang/libc++
rem are bundled inside the pip-installed ziglang package under .venv.
setlocal
set "ZIG=%~dp0..\..\.venv\Lib\site-packages\ziglang\zig.exe"
if not exist "%ZIG%" (
  echo zig compiler not found in .venv: "%ZIG%" 1>&2
  echo run: .venv\Scripts\python.exe -m pip install -r requirements.txt 1>&2
  exit /b 1
)
if not defined ZIG_GLOBAL_CACHE_DIR set "ZIG_GLOBAL_CACHE_DIR=%~dp0..\..\.venv\zig-cache"
if not defined ZIG_LOCAL_CACHE_DIR set "ZIG_LOCAL_CACHE_DIR=%~dp0..\..\.venv\zig-cache\local"
"%ZIG%" c++ -target x86_64-windows-gnu %*
