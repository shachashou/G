@echo off
rem C compiler for the venv-contained build. The compiler itself lives inside
rem .venv (pip-installed ziglang); nothing from the host Visual Studio install
rem is used. Target is fixed to the mingw-w64 ABI that Zig bundles so the Rust
rem bridge (x86_64-pc-windows-gnu) and the C++ engine share one ABI.
setlocal
set "ZIG=%~dp0..\..\.venv\Lib\site-packages\ziglang\zig.exe"
if not exist "%ZIG%" (
  echo zig compiler not found in .venv: "%ZIG%" 1>&2
  echo run: .venv\Scripts\python.exe -m pip install -r requirements.txt 1>&2
  exit /b 1
)
if not defined ZIG_GLOBAL_CACHE_DIR set "ZIG_GLOBAL_CACHE_DIR=%~dp0..\..\.venv\zig-cache"
if not defined ZIG_LOCAL_CACHE_DIR set "ZIG_LOCAL_CACHE_DIR=%~dp0..\..\.venv\zig-cache\local"
"%ZIG%" cc -target x86_64-windows-gnu %*
