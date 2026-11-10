#!/usr/bin/env python3
"""Build and test GomokuNNUE with the toolchain contained in ``.venv``.

On Windows nothing from Visual Studio, the Windows SDK or a global Rust
install is read:

    C/C++  pip-installed ziglang (clang + libc++ + mingw-w64), via
           tools/build/zig-cc.cmd and zig-cxx.cmd
    Rust   .venv/rustup + .venv/cargo (stable-x86_64-pc-windows-gnu), linking
           with the venv rust-lld; native crates use the zig wrappers

First-time setup::

    .venv\\Scripts\\python.exe -m pip install -r requirements.txt
    .venv\\Scripts\\python.exe tools\\build\\bootstrap_toolchain.py

Then::

    .venv\\Scripts\\python.exe tools\\build\\build.py --clean
    .venv\\Scripts\\python.exe tools\\build\\build.py --test
    .venv\\Scripts\\python.exe tools\\build\\build.py --test -R value_net
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV = ROOT / ".venv"
BUILD = ROOT / "build"


def venv_bin(name: str) -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / name
    return VENV / "bin" / name


def cmake_path() -> Path:
    return venv_bin("cmake.exe" if os.name == "nt" else "cmake")


def ctest_path() -> Path:
    return venv_bin("ctest.exe" if os.name == "nt" else "ctest")


def zig_path() -> Path:
    if os.name == "nt":
        return VENV / "Lib" / "site-packages" / "ziglang" / "zig.exe"
    return VENV / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" \
        / "site-packages" / "ziglang" / "zig"


def rust_ready() -> bool:
    toolchain = VENV / "rustup" / "toolchains" / "stable-x86_64-pc-windows-gnu"
    return (VENV / "cargo" / "bin" / "cargo.exe").is_file() and (
        toolchain / "lib" / "rustlib" / "x86_64-pc-windows-gnu" / "bin"
        / "rust-lld.exe").is_file()


def fail(message: str) -> int:
    print(f"build.py: {message}", file=sys.stderr)
    return 2


def clean_build() -> int:
    target = BUILD.resolve()
    expected = (ROOT / "build").resolve()
    if target != expected or target.parent != ROOT.resolve():
        return fail(f"refusing to delete unexpected path: {target}")
    if target.exists():
        print(f"removing {target}")
        shutil.rmtree(target)
    return 0


def environment() -> dict[str, str]:
    values = dict(os.environ)
    scripts = str(VENV / "Scripts") if os.name == "nt" else str(VENV / "bin")
    values["PATH"] = scripts + os.pathsep + values.get("PATH", "")
    if os.name == "nt":
        values["ZIG_GLOBAL_CACHE_DIR"] = str(VENV / "zig-cache")
        values["ZIG_LOCAL_CACHE_DIR"] = str(VENV / "zig-cache" / "local")
        values["RUSTUP_HOME"] = str(VENV / "rustup")
        values["CARGO_HOME"] = str(VENV / "cargo")
        values["RUSTUP_TOOLCHAIN"] = "stable-x86_64-pc-windows-gnu"
    return values


def run(argv: list[str]) -> int:
    print("+", " ".join(argv), flush=True)
    result = subprocess.run(argv, env=environment(), cwd=ROOT)
    return result.returncode


def configured_with_zig() -> bool:
    cache = BUILD / "CMakeCache.txt"
    if not cache.is_file():
        return True
    text = cache.read_text(encoding="utf-8", errors="replace")
    # The project enables only CXX, so CMake does not cache a C compiler.
    return "zig-cxx" in text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", action="store_true",
                        help="remove the build directory first")
    parser.add_argument("--test", action="store_true", help="run ctest")
    parser.add_argument("--test-regex", default=None,
                        help="passed to ctest -R")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--target", default=None,
                        help="build only this cmake target")
    args = parser.parse_args()

    for tool in (cmake_path(), ctest_path(), venv_bin(
            "python.exe" if os.name == "nt" else "python")):
        if not tool.is_file():
            return fail(f"missing venv tool: {tool}")
    if not zig_path().is_file():
        return fail(
            f"zig compiler is not installed in the venv ({zig_path()}).\n"
            "  run: .venv\\Scripts\\python.exe -m pip install -r requirements.txt")
    if os.name == "nt" and not rust_ready():
        return fail(
            "Rust toolchain is not installed in the venv.\n"
            "  run: .venv\\Scripts\\python.exe tools\\build\\bootstrap_toolchain.py")

    if args.clean:
        code = clean_build()
        if code != 0:
            return code
    if not configured_with_zig():
        return fail(
            "the existing build directory was configured with another "
            "compiler.\n  re-run with --clean (it is generated output)")

    configure = [
        str(cmake_path()), "-S", str(ROOT), "-B", str(BUILD),
        "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_MAKE_PROGRAM={venv_bin('ninja.exe' if os.name == 'nt' else 'ninja')}",
    ]
    if os.name == "nt":
        configure += [
            f"-DCMAKE_CXX_COMPILER={ROOT / 'tools' / 'build' / 'zig-cxx.cmd'}",
        ]
    code = run(configure)
    if code != 0:
        return code

    build = [str(cmake_path()), "--build", str(BUILD)]
    if args.target:
        build += ["--target", args.target]
    if args.jobs > 0:
        build += ["--parallel", str(args.jobs)]
    code = run(build)
    if code != 0:
        return code

    if args.test:
        tests = [str(ctest_path()), "--test-dir", str(BUILD),
                 "--output-on-failure"]
        if args.test_regex:
            tests += ["-R", args.test_regex]
        code = run(tests)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
