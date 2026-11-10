#!/usr/bin/env python3
"""One-time bootstrap of the venv-contained Rust toolchain (ticket 07).

Everything is installed *under* ``.venv`` — no files from the user's global
``~/.cargo`` / ``~/.rustup`` or from Visual Studio are read at build time:

    .venv/rustup-init.exe        installer (downloaded once)
    .venv/rustup/                RUSTUP_HOME (toolchains, incl. rust-mingw libs)
    .venv/cargo/                 CARGO_HOME (cargo/rustc proxies, crate cache)

Two toolchains are installed:

    stable-x86_64-pc-windows-gnu    host + target (mingw ABI); the default
    stable-x86_64-pc-windows-msvc   kept only so rustup's channel metadata is
                                    complete; never used for linking

The GNU host matters because proc-macro crates (wasm-bindgen's dependencies)
are compiled for the host and otherwise need the Visual Studio linker. With
the GNU host they link through the venv ``rust-lld`` and the mingw import
libraries shipped by rustup's ``rust-mingw`` component.

Usage:
    .venv\\Scripts\\python.exe tools/build/bootstrap_toolchain.py [--force]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV = ROOT / ".venv"
RUSTUP_HOME = VENV / "rustup"
CARGO_HOME = VENV / "cargo"
RUSTUP_INIT = VENV / "rustup-init.exe"
RUSTUP = CARGO_HOME / "bin" / "rustup.exe"
GNU_TOOLCHAIN = "stable-x86_64-pc-windows-gnu"
GNU_TARGET = "x86_64-pc-windows-gnu"
RUSTUP_URL = (
    "https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/"
    "rustup-init.exe"
)
RUST_LLD = (
    RUSTUP_HOME / "toolchains" / GNU_TOOLCHAIN / "lib" / "rustlib"
    / GNU_TARGET / "bin" / "rust-lld.exe"
)


def env() -> dict[str, str]:
    values = dict(os.environ)
    values["RUSTUP_HOME"] = str(RUSTUP_HOME)
    values["CARGO_HOME"] = str(CARGO_HOME)
    values["PATH"] = str(CARGO_HOME / "bin") + os.pathsep + values.get("PATH", "")
    return values


def run(argv: list[str]) -> None:
    print("+", " ".join(argv), flush=True)
    result = subprocess.run(argv, env=env())
    if result.returncode != 0:
        raise SystemExit(f"command failed with exit code {result.returncode}")


def already_installed() -> bool:
    return RUSTUP.is_file() and RUST_LLD.is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="re-run the installer even if the toolchain exists")
    args = parser.parse_args()

    if os.name != "nt":
        print("bootstrap_toolchain.py is Windows-only; use the system Rust "
              "toolchain on other platforms", file=sys.stderr)
        return 2
    if not (VENV / "Scripts" / "python.exe").is_file():
        print(f"no venv at {VENV}; create it first (see requirements.txt)",
              file=sys.stderr)
        return 2
    if already_installed() and not args.force:
        print(f"Rust toolchain already installed in {RUSTUP_HOME}")
        print(f"linker: {RUST_LLD}")
        return 0

    RUSTUP_HOME.mkdir(parents=True, exist_ok=True)
    CARGO_HOME.mkdir(parents=True, exist_ok=True)
    if not RUSTUP_INIT.is_file():
        print(f"downloading {RUSTUP_URL}")
        urllib.request.urlretrieve(RUSTUP_URL, RUSTUP_INIT)

    # Host + target GNU toolchain with the rust-mingw component (import libs).
    run([str(RUSTUP_INIT), "-y", "--no-modify-path", "--profile", "minimal",
         "--default-toolchain", "stable", "--target", GNU_TARGET])
    # A GNU *host* toolchain is what makes proc-macro crates link without MSVC.
    run([str(RUSTUP), "toolchain", "install", GNU_TOOLCHAIN,
         "--profile", "minimal"])
    run([str(RUSTUP), "default", GNU_TOOLCHAIN])
    run([str(RUSTUP), "target", "add", GNU_TARGET, "--toolchain", GNU_TOOLCHAIN])

    if not RUST_LLD.is_file():
        print(f"rust-lld is still missing: {RUST_LLD}", file=sys.stderr)
        return 1
    print(f"Rust toolchain ready: {RUSTUP_HOME}")
    print(f"linker: {RUST_LLD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
