# vcf-bridge: C ABI to the vendored upstream VCF solver

Ticket 03 makes the open-source Renju solver [quintet](../../vendor/quintet/UPSTREAM.md)
(MIT, pinned commit) the only VCF implementation the engine uses. The engine is
C++, the solver is Rust, so this crate is the smallest possible seam between
them:

- `quintet = { path = "../../vendor/quintet" }` — an unmodified upstream copy.
- `crate-type = ["staticlib"]` — CMake runs `cargo build --release` and links
  `target/release/libvcf_bridge.a`. On Windows the build uses the venv-contained
  Rust toolchain (`stable-x86_64-pc-windows-gnu` under `.venv/rustup`, plus
  `.venv/cargo`) and the venv `rust-lld`, so the ABI matches the Zig-built C++
  engine (mingw-w64) and no Visual Studio / global cargo is read.
- No search logic lives here: the crate converts the engine's cell array into
  the upstream `Board`, calls `solve_with_stats(SolveMode::VCFDFS, …)`, and
  copies the outcome back.

## ABI

`src/lib.rs` declares `#[repr(C)]` structs that `engine/src/vcf_bridge.h`
mirrors field for field; `engine/src/vcf.cpp` is the only consumer.

```text
VcfBridgeQuery { cells[225], side_to_move, rule, max_attack_plies,
                 max_nodes, time_limit_ms }
VcfBridgeResult { status, end_kind, end_a, end_b, first_move,
                  path_len, path[512], nodes, elapsed_ms, message[192] }
```

- Cell layout is the engine's: index = y * 15 + x, value 0/1/2.
- Points in `path`/`end_*`/`first_move` are engine move indices (y * 15 + x),
  not upstream `u8` codes.
- `status`: `PROVEN`, `DISPROVEN`, `BUDGET_EXHAUSTED`, `INTERRUPTED`,
  `INVALID_INPUT`, `UNSUPPORTED`.
- Wall-clock limits are implemented by running the query on a worker thread and
  abandoning it on timeout: upstream deliberately has no clock (it compiles for
  wasm). A timed-out query never reports a proof.

## Build and test

CMake builds this crate automatically (venv-contained toolchain on Windows):

```powershell
.venv\Scripts\python.exe tools\build\bootstrap_toolchain.py   # once
.venv\Scripts\python.exe tools\build\build.py --target vcf_bridge
```

Standalone from the venv (PowerShell; sets the same variables CMake sets):

```powershell
$env:RUSTUP_HOME = "$PWD\.venv\rustup"
$env:CARGO_HOME  = "$PWD\.venv\cargo"
$env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER = "$PWD\.venv\rustup\toolchains\stable-x86_64-pc-windows-gnu\lib\rustlib\x86_64-pc-windows-gnu\bin\rust-lld.exe"
$env:RUSTFLAGS = "-Clink-self-contained=yes"
& .venv\cargo\bin\cargo.exe build --release --manifest-path engine\vcf-bridge\Cargo.toml
& .venv\cargo\bin\cargo.exe test  --release --manifest-path engine\vcf-bridge\Cargo.toml
```

The crate's unit tests cover the encoding direction (a known VCF line's engine
indices), budget exhaustion versus disproof, the deadline, invalid input, and
null pointers; `ctest` runs them as `vcf_bridge_tests`.

Do not edit `vendor/quintet/src`: upgrade by re-vendoring a newer commit and
updating `vendor/quintet/UPSTREAM.md`, then re-run the tests.
