# vcf-bridge: C ABI to the vendored upstream VCF solver

Ticket 03 makes the open-source Renju solver [quintet](../../vendor/quintet/UPSTREAM.md)
(MIT, pinned commit) the only VCF implementation the engine uses. The engine is
C++, the solver is Rust, so this crate is the smallest possible seam between
them:

- `quintet = { path = "../../vendor/quintet" }` — an unmodified upstream copy.
- `crate-type = ["staticlib"]` — CMake runs `cargo build --release` and links
  `target/release/vcf_bridge.lib` (`.a` on POSIX).
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

CMake builds this crate automatically:

```powershell
.venv\Scripts\cmake.exe --build build
```

Standalone:

```powershell
cargo build --release --manifest-path engine/vcf-bridge/Cargo.toml
cargo test  --release --manifest-path engine/vcf-bridge/Cargo.toml
```

The crate's unit tests cover the encoding direction (a known VCF line's engine
indices), budget exhaustion versus disproof, the deadline, invalid input, and
null pointers; `ctest` runs them as `vcf_bridge_tests`.

Do not edit `vendor/quintet/src`: upgrade by re-vendoring a newer commit and
updating `vendor/quintet/UPSTREAM.md`, then re-run the tests.
