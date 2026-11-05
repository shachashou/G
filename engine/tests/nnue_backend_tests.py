#!/usr/bin/env python3
"""Ticket 10: AVX2 vs scalar bit-exact parity and single-evaluation timing.

Both backends must produce exactly the same scores (and the same sparse feature
indices) as the Python integer reference. The benchmark prints ns/eval so the
"microsecond-scale, not a search bottleneck" requirement has a measured number.

Usage: python nnue_backend_tests.py [path to nnue_probe.exe]
"""
from __future__ import annotations

import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "trainer"))

import features  # noqa: E402
import weights_format as wf  # noqa: E402

PROBE = sys.argv[1] if len(sys.argv) > 1 else "nnue_probe.exe"
BACKENDS = ("scalar", "avx2")
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        FAILURES.append(message)


def make_weights(seed: int = 23) -> dict:
    rng = random.Random(seed)
    w0 = [0] * wf.W0_LEN
    for i in range(0, wf.W0_LEN, 1013):
        w0[i] = rng.randint(-8, 8)
    return {
        "W0": w0,
        "b0": [rng.randint(16, 48) for _ in range(wf.B0_LEN)],
        "W1": [rng.randint(-6, 6) for _ in range(wf.W1_LEN)],
        "b1": [rng.randint(16, 48) for _ in range(wf.B1_LEN)],
        "W2": [rng.randint(-30, 30) for _ in range(wf.W2_LEN)],
        "b2": [rng.randint(-20, 20) for _ in range(wf.B2_LEN)],
    }


def make_positions(count: int = 24, seed: int = 31):
    rng = random.Random(seed)
    result = []
    for n in range(count):
        cells = ["."] * 225
        order = list(range(225))
        rng.shuffle(order)
        stones = order[:1 + n]
        for i, cell in enumerate(stones):
            cells[cell] = "x" if i % 2 == 0 else "o"
        black = [i for i, c in enumerate(cells) if c == "x"]
        white = [i for i, c in enumerate(cells) if c == "o"]
        result.append((len(stones) % 2, features.sparse_indices(black, white),
                       "".join(cells)))
    return result


def probe(weights_path: Path, backend: str, mode: str,
          lines: list[str]) -> list[str]:
    result = subprocess.run(
        [PROBE, "--weights", str(weights_path), "--backend", backend, mode],
        input="\n".join(lines) + "\n", capture_output=True, text=True,
        timeout=240)
    check(result.returncode == 0,
          f"{backend}/{mode} exits 0 (stderr={result.stderr.strip()[:160]!r})")
    out = result.stdout.splitlines()
    check(len(out) == len(lines),
          f"{backend}/{mode} returns {len(lines)} lines (got {len(out)})")
    return out


def test_parity(weights_path: Path, weights: dict, positions) -> None:
    print(f"[backend bit-exact parity: {len(positions)} positions]")
    board_lines = [f"{stm} {text}" for stm, _, text in positions]
    infer_lines = [f"{stm} " + " ".join(str(i) for i in indices)
                   for stm, indices, _ in positions]
    reference = [wf.infer_int(weights, indices, stm)
                 for stm, indices, _ in positions]

    outputs = {}
    for backend in BACKENDS:
        feature_lines = probe(weights_path, backend, "features", board_lines)
        cpp_indices = [[int(v) for v in line.split()] if line.strip() else []
                       for line in feature_lines]
        for n, (_, expected, _) in enumerate(positions):
            check(cpp_indices[n] == expected,
                  f"{backend}: position {n} feature indices match Python")
        outputs[f"{backend}/infer"] = [int(line.strip()) for line in
                                      probe(weights_path, backend, "infer",
                                            infer_lines)]
        outputs[f"{backend}/board"] = [int(line.strip()) for line in
                                      probe(weights_path, backend, "board",
                                            board_lines)]

    for key, values in outputs.items():
        check(values == reference, f"{key} == infer_int, bit for bit")
    check(outputs["scalar/infer"] == outputs["avx2/infer"],
          "full sparse forward: scalar == AVX2, bit for bit")
    check(outputs["scalar/board"] == outputs["avx2/board"],
          "incremental accumulator: scalar == AVX2, bit for bit")
    check(len(set(outputs["avx2/infer"])) > 1,
          "sampled positions produce varied scores")


def test_benchmark(weights_path: Path, positions) -> None:
    print("[single evaluation cost]")
    iterations = 20000
    stm, _, text = positions[len(positions) // 2]
    line = f"{stm} {text} {iterations}"
    timings = {}
    for backend in BACKENDS:
        out = probe(weights_path, backend, "bench", [line])
        match = re.search(r"ns_per_eval=([0-9.]+)", out[0] if out else "")
        check(match is not None, f"{backend}: bench reports ns_per_eval")
        if match:
            timings[backend] = float(match.group(1))
            print(f"  {backend:6s}: {timings[backend]:8.2f} ns/eval "
                  f"({iterations} evaluations)")

    if len(timings) == 2:
        speedup = timings["scalar"] / max(1.0, timings["avx2"])
        print(f"  AVX2 speedup: {speedup:.2f}x")
        check(timings["avx2"] < 20000.0,
              "AVX2 single evaluation is microsecond-scale (< 20 us)")
        check(timings["avx2"] < timings["scalar"] * 0.95,
              f"AVX2 is measurably faster than scalar (speedup {speedup:.2f}x)")


def main() -> int:
    weights = make_weights()
    positions = make_positions()
    with tempfile.TemporaryDirectory(prefix="gomoku_nnue_backend_") as raw_tmp:
        path = Path(raw_tmp) / "weights.nnue"
        path.write_bytes(wf.pack_weights(weights))
        test_parity(path, weights, positions)
        test_benchmark(path, positions)

    if FAILURES:
        print(f"\n{len(FAILURES)} backend check(s) failed")
        return 1
    print("nnue_backend_tests: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
