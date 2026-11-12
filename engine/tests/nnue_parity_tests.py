#!/usr/bin/env python3
"""Ticket 09 cross-language parity tests (C++ scalar NNUE vs Python spec).

Standard-library half (always runs):
  * C++ sparse feature indices == trainer/features.py on sampled positions
  * C++ full sparse forward == trainer/weights_format.infer_int, bit for bit
  * C++ incremental accumulator path == the same integer reference

Torch half (runs when installed):
  * PyTorch STE forward == C++ output exactly
  * PyTorch float forward == C++ output within the ticket-08 export tolerance,
    measured on the real exported weights when they are available locally

Usage: python nnue_parity_tests.py [path to nnue_probe.exe]
"""

import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "trainer"))

import binpack  # noqa: E402
import features  # noqa: E402
import weights_format as wf  # noqa: E402

PROBE = sys.argv[1] if len(sys.argv) > 1 else "nnue_probe.exe"
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)
        print(f"FAIL: {message}")


def has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def make_small_weights() -> dict:
    """Hand-computable weights shared with training_tests.py / nnue_tests.cpp."""
    weights = {
        "W0": [0] * wf.W0_LEN, "b0": [0] * wf.B0_LEN,
        "W1": [0] * wf.W1_LEN, "b1": [0] * wf.B1_LEN,
        "W2": [0] * wf.W2_LEN, "b2": [0] * wf.B2_LEN,
    }
    weights["W0"][112 * wf.N_ACC] = 5
    weights["b0"][0] = 3
    weights["W1"][0] = 7
    weights["b1"][0] = 1
    weights["W2"][0] = 11
    weights["W2"][wf.N_L1] = -9
    weights["b2"] = [2, 5]
    return weights


def make_random_weights(seed: int = 11) -> dict:
    """Integer weights in the same ranges as a trained checkpoint."""
    rng = random.Random(seed)
    w0 = [0] * wf.W0_LEN
    for i in range(0, wf.W0_LEN, 997):
        w0[i] = rng.randint(-8, 8)
    return {
        "W0": w0,
        "b0": [rng.randint(16, 48) for _ in range(wf.B0_LEN)],
        "W1": [rng.randint(-6, 6) for _ in range(wf.W1_LEN)],
        "b1": [rng.randint(16, 48) for _ in range(wf.B1_LEN)],
        "W2": [rng.randint(-30, 30) for _ in range(wf.W2_LEN)],
        "b2": [rng.randint(-20, 20) for _ in range(wf.B2_LEN)],
    }


def synthetic_positions(count: int = 24, seed: int = 5):
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


def real_positions(count: int = 16):
    """Sample the same distribution the checkpoint was trained on, if present."""
    path = ROOT / "data" / "raw" / "bench-d4" / "train.binpack"
    if not path.exists():
        return []
    result = []
    with binpack.BinpackReader(path) as reader:
        total = reader.available
        stride = max(1, total // count)
        for index in range(0, total, stride):
            if len(result) >= count:
                break
            record = reader[index]
            cells = ["."] * 225
            for feature in record.feature_indices():
                plane, cell = divmod(feature, 225)
                cells[cell] = "x" if plane == 0 else "o"
            result.append((record.stm, record.feature_indices(),
                           "".join(cells)))
    return result


def board_lines(positions) -> list[str]:
    return [f"{stm} {text}" for stm, _, text in positions]


def infer_lines(positions) -> list[str]:
    return [f"{stm} " + " ".join(str(i) for i in indices)
            for stm, indices, _ in positions]


def run_probe(weights_path: Path, mode: str, lines: list[str]) -> list[str]:
    result = subprocess.run(
        [PROBE, "--weights", str(weights_path), mode],
        input="\n".join(lines) + "\n", capture_output=True, text=True,
        timeout=120)
    check(result.returncode == 0,
          f"probe {mode} exits 0 (stderr={result.stderr.strip()[:200]!r})")
    out = result.stdout.splitlines()
    check(len(out) == len(lines),
          f"probe {mode} returns {len(lines)} lines (got {len(out)})")
    return out


def parse_index_lines(lines: list[str]) -> list[list[int]]:
    return [[int(v) for v in line.split()] if line.strip() else []
            for line in lines]


def test_cross_language(weights_path: Path, weights: dict, positions) -> list[int]:
    print(f"[cross-language: {len(positions)} positions]")
    cpp_indices = parse_index_lines(
        run_probe(weights_path, "features", board_lines(positions)))
    for n, (_, expected, _) in enumerate(positions):
        check(cpp_indices[n] == expected,
              f"position {n}: C++ feature indices match features.sparse_indices")

    cpp_infer = [int(line.strip()) for line in
                 run_probe(weights_path, "infer", infer_lines(positions))]
    cpp_board = [int(line.strip()) for line in
                 run_probe(weights_path, "board", board_lines(positions))]
    reference = [wf.infer_int(weights, indices, stm)
                 for stm, indices, _ in positions]
    for n, (got, want) in enumerate(zip(cpp_infer, reference)):
        check(got == want,
              f"position {n}: scalar forward {got} == infer_int {want}")
    for n, (got, want) in enumerate(zip(cpp_board, reference)):
        check(got == want,
              f"position {n}: incremental accumulator {got} == infer_int {want}")
    check(cpp_board == cpp_infer, "incremental path == full sparse path")
    check(len(set(cpp_infer)) > 1, "sampled positions produce varied scores")
    return cpp_infer


def test_torch(small_path: Path, small: dict, random_path: Path,
               random_weights: dict, positions) -> None:
    import torch
    import nnue

    print("[torch: STE / float alignment]")
    for label, path, weights in (("small", small_path, small),
                                 ("random", random_path, random_weights)):
        model = nnue.NNUEModel.from_quantized_weights(weights)
        x = torch.tensor([features.dense_vector(indices)
                          for _, indices, _ in positions], dtype=torch.float32)
        stm = torch.tensor([s for s, _, _ in positions], dtype=torch.long)
        with torch.no_grad():
            quant = [int(round(float(v))) for v in model(x, stm, quantized=True)]
        cpp = [int(line.strip()) for line in
               run_probe(path, "infer", infer_lines(positions))]
        check(quant == cpp, f"{label} weights: PyTorch STE == C++ scalar output")

    real = ROOT / "weights" / "nnue_v1.nnue"
    if not real.exists():
        print("real weights not present: float-parity acceptance SKIPPED "
              "(run trainer/export.py to enable)")
        return
    real_weights, meta = wf.read_weights(real)
    check(meta["version"] == 1 and meta["qa"] == wf.QA,
          f"real weights header matches spec ({meta['version']})")
    tolerance = 2
    report_path = ROOT / "weights" / "export_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        tolerance = report["parity"]["suggested_cpp_tolerance"]
    sample = real_positions() or positions
    model = nnue.NNUEModel.from_quantized_weights(real_weights)
    x = torch.tensor([features.dense_vector(indices)
                      for _, indices, _ in sample], dtype=torch.float32)
    stm = torch.tensor([s for s, _, _ in sample], dtype=torch.long)
    with torch.no_grad():
        float_eval = [float(v) for v in model(x, stm, quantized=False)]
        quant_eval = [int(round(float(v))) for v in model(x, stm, quantized=True)]
    cpp = [int(line.strip()) for line in
           run_probe(real, "infer", infer_lines(sample))]
    check(quant_eval == cpp, "real weights: PyTorch STE == C++ scalar output")
    diffs = [abs(a - b) for a, b in zip(float_eval, cpp)]
    worst = max(diffs) if diffs else 0.0
    mae = sum(diffs) / max(1, len(diffs))
    print(f"  float vs C++: MAE {mae:.4f}, max {worst:.4f}, "
          f"tolerance <= {tolerance}, samples {len(sample)}")
    check(worst <= tolerance,
          f"PyTorch float vs C++ within <= {tolerance} (max {worst:.4f})")


def main() -> int:
    small = make_small_weights()
    random_weights = make_random_weights()
    positions = synthetic_positions()
    real = real_positions()
    if real:
        print(f"real binpack sampled: {len(real)} positions")
        positions += real

    with tempfile.TemporaryDirectory(prefix="gomoku_nnue_parity_") as raw_tmp:
        tmp = Path(raw_tmp)
        small_path = tmp / "small.nnue"
        random_path = tmp / "random.nnue"
        small_path.write_bytes(wf.pack_weights(small))
        random_path.write_bytes(wf.pack_weights(random_weights))
        check(len(small_path.read_bytes()) == wf.FILE_LEN,
              "synthetic small weight file has the documented size")

        hand = [int(line.strip()) for line in
                run_probe(small_path, "infer", ["0 112", "1 112", "0 "])]
        check(hand == [2, 5, 2],
              f"hand-computed integer forward matches Python (got {hand})")

        test_cross_language(small_path, small, positions)
        test_cross_language(random_path, random_weights, positions)
        if has_torch():
            test_torch(small_path, small, random_path, random_weights, positions)
        else:
            print("torch not installed: STE / float alignment tests SKIPPED")

    if FAILURES:
        print(f"\n{len(FAILURES)} NNUE parity check(s) failed")
        return 1
    print("nnue_parity_tests: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
