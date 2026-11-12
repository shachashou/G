#!/usr/bin/env python3
"""Ticket 07 cross-language parity and export fidelity for the deployed model.

Standard-library half (always runs):
  * C++ sparse feature indices == trainer/features.py
  * C++ scalar sparse forward == trainer/value_format.infer_int, bit for bit
  * C++ full first-layer rebuild (``board``) == the same integer reference
  * C++ public accumulator rebuild (``acc``) == the same integer reference
  * side-to-move sign conversion in ``score`` output
  * malformed / incompatible GMVALUE1 files are rejected by the probe

Torch half (runs when installed and the local demo model exists):
  * the exported integer model is the training model it claims to be
    (recomputed model_id matches the metadata)
  * C++ value = tanh(z / 2^ea) matches the original PyTorch model within the
    frozen trainer/value_format.VALUE_TOLERANCE

Usage: python value_parity_tests.py [path to value_probe.exe]
"""

import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "trainer"))

import features  # noqa: E402
import value_format as vf  # noqa: E402

PROBE = sys.argv[1] if len(sys.argv) > 1 else "value_probe.exe"
REAL_MODEL = ROOT / "weights" / "value-demo" / "model.gmvalue"
REAL_CHECKPOINTS = [ROOT / "weights" / "value-demo" / "last.pt",
                    ROOT / "weights" / "value-demo" / "best.pt"]
POOL = ROOT / "data" / "pool" / "pool.sqlite3"
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        FAILURES.append(message)


def has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def board_text(cells: list[int]) -> str:
    return "".join({0: ".", 1: "x", 2: "o"}[int(c)] for c in cells)


def synthetic_positions(count: int = 20, seed: int = 5):
    """Legal-ish boards: black to move when counts are equal, else white."""
    rng = random.Random(seed)
    result = []
    for n in range(1, count + 1):
        cells = [0] * 225
        order = list(range(225))
        rng.shuffle(order)
        stones = order[:n * 2]
        black = []
        white = []
        for i, cell in enumerate(stones):
            if i % 2 == 0:
                cells[cell] = 1
                black.append(cell)
            else:
                cells[cell] = 2
                white.append(cell)
        result.append((0, features.sparse_indices(black, white),
                       board_text(cells)))
    return result


def pool_positions(count: int = 24) -> list[tuple[int, list[int], str]]:
    """Read committed sample-pool positions; the parity target is position-free."""
    if not POOL.is_file():
        return []
    result = []
    try:
        connection = sqlite3.connect(f"file:{POOL}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        rows = connection.execute(
            "SELECT board, side_to_move FROM samples ORDER BY id LIMIT ?",
            (count * 7,)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()
    stride = max(1, len(rows) // count)
    for cells_blob, side in rows[::stride]:
        cells = list(cells_blob)
        if len(cells) != 225 or any(c not in (0, 1, 2) for c in cells):
            continue
        stm = 0 if side == "black" else 1
        black = [i for i, c in enumerate(cells) if c == 1]
        white = [i for i, c in enumerate(cells) if c == 2]
        expected_stm = 0 if len(black) == len(white) else 1
        if stm != expected_stm:
            continue
        result.append((stm, features.sparse_indices(black, white),
                       board_text(cells)))
        if len(result) >= count:
            break
    return result


def run_probe(model: Path, mode: str, lines: list[str],
              backend: str | None = None) -> list[str]:
    argv = [PROBE, "--model", str(model)]
    if backend is not None:
        argv += ["--backend", backend]
    argv.append(mode)
    result = subprocess.run(
        argv, input="\n".join(lines) + "\n", capture_output=True, text=True,
        timeout=300)
    check(result.returncode == 0,
          f"probe {backend or 'default'}/{mode} exits 0 "
          f"(stderr={result.stderr.strip()[:160]!r})")
    out = result.stdout.splitlines()
    check(len(out) == len(lines),
          f"probe {backend or 'default'}/{mode} returns {len(lines)} lines "
          f"(got {len(out)})")
    return out


def run_walk(model: Path, backend: str,
             commands: list[str]) -> list[dict]:
    result = subprocess.run(
        [PROBE, "--model", str(model), "--backend", backend, "walk"],
        input="\n".join(commands) + "\n", capture_output=True, text=True,
        timeout=300)
    check(result.returncode == 0,
          f"walk {backend} exits 0 (stderr={result.stderr.strip()[:160]!r})")
    records = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            check(False, f"walk {backend}: non-JSON output {line[:120]!r}")
    check(len(records) == len(commands),
          f"walk {backend} returns {len(records)} records for "
          f"{len(commands)} commands")
    return records


def check_incremental_parity(model: Path, label: str, positions) -> None:
    """Drive real place/undo sessions and compare with the full reference."""
    print(f"[incremental first layer: {label}, place/undo walk]")
    weights, _ = vf.read_value_weights(model)
    rng = random.Random(17)
    commands: list[str] = []
    segments = []
    for position_index, (stm, indices, text) in enumerate(positions):
        empty = [i for i, c in enumerate(text) if c == "."]
        rng.shuffle(empty)
        picks = empty[:12]
        start = len(commands)
        commands.append(f"load {stm} {text}")
        for cell in picks:
            commands.append(f"place {cell}")
        for _ in picks:
            commands.append("undo")
        commands.append("state")
        segments.append((start, len(commands), stm, indices, len(picks)))

    backend_records: dict[str, list[dict]] = {}
    for backend in ("scalar", "avx2"):
        backend_records[backend] = run_walk(model, backend, commands)

    for backend, records in backend_records.items():
        ok = True
        for start, end, stm, indices, picks in segments:
            seg = records[start:end]
            if len(seg) != end - start:
                ok = False
                continue
            load = seg[0]
            final = seg[-1]
            placed = [r for r in seg[1:1 + picks] if r.get("op") == "place"]
            if (load.get("op") != "load" or final.get("op") != "state"
                    or not placed):
                ok = False
                continue
            if load.get("z") != vf.infer_int(weights, indices):
                ok = False
            if final.get("z") != load.get("z"):
                ok = False
            if final.get("side_to_move") != stm:
                ok = False
            for record in seg:
                if record.get("op") == "error":
                    continue
                if record.get("match") is not True:
                    ok = False
                if record.get("z") != record.get("reference"):
                    ok = False
        check(ok, f"{label}/{backend}: every step matches the full reference, "
                  f"and full undo restores the initial position")

    scalar = backend_records["scalar"]
    avx2 = backend_records["avx2"]
    check(len(scalar) == len(avx2),
          f"{label}: scalar and AVX2 walks produce the same record count")
    compare_ok = True
    for a, b in zip(scalar, avx2):
        if a.get("op") == "error" or b.get("op") == "error":
            continue
        if (a.get("z") != b.get("z")
                or a.get("reference") != b.get("reference")):
            compare_ok = False
    check(compare_ok,
          f"{label}: scalar and AVX2 give identical z at every step")


def check_benchmark(model: Path, positions) -> None:
    """Single-inference timing. Conditions are printed, not assumed."""
    print("[single-inference timing: incremental vs full rebuild]")
    stm, _, text = max(
        positions, key=lambda p: sum(1 for c in p[2] if c != "."))
    iterations = 20000
    line = f"{stm} {text} {iterations} incremental"
    timings: dict[str, float] = {}
    for backend in ("scalar", "avx2"):
        out = run_probe(model, "bench", [line], backend=backend)
        match = re.search(r"ns_per_eval=([0-9.]+)", out[0] if out else "")
        check(match is not None, f"{backend}: bench reports ns_per_eval")
        if match:
            timings[backend] = float(match.group(1))
    full = run_probe(model, "bench",
                     [f"{stm} {text} {iterations} full"], backend="reference")
    match = re.search(r"ns_per_eval=([0-9.]+)", full[0] if full else "")
    check(match is not None, "reference: bench reports ns_per_eval")
    if not match:
        return
    reference_ns = float(match.group(1))
    model_bytes = model.read_bytes()
    print(f"  conditions: {platform.platform()}, CPU={platform.processor() or '?'}, "
          f"cores={os.cpu_count()}, iterations={iterations}, "
          f"stones={sum(1 for c in text if c != '.')}, "
          f"model={model.name} sha256={hashlib.sha256(model_bytes).hexdigest()[:16]}")
    print(f"  full rebuild (reference): {reference_ns:9.2f} ns/eval")
    for backend in ("scalar", "avx2"):
        if backend in timings:
            speedup = reference_ns / max(1.0, timings[backend])
            print(f"  incremental {backend:6s}: {timings[backend]:9.2f} ns/eval "
                  f"({speedup:.2f}x vs full rebuild)")
    if "scalar" in timings and "avx2" in timings:
        speedup = timings["scalar"] / max(1.0, timings["avx2"])
        print(f"  AVX2 vs scalar forward: {speedup:.2f}x")
        check(timings["avx2"] < reference_ns,
              "incremental AVX2 is faster than a full rebuild")
        check(timings["scalar"] < reference_ns,
              "incremental scalar is faster than a full rebuild")
        check(timings["avx2"] <= timings["scalar"] * 1.10,
              f"AVX2 forward is not slower than scalar (ratio {speedup:.2f}x)")
        check(timings["avx2"] < 20000.0,
              "incremental AVX2 inference stays microsecond-scale (< 20 us)")


def reference_scores(weights: dict, positions) -> list[int]:
    return [vf.infer_int(weights, indices) for _, indices, _ in positions]


def check_integer_reference(model: Path, label: str, positions) -> None:
    weights, _ = vf.read_value_weights(model)
    expected_indices = [indices for _, indices, _ in positions]
    cpp_indices = [
        [int(v) for v in line.split()] if line.strip() else []
        for line in run_probe(
            model, "features", [f"{stm} {text}" for stm, _, text in positions])
    ]
    for n, (got, want) in enumerate(zip(cpp_indices, expected_indices)):
        check(got == want,
              f"{label} position {n}: C++ feature indices match features.py")

    reference = reference_scores(weights, positions)
    cpp_sparse = [int(line) for line in run_probe(
        model, "infer",
        [" ".join(str(i) for i in indices) for _, indices, _ in positions])]
    cpp_board = [int(line) for line in run_probe(
        model, "board", [f"{stm} {text}" for stm, _, text in positions])]
    cpp_acc = [int(line) for line in run_probe(
        model, "acc", [f"{stm} {text}" for stm, _, text in positions])]
    check(cpp_sparse == reference,
          f"{label}: scalar sparse forward == infer_int for {len(reference)} positions")
    check(cpp_board == reference,
          f"{label}: full first-layer rebuild == infer_int")
    check(cpp_acc == reference,
          f"{label}: public accumulator rebuild == infer_int")
    check(len(set(reference)) > 1,
          f"{label}: sampled positions produce varied fixed-point scores")

    scale = 1 << int(weights["ea"])
    score_lines = [f"{stm} {text}" for stm, _, text in positions]
    reports = [json.loads(line) for line in run_probe(
        model, "score", score_lines)]
    perspective_ok = True
    for n, (report, (stm, _, _)) in enumerate(zip(reports, positions)):
        expected_black = max(-100000, min(
            100000, vf.round_div(report["z"] * 1000, scale)))
        expected_stm = expected_black if stm == 0 else -expected_black
        if (report["z"] != reference[n]
                or report["black_score"] != expected_black
                or report["stm_score"] != expected_stm
                or report["side_to_move"] != stm):
            perspective_ok = False
            print(f"    position {n}: {report}")
    check(perspective_ok,
          f"{label}: black-absolute z and side-to-move sign conversion")


def synthetic_model(tmp: Path) -> Path:
    rng = random.Random(7)
    ea = 9
    scale = 1 << ea
    weights = {
        "W0": [rng.randint(-8, 8) for _ in range(vf.N_FEATURES * vf.N_ACC)],
        "b0": [rng.randint(-64, 64) for _ in range(vf.N_ACC)],
        "W1": [rng.randint(-16, 16) for _ in range(vf.N_ACC * vf.N_H1)],
        "b1": [rng.randint(-64, 64) for _ in range(vf.N_H1)],
        "W2": [rng.randint(-32, 32) for _ in range(vf.N_H1 * vf.N_H2)],
        "b2": [rng.randint(-32, 32) for _ in range(vf.N_H2)],
        "W3": [rng.randint(-64, 64) for _ in range(vf.N_H2 * vf.N_OUT)],
        "b3": [rng.randint(-128, 128) for _ in range(vf.N_OUT)],
        "ea": ea,
        "act_max_int": int(round(8.0 * scale)),
    }
    metadata = {
        "model_id": "parity-synthetic-model",
        "arch": [vf.N_FEATURES, vf.N_ACC, vf.N_H1, vf.N_H2, vf.N_OUT],
        "rule": "renju",
        "feature_spec": "absolute-color-450",
        "value_contract": "black-absolute-tanh-v1",
        "label_versions": ["parity-v1"],
        "training": {"steps": 0, "created_at": "2026-11-10T00:00:00Z"},
    }
    path = tmp / "synthetic.gmvalue"
    path.write_bytes(vf.pack_value_weights(weights, metadata, ea=ea))
    return path


def check_rejections(tmp: Path, model: Path) -> None:
    print("[format rejection at the probe boundary]")
    blob = model.read_bytes()
    cases = {
        "bad magic": bytes([ord("X")]) + blob[1:],
        "truncated": blob[:-1],
        "checksum": blob[:-1] + bytes([blob[-1] ^ 0xFF]),
        "short header": blob[:32],
    }
    for label, data in cases.items():
        path = tmp / f"reject-{label.replace(' ', '-')}.gmvalue"
        path.write_bytes(data)
        result = subprocess.run(
            [PROBE, "--model", str(path), "board", "0 " + "." * 225],
            input="0 " + "." * 225 + "\n", capture_output=True, text=True,
            timeout=120)
        check(result.returncode != 0,
              f"{label}: probe refuses to load the file")
        check(result.stdout.strip() == "",
              f"{label}: nothing is evaluated from a rejected file")


def check_export_fidelity(positions) -> None:
    """Compare the training model with the deployed C++ integer model."""
    if not REAL_MODEL.is_file():
        print("demo model not present: export fidelity SKIPPED")
        return
    if not has_torch():
        print("torch not installed: export fidelity SKIPPED")
        return
    import torch
    import value_model
    import value_train

    weights, metadata = vf.read_value_weights(REAL_MODEL)
    config = value_model.ValueConfig(
        act_max=weights["act_max_int"] / float(1 << int(weights["ea"])),
        ea=int(weights["ea"]))
    model = None
    used = None
    for path in REAL_CHECKPOINTS:
        if not path.is_file():
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        candidate = value_model.ValueNet(config=config)
        candidate.load_state_dict(blob["model_state"])
        candidate.eval()
        if value_train.model_id(candidate) == metadata["model_id"]:
            model, used = candidate, path
            break
    if model is None:
        check(False, "no local checkpoint matches the exported model_id "
                     "(cannot claim export fidelity)")
        return

    x = torch.tensor([features.dense_vector(indices)
                      for _, indices, _ in positions], dtype=torch.float32)
    with torch.no_grad():
        float_values = [float(v) for v in model(x)]
    cpp_z = [int(line) for line in run_probe(
        REAL_MODEL, "board", [f"{stm} {text}" for stm, _, text in positions])]
    scale = float(1 << int(weights["ea"]))
    cpp_values = [math.tanh(z / scale) for z in cpp_z]
    diffs = [abs(a - b) for a, b in zip(float_values, cpp_values)]
    metrics = {"count": len(diffs),
               "mean_abs": sum(diffs) / len(diffs),
               "p95_abs": p95(diffs),
               "max_abs": max(diffs)}
    print(f"  training model vs C++ deployment ({used.name}, "
          f"{metrics['count']} positions): mean {metrics['mean_abs']:.6f}, "
          f"p95 {metrics['p95_abs']:.6f}, max {metrics['max_abs']:.6f}")
    tolerance = vf.VALUE_TOLERANCE
    check(metrics["mean_abs"] <= tolerance["mean_abs"],
          f"export fidelity mean {metrics['mean_abs']:.6f} <= "
          f"{tolerance['mean_abs']}")
    check(metrics["p95_abs"] <= tolerance["p95_abs"],
          f"export fidelity p95 {metrics['p95_abs']:.6f} <= "
          f"{tolerance['p95_abs']}")
    check(metrics["max_abs"] <= tolerance["max_abs"],
          f"export fidelity max {metrics['max_abs']:.6f} <= "
          f"{tolerance['max_abs']}")


def main() -> int:
    synthetic_positions_ = synthetic_positions()
    real_positions = pool_positions()
    if real_positions:
        print(f"pool-sampled positions: {len(real_positions)}")
        positions = synthetic_positions_ + real_positions
    else:
        print("no sample pool: using synthetic positions only")
        positions = synthetic_positions_

    with tempfile.TemporaryDirectory(prefix="gomoku_value_parity_") as raw:
        tmp = Path(raw)
        synthetic = synthetic_model(tmp)
        print("[synthetic GMVALUE1]")
        check_integer_reference(synthetic, "synthetic", positions)
        check_incremental_parity(synthetic, "synthetic", positions)
        check_rejections(tmp, synthetic)

        if REAL_MODEL.is_file():
            print("[local demo GMVALUE1]")
            check_integer_reference(REAL_MODEL, "demo", positions)
            check_incremental_parity(REAL_MODEL, "demo", positions)
            print("[export fidelity: training model vs C++ deployment]")
            check_export_fidelity(positions)
            check_benchmark(REAL_MODEL, positions)
        else:
            print("demo model not present: real-model parity SKIPPED")
            check_benchmark(synthetic, positions)

    if FAILURES:
        print(f"\n{len(FAILURES)} value parity check(s) failed")
        return 1
    print("value_parity_tests: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
