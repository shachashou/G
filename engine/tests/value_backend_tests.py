#!/usr/bin/env python3
"""Ticket 11: real-engine backend identity, search throughput and paired play.

The probe tests (value_parity_tests) prove position-by-position numeric
equality. This test drives the actual Alpha-Beta search and the match runner:

  * every backend reports its own identity through the executor;
  * fixed-depth search returns the same move, score and node count on all
    three paths (so the optimization cannot change play);
  * search throughput is measured under identical conditions and printed;
  * a paired-opening match between the AVX2 and full-rebuild reference engines
    produces the same move sequence for each opening, i.e. the optimization
    changes cost, not strength.

Usage: python value_backend_tests.py [gomoku_nnue.exe] [tools dir]
"""
from __future__ import annotations

import json
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "trainer"))
sys.path.insert(0, str(ROOT / "tools"))

import value_format as vf  # noqa: E402
from executor import ExecutorClient  # noqa: E402

EXE = sys.argv[1] if len(sys.argv) > 1 else str(
    ROOT / "build" / "engine" / "gomoku_nnue.exe")
TOOLS = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "tools"
REAL_MODEL = ROOT / "weights" / "value-demo" / "model.gmvalue"
SIZE = 15
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        FAILURES.append(message)


def cells_from(blacks, whites) -> list[int]:
    cells = [0] * (SIZE * SIZE)
    for x, y in blacks:
        cells[y * SIZE + x] = 1
    for x, y in whites:
        cells[y * SIZE + x] = 2
    return cells


def synthetic_model(tmp: Path) -> Path:
    rng = random.Random(41)
    ea = 9
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
        "act_max_int": int(round(8.0 * (1 << ea))),
    }
    metadata = {
        "model_id": "ticket11-backend-model",
        "arch": [vf.N_FEATURES, vf.N_ACC, vf.N_H1, vf.N_H2, vf.N_OUT],
        "rule": "renju",
        "feature_spec": "absolute-color-450",
        "value_contract": "black-absolute-tanh-v1",
        "label_versions": ["ticket11-v1"],
        "training": {"steps": 0, "created_at": "2026-11-11T00:00:00Z"},
    }
    path = tmp / "backend-synthetic.gmvalue"
    path.write_bytes(vf.pack_value_weights(weights, metadata, ea=ea))
    return path


def target_position() -> list[int]:
    return cells_from(
        [(7, 7), (9, 7), (10, 6), (6, 8), (8, 9), (11, 8)],
        [(8, 7), (7, 8), (9, 8), (8, 6), (10, 8), (6, 6)])


def measure_search(model: Path, backend: str, depth: int,
                   rounds: int = 3) -> list[dict]:
    position = target_position()
    results = []
    for _ in range(rounds):
        with ExecutorClient(EXE, "--value-model", str(model),
                            "--value-backend", backend) as client:
            about = client.about()
            expected = {
                "avx2": "incremental-avx2",
                "scalar": "incremental-scalar",
                "reference": "full-rebuild-scalar",
            }[backend]
            check((about.get("value_model") or {}).get("backend") == expected,
                  f"{backend}: executor reports {expected}")
            client.new_game("renju")
            state = client.rebuild(position, "black", "renju")
            check(state.get("ok") is True, f"{backend}: position rebuilt")
            started = time.perf_counter()
            response = client.think(depth=depth, vcf_policy="off")
            elapsed = time.perf_counter() - started
        check(response.get("ok") is True, f"{backend}: think succeeds")
        results.append({"response": response, "seconds": elapsed,
                        "nodes": int(response.get("nodes", 0))})
    return results


def test_search_parity_and_throughput(model: Path) -> None:
    depth = 4
    print(f"[search parity and throughput, fixed depth {depth}, VCF off]")
    runs = {backend: measure_search(model, backend, depth)
            for backend in ("reference", "scalar", "avx2")}
    first = {backend: value[0]["response"] for backend, value in runs.items()}
    first_keys = [(backend, tuple(first[backend]["move"]),
                   int(first[backend]["score"]), int(first[backend]["nodes"]))
                  for backend in first]
    check(len({item[1:] for item in first_keys}) == 1,
          "all backends return the same move, score and node count")

    medians = {}
    for backend, values in runs.items():
        times = [v["seconds"] for v in values]
        nodes = values[0]["nodes"]
        medians[backend] = statistics.median(times)
        print(f"  {backend:9s}: nodes={nodes:6d} median={medians[backend]:.3f}s "
              f"nps={nodes / max(1e-9, medians[backend]):.0f}")
    if "reference" in medians and "avx2" in medians:
        speedup = medians["reference"] / max(1e-9, medians["avx2"])
        print(f"  AVX2 search speedup vs full rebuild: {speedup:.2f}x")
        # Correctness gate is above; this is only a loose watchdog against a
        # large whole-engine regression, not a strength claim.
        check(medians["avx2"] < medians["reference"] * 1.20,
              "AVX2 search is not slower than the full-rebuild reference")


def engine_command(model: Path, backend: str) -> str:
    return (f"{EXE} --value-model {model} --value-backend {backend} "
            "--depth 2 --vcf-policy off")


def test_paired_openings(model: Path, tmp: Path) -> None:
    print("[paired openings: AVX2 vs full-rebuild reference]")
    output = tmp / "paired-openings.json"
    games = 4
    command = [
        sys.executable, str(TOOLS / "match_runner.py"),
        "--engine-a", engine_command(model, "avx2"),
        "--engine-b", engine_command(model, "reference"),
        "--games", str(games), "--paired-openings", "--seed", "11",
        "--opening-plies", "2", "--output", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=300)
    check(result.returncode == 0,
          f"paired match completes (stderr={result.stderr.strip()[:160]!r})")
    if result.returncode != 0:
        return
    data = json.loads(output.read_text(encoding="utf-8"))
    check(data.get("complete") is True, "paired match report is complete")
    check(not data.get("invalid_reason"),
          "paired match report has no invalid-configuration reason")
    games_data = data.get("games", [])
    check(len(games_data) == games,
          f"report contains {games} games")
    opening_plies = int(data["config"].get("opening_plies", 2))

    def winner_color(game: dict) -> str | None:
        winner = game.get("winner")
        if winner == "A":
            return game.get("a_color")
        if winner == "B":
            return game.get("b_color")
        return winner

    same_sequences = True
    same_winners = True
    for index in range(0, len(games_data) - 1, 2):
        first, second = games_data[index], games_data[index + 1]
        moves_first = [m["move"] for m in first["moves"][opening_plies:]]
        moves_second = [m["move"] for m in second["moves"][opening_plies:]]
        if moves_first != moves_second:
            same_sequences = False
        if winner_color(first) != winner_color(second):
            same_winners = False
    check(same_sequences,
          "each opening pair plays the identical move sequence with both "
          "backends")
    check(same_winners,
          "each opening pair has the same winning colour with both backends")
    summary = data.get("summary", {})
    print(f"  summary: A {summary.get('a_wins')}W/{summary.get('draws')}D/"
          f"{summary.get('a_losses')}L, win_rate={summary.get('win_rate')}")


def main() -> int:
    if not Path(EXE).is_file():
        print(f"engine not found: {EXE}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="gomoku_value_backend_") as raw:
        tmp = Path(raw)
        if REAL_MODEL.is_file():
            model = REAL_MODEL
            print(f"using local demo model: {model}")
        else:
            model = synthetic_model(tmp)
            print("demo model absent: using a synthetic network")
        test_search_parity_and_throughput(model)
        test_paired_openings(model, tmp)

    if FAILURES:
        print(f"\n{len(FAILURES)} backend check(s) failed")
        return 1
    print("value_backend_tests: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
