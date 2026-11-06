#!/usr/bin/env python3
"""Self-tests for the ticket 06 match_runner.

Usage: python match_runner_tests.py [gomoku_nnue.exe] [tools/match_runner.py]

Covers the referee helpers, a real 04-style engine vs 02 random baseline
(100% win gate with colour alternation), JSON replayability, and the three
anomaly paths (illegal move / timeout / crash) via a tiny fake engine.
"""
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import match_runner as mr  # noqa: E402

EXE = Path(sys.argv[1]).resolve()
RUNNER = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else TOOLS / "match_runner.py"
PYTHON = sys.executable

failures = 0

FAKE_ENGINE = '''import sys
mode = sys.argv[1] if len(sys.argv) > 1 else "quiet"
for line in sys.stdin:
    line = line.strip()
    if line == "END":
        break
    if line.startswith("START"):
        print("OK", flush=True)
        if mode == "crash":
            break
    elif line == "DONE":
        if mode == "illegal":
            print("99,99", flush=True)
        elif mode == "legal":
            print("7,7", flush=True)
'''


def check(cond, msg):
    global failures
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures += 1


def test_helpers():
    print("[referee helpers]")
    argv = mr.split_command(r'"E:\G\build\engine\Release\gomoku_nnue.exe" --depth 2')
    check(argv == [r"E:\G\build\engine\Release\gomoku_nnue.exe", "--depth", "2"],
          "Windows command string with quoted path and args")

    board = [0] * (mr.SIZE * mr.SIZE)
    for x in (3, 4, 5, 6, 7):
        board[7 * mr.SIZE + x] = 1
    check(mr.move_state(board, 7, 7, 1, 2) == "win", "Renju black exactly five wins")

    board = [0] * (mr.SIZE * mr.SIZE)
    for x in (3, 4, 5, 6, 7, 8):
        board[7 * mr.SIZE + x] = 1
    check(mr.move_state(board, 7, 7, 1, 2) == "foul-overline",
          "Renju black overline is an illegal move")
    check(mr.move_state(board, 7, 7, 1, 1) == "win",
          "freestyle black overline wins")

    board = [0] * (mr.SIZE * mr.SIZE)
    for x in (3, 4, 5, 6, 7, 8):
        board[7 * mr.SIZE + x] = 2
    check(mr.move_state(board, 7, 7, 2, 2) == "win", "Renju white overline wins")

    first = mr.random_opening(random.Random(42), 4, 7)
    second = mr.random_opening(random.Random(42), 4, 7)
    check(first == second and len(first) == 4 and len(set(first)) == 4,
          "random opening is reproducible and has distinct cells")
    check(all(4 <= x <= 10 and 4 <= y <= 10 for x, y in first),
          "random opening stays inside the centre 7x7")
    check(mr.random_opening(random.Random(1), 0, 7) == [], "zero opening plies")


def test_strength_gate(tmp):
    print("[04-style engine vs 02 random baseline]")
    output = tmp / "strength.json"
    cmd = [
        PYTHON, str(RUNNER),
        "--engine-a", f'"{EXE}" --depth 2',
        "--engine-b", f'"{EXE}" --random',
        "--games", "4", "--rule", "2", "--opening-plies", "2",
        "--seed", "2026", "--move-timeout", "10", "--max-plies", "80",
        "--output", str(output), "--min-win-rate", "1.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    check(result.returncode == 0,
          f"100% win-rate gate passes (exit {result.returncode}) "
          f"{result.stdout[-500:]} {result.stderr[-200:]}")
    check(output.exists(), "JSON result file written")
    if not output.exists():
        return
    data = json.loads(output.read_text(encoding="utf-8"))
    summary = data["summary"]
    check(summary["games"] == 4, "4 games played")
    check(summary["a_wins"] == 4 and summary["a_losses"] == 0 and summary["draws"] == 0,
          f"engine A won 4/4 ({summary['a_wins']}W/{summary['draws']}D/{summary['a_losses']}L)")
    check(summary["win_rate"] == 1.0, "win_rate = 100%")
    check(summary["a_as_black"] == 2, "colours alternated (A black in 2 games)")
    check(summary["anomaly_counts"] == {}, "no anomalies recorded")

    games = data["games"]
    check([g["a_color"] for g in games] == ["black", "white", "black", "white"],
          "A plays black / white / black / white")
    for game in games:
        tag = f"game {game['game']}"
        check(game["result"] == "A-win" and game["reason"] == "five-in-row",
              f"{tag} ended by a five-in-row win for A")
        moves = game["moves"]
        check(len(moves) >= 5 and game["engine_plies"] >= 3,
              f"{tag} has a real move sequence ({len(moves)} plies)")
        check(all(0 <= m["x"] < mr.SIZE and 0 <= m["y"] < mr.SIZE for m in moves),
              f"{tag} all moves in bounds")
        check(len({(m["x"], m["y"]) for m in moves}) == len(moves),
              f"{tag} no move replayed on an occupied point")
        check(len(moves[:2]) == 2 and all(m["opening"] and m["engine"] is None
                                         for m in moves[:2]),
              f"{tag} two recorded random opening stones")
        expected = game["a_color"]
        ok = True
        for move in moves[2:]:
            if move["opening"] or move["engine"] is None:
                ok = False
                break
            color_engine = "A" if expected == move["color"] else "B"
            if move["engine"] != color_engine:
                ok = False
                break
        check(ok, f"{tag} every engine move attributed to the correct side")
        check(game["anomalies"] == [], f"{tag} no anomalies")
    check(data["config"]["rule"] == 2 and data["config"]["seed"] == 2026,
          "result file records the run configuration")


def test_freestyle_mode(tmp):
    print("[freestyle rule mode]")
    output = tmp / "freestyle.json"
    cmd = [
        PYTHON, str(RUNNER),
        "--engine-a", f'"{EXE}" --depth 1',
        "--engine-b", f'"{EXE}" --depth 1',
        "--games", "2", "--rule", "1", "--opening-plies", "0",
        "--seed", "11", "--move-timeout", "10", "--max-plies", "60",
        "--output", str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    check(result.returncode == 0,
          f"freestyle matches run cleanly (exit {result.returncode}) {result.stderr[-200:]}")
    if not output.exists():
        check(False, "freestyle result file written")
        return
    data = json.loads(output.read_text(encoding="utf-8"))
    check(data["config"]["rule"] == 1, "INFO rule 1 recorded in config")
    check(all(g["result"] in ("A-win", "B-win", "draw") for g in data["games"]),
          "both freestyle games reached a terminal result")
    check(data["summary"]["anomaly_counts"] == {}, "freestyle games had no anomalies")


def test_verify_backend_flag():
    print("[backend verification flag]")
    base = ["--games", "2", "--opening-plies", "0"]
    check(not mr.parse_args(base).verify_backend,
          "backend verification stays off by default so fake engines still work")
    check(mr.parse_args([*base, "--verify-backend"]).verify_backend,
          "--verify-backend reaches MatchConfig for screening matches")
    formal = mr.parse_args(["--games", "200", "--m5-gate", "--seed", "20261109",
                            "--openings", "formal_openings.json",
                            "--engine-a", "gomoku_nnue.exe --depth 2 --nnue candidate.nnue",
                            "--engine-b", "gomoku_nnue.exe --depth 2"])
    check(formal.verify_backend,
          "formal M5 gate always verifies the backend")


def test_gate_failure(tmp):
    print("[win-rate gate failure]")
    fake = tmp / "fake_engine.py"
    fake.write_text(FAKE_ENGINE, encoding="utf-8")
    output = tmp / "gate.json"
    cmd = [
        PYTHON, str(RUNNER),
        "--engine-a", f'"{PYTHON}" "{fake}" illegal',
        "--engine-b", f'"{PYTHON}" "{fake}" quiet',
        "--games", "1", "--opening-plies", "0", "--move-timeout", "5",
        "--output", str(output), "--min-win-rate", "0.65",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    check(result.returncode == 2 and "FAIL" in result.stdout,
          f"gate below threshold exits 2 (exit {result.returncode})")


def test_anomalies(tmp):
    print("[illegal move / timeout / crash]")
    fake = tmp / "fake_engine.py"
    fake.write_text(FAKE_ENGINE, encoding="utf-8")
    for mode, expected in (("illegal", "illegal-move"),
                           ("timeout", "timeout"),
                           ("crash", "crash")):
        output = tmp / f"anomaly_{mode}.json"
        cmd = [
            PYTHON, str(RUNNER),
            "--engine-a", f'"{PYTHON}" "{fake}" {mode}',
            "--engine-b", f'"{PYTHON}" "{fake}" quiet',
            "--games", "1", "--opening-plies", "0", "--max-plies", "5",
            "--move-timeout", "0.5", "--output", str(output),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        check(result.returncode == 0,
              f"{mode}: runner exits cleanly ({result.returncode}) {result.stderr[-200:]}")
        if not output.exists():
            check(False, f"{mode}: result file written")
            continue
        data = json.loads(output.read_text(encoding="utf-8"))
        game = data["games"][0]
        check(game["result"] == "B-win",
              f"{mode}: offending engine A is charged with a loss")
        kinds = [a["kind"] for a in game["anomalies"]]
        check(kinds == [expected], f"{mode}: anomaly recorded as {expected}, got {kinds}")
        check(data["summary"]["a_losses"] == 1 and data["summary"]["win_rate"] == 0.0,
              f"{mode}: loss counted in the summary")


def main():
    test_helpers()
    test_verify_backend_flag()
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        test_strength_gate(tmp)
        test_freestyle_mode(tmp)
        test_gate_failure(tmp)
        test_anomalies(tmp)
    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
