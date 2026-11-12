#!/usr/bin/env python3
"""Ticket 07 black-box tests: the engine really plays with a GMVALUE1 model.

Covers the whole user-visible contract through the two process front ends:

  * executor: loaded-model identity, full legal game under a time budget,
    replay from the move record, side-to-move sign conversion, terminal mate
    and proven-VCF scores that the network cannot override, time control
  * Gomocup protocol: a complete headless self-play game with the model
  * strict loading: corrupted / truncated / legacy / mutually-exclusive model
    arguments abort the process instead of silently playing handcrafted

Usage: python value_engine_blackbox.py [path to gomoku_nnue.exe]
"""

import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "trainer"))
sys.path.insert(0, str(ROOT / "tools"))

import value_format as vf  # noqa: E402
from executor import ExecutorClient, ExecutorError  # noqa: E402

EXE = sys.argv[1] if len(sys.argv) > 1 else str(
    ROOT / "build" / "engine" / "gomoku_nnue.exe")
REAL_MODEL = ROOT / "weights" / "value-demo" / "model.gmvalue"
SIZE = 15
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))
MATE_BOUND = 900000
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        FAILURES.append(message)


# ---------------------------------------------------------------- helpers

def constant_weights(b3: int, ea: int = 9) -> dict:
    """Board-independent black-absolute fixed-point z == b3."""
    return {
        "W0": [0] * (vf.N_FEATURES * vf.N_ACC),
        "b0": [0] * vf.N_ACC,
        "W1": [0] * (vf.N_ACC * vf.N_H1),
        "b1": [0] * vf.N_H1,
        "W2": [0] * (vf.N_H1 * vf.N_H2),
        "b2": [0] * vf.N_H2,
        "W3": [0] * (vf.N_H2 * vf.N_OUT),
        "b3": [b3],
        "ea": ea,
        "act_max_int": int(round(8.0 * (1 << ea))),
    }


def write_constant(path: Path, b3: int, model_id: str) -> Path:
    ea = 9
    metadata = {
        "model_id": model_id,
        "arch": [vf.N_FEATURES, vf.N_ACC, vf.N_H1, vf.N_H2, vf.N_OUT],
        "rule": "renju",
        "feature_spec": "absolute-color-450",
        "value_contract": "black-absolute-tanh-v1",
        "label_versions": ["blackbox-v1"],
        "training": {"steps": 3, "created_at": "2026-11-10T00:00:00Z"},
    }
    path.write_bytes(vf.pack_value_weights(
        constant_weights(b3, ea), metadata, ea=ea))
    return path


def cells_from(blacks, whites) -> list[int]:
    cells = [0] * (SIZE * SIZE)
    for x, y in blacks:
        cells[y * SIZE + x] = 1
    for x, y in whites:
        cells[y * SIZE + x] = 2
    return cells


def moves_from_state(state: dict) -> list[list[int]]:
    return [[int(p[0]), int(p[1])] for p in state.get("history", [])]


def wins(board, x: int, y: int) -> bool:
    colour = board[y][x]
    if colour == 0:
        return False
    for dx, dy in DIRS:
        count = 1
        for sign in (1, -1):
            nx, ny = x + dx * sign, y + dy * sign
            while 0 <= nx < SIZE and 0 <= ny < SIZE and board[ny][nx] == colour:
                count += 1
                nx += dx * sign
                ny += dy * sign
        if count >= 5:
            return True
    return False


def play_constant_score(client: ExecutorClient, side: str) -> int:
    response = client.think(depth=1, vcf_policy="off")
    check(response.get("ok") is True,
          f"{side}-to-move think succeeds")
    return int(response.get("score", 0))


# ---------------------------------------------------------------- tests

def test_identity_and_full_game(model: Path, metadata: dict) -> None:
    print("[executor: identity, full game, replay, time control]")
    with ExecutorClient(EXE, "--value-model", str(model)) as client:
        about = client.about()
        check(about.get("value_model_loaded") is True,
              "about reports the value model as loaded")
        check(about.get("eval_backend") == "value-gmvalue1-incremental-avx2",
              "about names the incremental AVX2 GMVALUE1 deployment backend")
        check(about.get("eval") == "nnue",
              "about reports NNUE evaluation, not handcrafted")
        value = about.get("value_model") or {}
        check(value.get("model_id") == metadata["model_id"],
              "about shows the model_id from the file metadata")
        check(value.get("arch") == "450-1024-64-32-1",
              "about shows the 450-1024-64-32-1 architecture")
        check(value.get("value_contract") == "black-absolute-tanh-v1",
              "about shows the frozen value contract")
        check(value.get("rule") == "renju"
              and value.get("feature_spec") == "absolute-color-450",
              "about shows the Renju rule and absolute-colour features")
        check(value.get("backend") == "incremental-avx2"
              and value.get("accumulator") == "incremental"
              and value.get("compute") == "avx2",
              "about reports the incremental first layer and AVX2 compute")
        check(value.get("promoted") is False
              and value.get("status") == "candidate-unpromoted",
              "a loaded model is a playable candidate, not an auto-promotion")
        check("GMVALUE1" in str(about.get("eval_description", "")),
              "eval description identifies the deployed GMVALUE1 network")

        state = client.new_game("renju")
        started = time.monotonic()
        think_times: list[float] = []
        while state["result"] == "ongoing" and len(state.get("history", [])) < 220:
            legal = {tuple(p) for p in client.legal_moves().get("moves", [])}
            check(len(legal) > 0, "legal move set is available during the game")
            move_started = time.monotonic()
            response = client.think(movetime_ms=60, vcf_policy="root",
                                    vcf_nodes=2000)
            think_times.append(time.monotonic() - move_started)
            if not response.get("ok"):
                check(False, f"think failed: {response}")
                break
            move = tuple(int(v) for v in response["move"])
            check(move in legal, f"engine move {move} is in the legal set")
            if move not in legal:
                break
            outcome = client.play(move[0], move[1])
            check(outcome.get("ok") is True,
                  f"engine move {move} is accepted by the executor")
            state = outcome
        elapsed = time.monotonic() - started
        check(state["result"] in ("black_win", "white_win", "draw"),
              f"game reached a natural result ({state['result']}, "
              f"{len(state.get('history', []))} plies)")
        check(elapsed < 120,
              f"time-controlled full game finished ({elapsed:.1f}s)")
        check(all(t < 5.0 for t in think_times) and think_times,
              "no move exceeded the time budget by more than a safety margin")
        check(state.get("move_count", 0) == len(state.get("history", [])),
              "state move count matches the move record")

        history = moves_from_state(state)
        replayed = client.replay(history, "renju")
        check(replayed.get("ok") is True,
              "replaying the move record through the executor succeeds")
        check(replayed.get("result") == state["result"],
              f"replay reproduces the terminal result ({state['result']})")
        check(replayed.get("cells") == state["cells"],
              "replay reproduces the final board")


def test_side_to_move_perspective(tmp: Path) -> None:
    print("[executor: value sign is converted to the side to move]")
    positive = write_constant(tmp / "positive.gmvalue", 500, "positive-const")
    negative = write_constant(tmp / "negative.gmvalue", -500, "negative-const")
    scale = 1 << 9
    expected = round(500 * 1000 / scale)  # 977

    with ExecutorClient(EXE, "--value-model", str(positive)) as client:
        client.new_game("renju")
        black_score = play_constant_score(client, "black")
        check(black_score == expected,
              f"black to move receives +{expected} (got {black_score})")
        client.play(7, 7)
        white_score = play_constant_score(client, "white")
        check(white_score == -expected,
              f"white to move receives -{expected} (got {white_score})")

    with ExecutorClient(EXE, "--value-model", str(negative)) as client:
        client.new_game("renju")
        black_score = play_constant_score(client, "black")
        check(black_score == -expected,
              f"negative black-absolute z gives -{expected} to black "
              f"(got {black_score})")


def test_backend_identity_and_parity(model: Path) -> None:
    print("[executor: backend identity and bit-identical move selection]")
    cells = cells_from(
        [(7, 7), (9, 7), (7, 9), (9, 9), (8, 8)],
        [(8, 7), (7, 8), (9, 8), (8, 9), (6, 6)])
    outcomes: dict[str, tuple] = {}
    for backend, expected in (("avx2", "incremental-avx2"),
                              ("scalar", "incremental-scalar"),
                              ("reference", "full-rebuild-scalar")):
        with ExecutorClient(EXE, "--value-model", str(model),
                            "--value-backend", backend) as client:
            about = client.about()
            value = about.get("value_model") or {}
            check(value.get("backend") == expected,
                  f"{backend}: about reports {expected}")
            client.new_game("renju")
            state = client.rebuild(cells, "black", "renju")
            check(state.get("ok") is True, f"{backend}: test position rebuilt")
            response = client.think(depth=2, vcf_policy="off")
            check(response.get("ok") is True,
                  f"{backend}: fixed-depth think succeeds")
            check(response.get("value_model_backend") == expected,
                  f"{backend}: think echoes the active backend")
            outcomes[backend] = (tuple(response["move"]),
                                 int(response["score"]),
                                 int(response["nodes"]))
    check(outcomes["avx2"] == outcomes["scalar"]
          and outcomes["avx2"] == outcomes["reference"],
          "avx2 / scalar / reference choose the same move, score and node count")
    print(f"  shared selection: move={outcomes['avx2'][0]}, "
          f"score={outcomes['avx2'][1]}, nodes={outcomes['avx2'][2]}")


def test_terminal_and_vcf(tmp: Path) -> None:
    print("[executor: terminal and proven-VCF scores dominate the network]")
    model = write_constant(tmp / "negative-fatal.gmvalue", -500,
                           "negative-terminal")
    with ExecutorClient(EXE, "--value-model", str(model)) as client:
        # Black has an open four; the network says black is lost. The visible
        # five must still win immediately (mate score, not a network score).
        four = cells_from(
            [(7, 7), (8, 7), (9, 7), (10, 7)],
            [(6, 7), (6, 8), (5, 9)])
        state = client.rebuild(four, "black", "renju")
        check(state.get("ok") is True, "open-four position rebuilt")
        response = client.think(depth=1, vcf_policy="off")
        check(response.get("ok") is True and response.get("from_vcf") is False,
              "mate search runs without VCF")
        check(int(response.get("score", 0)) > MATE_BOUND,
              f"terminal mate score {response.get('score')} dominates the "
              "negative network value")
        move = response["move"]
        outcome = client.play(move[0], move[1])
        check(outcome.get("result") == "black_win",
              f"engine's move {tuple(move)} completes five and wins")

        # A proven upstream VCF must also keep its own mate semantics.
        blacks = [(7, 7), (9, 7), (10, 7), (8, 6), (11, 6), (8, 5)]
        whites = [(7, 8), (11, 7), (9, 6), (10, 5), (8, 4), (11, 4)]
        state = client.rebuild(cells_from(blacks, whites), "black", "renju")
        check(state.get("ok") is True, "VCF test position rebuilt")
        response = client.think(depth=1, vcf_policy="root", vcf_nodes=50000)
        check(response.get("from_vcf") is True,
              "proven VCF is used even though the network is negative")
        check(tuple(response["move"]) == (8, 7),
              f"proven VCF first move is (8,7) (got {response['move']})")
        check(int(response.get("score", 0)) > MATE_BOUND,
              f"proven VCF score {response.get('score')} is a mate score")


def test_strict_rejection(tmp: Path, good: Path) -> None:
    print("[executor: malformed or incompatible models fail clearly]")
    blob = good.read_bytes()
    cases = {
        "bad magic": bytes([ord("X")]) + blob[1:],
        "truncated": blob[:-1],
        "checksum": blob[:-1] + bytes([blob[-1] ^ 0xFF]),
        "legacy GMNNUE01": b"GMNNUE01" + blob[8:],
    }
    for label, data in cases.items():
        path = tmp / f"reject-{label.replace(' ', '-')}.gmvalue"
        path.write_bytes(data)
        result = subprocess.run(
            [EXE, "--executor", "--value-model", str(path)],
            input='{"cmd":"about"}\n', capture_output=True, text=True,
            timeout=120)
        check(result.returncode != 0,
              f"{label}: process exits non-zero instead of falling back")
        check(result.stdout.strip() == "",
              f"{label}: no response claims an evaluation")
        check("value model failed to load" in result.stderr,
              f"{label}: stderr explains the load failure "
              f"({result.stderr.strip()[:100]!r})")

    missing = tmp / "does-not-exist.gmvalue"
    result = subprocess.run(
        [EXE, "--executor", "--value-model", str(missing)],
        input='{"cmd":"about"}\n', capture_output=True, text=True, timeout=120)
    check(result.returncode != 0 and "cannot open" in result.stderr,
          "missing model fails clearly")
    check("handcrafted" not in result.stdout,
          "a failed model never continues as handcrafted while claiming NNUE")

    legacy = write_constant(tmp / "legacy-probe.gmvalue", 1, "legacy-probe")
    result = subprocess.run(
        [EXE, "--executor", "--value-model", str(good), "--nnue", str(legacy)],
        input='{"cmd":"about"}\n', capture_output=True, text=True, timeout=120)
    check(result.returncode == 2 and "mutually exclusive" in result.stderr,
          "--value-model and --nnue together are rejected")


class Engine:
    """Minimal Gomocup-protocol process driver."""

    def __init__(self, args):
        self.proc = subprocess.Popen(
            [EXE, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)

    def send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def recv(self):
        line = self.proc.stdout.readline()
        return line.strip() if line else None

    def close(self) -> int:
        try:
            self.send("END")
            self.proc.stdin.close()
            return self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()
            return self.proc.returncode or 1


def parse_move(text):
    match = re.fullmatch(r"(\d+),(\d+)", text or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def test_protocol_full_game(model: Path, metadata: dict) -> None:
    print("[Gomocup protocol: full headless self-play game]")
    args = ["--value-model", str(model), "--depth", "1", "--vcf-policy", "off"]
    black, white = Engine(args), Engine(args)
    for engine in (black, white):
        engine.send("START 15")
        check(engine.recv() == "OK", "START 15 -> OK with the value model")
    black.send("ABOUT")
    about = black.recv() or ""
    check('eval="nnue"' in about
          and 'eval_engine="gmvalue1-incremental-avx2"' in about,
          "ABOUT identifies the incremental AVX2 GMVALUE1 deployment")
    check(f'value_model_id="{metadata["model_id"]}"' in about,
          "ABOUT reports the loaded model id")
    check('value_model_promoted="false"' in about,
          "ABOUT marks the model as an unpromoted candidate")

    board = [[0] * SIZE for _ in range(SIZE)]
    moves: list[list[int]] = []
    mover = black
    black.send("BEGIN")
    move = parse_move(black.recv())
    while move is not None and len(moves) < SIZE * SIZE:
        x, y = move
        if not (0 <= x < SIZE and 0 <= y < SIZE) or board[y][x] != 0:
            check(False, f"protocol move {move} is legal")
            break
        board[y][x] = 1 if mover is black else 2
        moves.append([x, y])
        if wins(board, x, y):
            break
        other = white if mover is black else black
        other.send(f"TURN {x},{y}")
        move = parse_move(other.recv())
        mover = other
    check(len(moves) > 1, f"protocol game produced real moves ({len(moves)})")

    with ExecutorClient(EXE) as referee:
        draft = referee.replay(moves, "renju")
    check(draft.get("ok") is True,
          "protocol move record replays through the authoritative executor")
    if draft.get("ok"):
        check(draft.get("result") in ("black_win", "white_win", "draw", "ongoing"),
              "replay reports a defined Renju outcome")
    check(black.close() == 0 and white.close() == 0,
          "both protocol engines exit cleanly")


def main() -> int:
    if not Path(EXE).is_file():
        print(f"engine not found: {EXE}", file=sys.stderr)
        return 2
    random.seed(7)
    with tempfile.TemporaryDirectory(prefix="gomoku_value_engine_") as raw:
        tmp = Path(raw)
        if REAL_MODEL.is_file():
            model = REAL_MODEL
            metadata = vf.read_value_weights(model)[1]
            print(f"using local demo model: {model}")
        else:
            model = write_constant(tmp / "fallback.gmvalue", 100,
                                   "blackbox-fallback")
            metadata = vf.read_value_weights(model)[1]
            print("demo model absent: using a synthetic fallback model")

        test_identity_and_full_game(model, metadata)
        test_backend_identity_and_parity(model)
        test_side_to_move_perspective(tmp)
        test_terminal_and_vcf(tmp)
        test_strict_rejection(tmp, model)
        test_protocol_full_game(model, metadata)

    if FAILURES:
        print(f"\n{len(FAILURES)} value engine check(s) failed")
        return 1
    print("value_engine_blackbox: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
