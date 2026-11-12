#!/usr/bin/env python3
"""Ticket 09 black-box tests: the engine plays a full game with NNUE on.

Also checks that a malformed/missing weight file produces a clear stderr
message, never pollutes the Gomocup stdout stream, and falls back to the
handcrafted evaluator instead of crashing.

Usage: python nnue_blackbox.py [path to gomoku_nnue.exe]
"""

import random
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "trainer"))

import weights_format as wf  # noqa: E402

EXE = sys.argv[1] if len(sys.argv) > 1 else "gomoku_nnue.exe"
SIZE = 15
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))
WATCHDOG_SECONDS = 150

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        FAILURES.append(message)


class Engine:
    def __init__(self, args=(), capture_stderr: bool = False):
        argv = [EXE, *args]
        self.p = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1)
        self.stderr = ""
        self.watchdog = threading.Timer(WATCHDOG_SECONDS, self.p.kill)
        self.watchdog.daemon = True
        self.watchdog.start()

    def send(self, line: str) -> None:
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def recv(self) -> str | None:
        line = self.p.stdout.readline()
        return line.strip() if line else None

    def close(self) -> int:
        try:
            self.send("END")
            self.p.stdin.close()
            code = self.p.wait(timeout=10)
        except Exception:
            self.p.kill()
            code = self.p.returncode
        self.watchdog.cancel()
        if self.p.stderr is not None:
            try:
                self.stderr = self.p.stderr.read()
            except Exception:
                pass
        return code


def parse_move(s: str | None):
    m = re.fullmatch(r"(\d+),(\d+)", s or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def wins(board, x, y) -> bool:
    c = board[y][x]
    if c == 0:
        return False
    for dx, dy in DIRS:
        n = 1
        for s in (1, -1):
            nx, ny = x + dx * s, y + dy * s
            while 0 <= nx < SIZE and 0 <= ny < SIZE and board[ny][nx] == c:
                n += 1
                nx += dx * s
                ny += dy * s
        if n >= 5:
            return True
    return False


def test_nnue_full_game(weights_path: Path) -> None:
    print("[NNUE full self-play game]")
    args = ["--nnue", str(weights_path), "--depth", "1"]
    black, white = Engine(args), Engine(args)
    for e in (black, white):
        e.send("START 15")
        check(e.recv() == "OK", "START 15 -> OK with NNUE enabled")
    black.send("ABOUT")
    about = black.recv() or ""
    check('eval="nnue"' in about and 'nnue_backend="avx2"' in about,
          "ABOUT reports NNUE with the default AVX2 backend")
    white.send("ABOUT")
    check('nnue_backend="avx2"' in (white.recv() or ""),
          "both engines use the AVX2 backend")

    black.send("BEGIN")
    board = [[0] * SIZE for _ in range(SIZE)]
    moves = 0
    outcome = None
    mover = black
    mv = parse_move(black.recv())
    while mv is not None:
        legal = (0 <= mv[0] < SIZE and 0 <= mv[1] < SIZE and
                 board[mv[1]][mv[0]] == 0)
        if not legal:
            outcome = "invalid-move"
            check(False, f"move #{moves + 1} {mv} is legal")
            break
        board[mv[1]][mv[0]] = 1 if mover is black else 2
        moves += 1
        if wins(board, mv[0], mv[1]):
            outcome = "black-win" if mover is black else "white-win"
            break
        if moves >= SIZE * SIZE:
            outcome = "draw"
            break
        other = white if mover is black else black
        other.send(f"TURN {mv[0]},{mv[1]}")
        nxt = parse_move(other.recv())
        if nxt is None:
            outcome = "bad-reply"
            break
        mv, mover = nxt, other

    check(outcome in ("black-win", "white-win", "draw"),
          f"game reached a natural end ({outcome}, {moves} moves)")
    check(black.close() == 0 and white.close() == 0,
          "both NNUE engines exit cleanly")


def test_backend_selection(weights_path: Path, bad_path: Path) -> None:
    print("[backend selection and fallback]")
    plain = Engine(["--depth", "1"])
    plain.send("ABOUT")
    about = plain.recv() or ""
    check('eval="handcrafted"' in about and 'nnue_backend="none"' in about,
          "default backend is handcrafted (three-way switch)")
    check(plain.close() == 0, "default engine exits cleanly")

    for backend in ("scalar", "avx2"):
        e = Engine(["--nnue", str(weights_path), "--nnue-backend", backend,
                    "--depth", "1"])
        e.send("ABOUT")
        check(f'nnue_backend="{backend}"' in (e.recv() or ""),
              f"--nnue-backend {backend} is selectable")
        e.send("TURN 7,7")
        check(parse_move(e.recv()) is not None,
              f"--nnue-backend {backend} plays a legal move")
        check(e.close() == 0, f"--nnue-backend {backend} exits cleanly")

    unknown = Engine(["--nnue", str(weights_path), "--nnue-backend", "sse2",
                      "--depth", "1"], capture_stderr=True)
    unknown.send("ABOUT")
    check('nnue_backend="avx2"' in (unknown.recv() or ""),
          "unknown backend falls back to AVX2")
    check(unknown.close() == 0, "unknown backend exits cleanly")
    check("unknown backend" in unknown.stderr,
          "unknown backend is reported on stderr")

    for label, path, keyword in (
        ("bad magic", bad_path, "bad magic"),
        ("missing file", bad_path.parent / "does_not_exist.nnue", "cannot open"),
    ):
        e = Engine(["--nnue", str(path), "--depth", "1"], capture_stderr=True)
        e.send("START 15")
        check(e.recv() == "OK",
              f"{label}: stdout stays a clean Gomocup stream")
        e.send("ABOUT")
        check('eval="handcrafted"' in (e.recv() or ""),
              f"{label}: falls back to handcrafted evaluation")
        e.send("TURN 7,7")
        check(parse_move(e.recv()) is not None,
              f"{label}: engine still replies with a legal move")
        check(e.close() == 0, f"{label}: engine exits cleanly")
        check(keyword in e.stderr and "NNUE" in e.stderr,
              f"{label}: stderr explains the failure ({e.stderr.strip()[:120]!r})")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="gomoku_nnue_blackbox_") as raw_tmp:
        tmp = Path(raw_tmp)
        weights = {
            "W0": [0] * wf.W0_LEN, "b0": [0] * wf.B0_LEN,
            "W1": [0] * wf.W1_LEN, "b1": [0] * wf.B1_LEN,
            "W2": [0] * wf.W2_LEN, "b2": [0] * wf.B2_LEN,
        }
        rng = random.Random(17)
        for i in range(0, wf.W0_LEN, 1013):
            weights["W0"][i] = rng.randint(-6, 6)
        weights["b0"] = [rng.randint(16, 48) for _ in range(wf.B0_LEN)]
        weights["W1"] = [rng.randint(-4, 4) for _ in range(wf.W1_LEN)]
        weights["b1"] = [rng.randint(16, 48) for _ in range(wf.B1_LEN)]
        weights["W2"] = [rng.randint(-20, 20) for _ in range(wf.W2_LEN)]
        blob = bytearray(wf.pack_weights(weights))
        good = tmp / "probe.nnue"
        good.write_bytes(blob)
        bad = bytearray(blob)
        bad[0] = ord("X")
        bad_path = tmp / "bad_magic.nnue"
        bad_path.write_bytes(bad)

        test_nnue_full_game(good)
        test_backend_selection(good, bad_path)

    if FAILURES:
        print(f"\n{len(FAILURES)} NNUE black-box check(s) failed")
        return 1
    print("nnue_blackbox: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
