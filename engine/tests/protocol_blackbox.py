#!/usr/bin/env python3
"""Black-box protocol tests for the GomokuNNUE engine (ticket 02).

Drives the exe via stdin/stdout, asserting Gomocup protocol behaviour.
Usage: python protocol_blackbox.py [path to gomoku_nnue.exe]
"""
import re
import subprocess
import sys
import threading

EXE = sys.argv[1] if len(sys.argv) > 1 else "gomoku_nnue.exe"
SIZE = 15
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))
WATCHDOG_SECONDS = 10

failures = 0


def check(cond, msg):
    global failures
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures += 1


class Engine:
    def __init__(self):
        self.p = subprocess.Popen(
            [EXE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1)
        # Watchdog so a protocol bug can never hang the suite.
        self.watchdog = threading.Timer(WATCHDOG_SECONDS, self.p.kill)
        self.watchdog.daemon = True
        self.watchdog.start()

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def recv(self):
        line = self.p.stdout.readline()
        return line.strip() if line else None

    def close(self):
        code = 0
        try:
            self.send("END")
            self.p.stdin.close()
            code = self.p.wait(timeout=10)
        except Exception:
            self.p.kill()
            code = self.p.returncode
        self.watchdog.cancel()
        return code


def parse_move(s):
    m = re.fullmatch(r"(\d+),(\d+)", s or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def wins(board, x, y):
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


def test_version_flag():
    print("[--version]")
    r = subprocess.run([EXE, "--version"], capture_output=True, text=True, timeout=30)
    check(r.returncode == 0, "--version exits 0")
    check("GomokuNNUE" in r.stdout, "--version prints engine banner")


def test_basic_commands():
    print("[basic commands]")
    e = Engine()
    e.send("START 15")
    check(e.recv() == "OK", "START 15 -> OK")
    e.send("START 20")
    check((e.recv() or "").startswith("ERROR"), "START 20 -> ERROR (only 15x15)")
    e.send("ABOUT")
    check("name" in (e.recv() or ""), "ABOUT -> engine info line")
    e.send("FROBNICATE")
    check(e.recv() == "UNKNOWN", "unknown command -> UNKNOWN")
    e.send("RESTART")
    check(e.recv() == "OK", "RESTART -> OK")
    e.send("INFO timeout_turn 5000")
    e.send("INFO rule 1")
    e.send("START 15")
    check(e.recv() == "OK", "INFO produces no reply; next command still works")
    e.send("TURN 7,7")
    mv = parse_move(e.recv())
    check(mv is not None, "TURN -> reply in x,y format")
    check(mv is not None and 0 <= mv[0] < SIZE and 0 <= mv[1] < SIZE, "TURN move in bounds")
    check(mv != (7, 7), "engine does not play on occupied cell")
    e.send("TURN 99,99")
    check((e.recv() or "").startswith("ERROR"), "out-of-bounds TURN -> ERROR")
    e.send("TURN 7,7")
    check((e.recv() or "").startswith("ERROR"), "TURN on occupied cell -> ERROR")
    e.send("TURN garbage")
    check((e.recv() or "").startswith("ERROR"), "malformed TURN -> ERROR")
    check(e.close() == 0, "END exits with code 0")


def test_board_sync():
    print("[BOARD sync]")
    e = Engine()
    e.send("START 15")
    e.recv()
    e.send("BOARD")
    stones = [(7, 7, 1), (8, 8, 2), (7, 8, 1), (8, 9, 2),
              (7, 9, 1), (8, 10, 2), (7, 10, 1)]
    for x, y, c in stones:
        e.send(f"{x},{y},{c}")
    e.send("DONE")
    mv = parse_move(e.recv())
    occupied = {(x, y) for x, y, _ in stones}
    check(mv is not None and mv not in occupied, "BOARD sync -> legal reply move")

    # Full draw board -> no legal move -> ERROR.
    e.send("BOARD")
    for y in range(SIZE):
        for x in range(SIZE):
            c = 1 if (x + 2 * y) % 5 < 2 else 2
            e.send(f"{x},{y},{c}")
    e.send("DONE")
    check((e.recv() or "").startswith("ERROR"), "full BOARD -> ERROR no legal move")
    e.close()


def test_rule_switch():
    print("[rule switch]")
    e = Engine()
    e.send("START 15")
    e.recv()
    e.send("INFO rule 2")
    e.send("TURN 7,7")
    mv = parse_move(e.recv())
    check(mv is not None and mv != (7, 7), "Renju mode: TURN answered normally")
    e.send("INFO rule 1")
    e.send("RESTART")
    check(e.recv() == "OK", "freestyle mode: RESTART -> OK")
    e.send("BEGIN")
    mv2 = parse_move(e.recv())
    check(mv2 is not None, "freestyle mode: BEGIN answered normally")
    e.close()


def test_self_play():
    print("[full self-play game]")
    black = Engine()
    white = Engine()
    for e in (black, white):
        e.send("START 15")
        e.recv()
    black.send("BEGIN")

    board = [[0] * SIZE for _ in range(SIZE)]
    moves = 0
    outcome = None
    mover = black
    mv = parse_move(black.recv())
    while mv is not None:
        legal = 0 <= mv[0] < SIZE and 0 <= mv[1] < SIZE and board[mv[1]][mv[0]] == 0
        check(legal, f"move #{moves + 1} {mv} is legal")
        if not legal:
            outcome = "invalid-move"
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
        reply = other.recv()
        nxt = parse_move(reply)
        if nxt is None:
            check(False, f"engine replied with a valid move, got {reply!r}")
            outcome = "bad-reply"
            break
        mv = nxt
        mover = other

    check(outcome in ("black-win", "white-win", "draw"),
          f"game reached a natural end ({outcome}, {moves} moves)")
    check(black.close() == 0, "black engine exits cleanly")
    check(white.close() == 0, "white engine exits cleanly")


def main():
    test_version_flag()
    test_basic_commands()
    test_board_sync()
    test_rule_switch()
    test_self_play()
    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
