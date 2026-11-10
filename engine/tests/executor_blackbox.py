#!/usr/bin/env python3
"""Black-box tests for the JSON-lines authoritative executor (ticket 02).

Drives ``gomoku_nnue.exe --executor`` through tools/executor.py. The Python
side never computes Renju legality itself: it only sends boards and asserts
on the verdicts returned by the executor.

Usage: python executor_blackbox.py [gomoku_nnue.exe] [tools/executor.py]
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

EXE = sys.argv[1] if len(sys.argv) > 1 else "build/engine/gomoku_nnue.exe"
CLIENT_MODULE = sys.argv[2] if len(sys.argv) > 2 else None
if CLIENT_MODULE:
    sys.path.insert(0, str(Path(CLIENT_MODULE).resolve().parent))
from executor import ExecutorClient  # noqa: E402

SIZE = 15
STATE_FIELDS = ("cells", "side_to_move", "result", "rule", "move_count",
                "history", "last_move")
failures = 0


def check(cond, msg):
    global failures
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures += 1


def state_equal(a, b):
    return all(a.get(k) == b.get(k) for k in STATE_FIELDS)


def cells_from(blacks, whites):
    cells = [0] * (SIZE * SIZE)
    for x, y in blacks:
        cells[y * SIZE + x] = 1
    for x, y in whites:
        cells[y * SIZE + x] = 2
    return cells


def test_about(client):
    print("[about: evaluation identity]")
    r = client.about()
    check(r.get("ok") is True, "about -> ok")
    check(r.get("engine") == "GomokuNNUE", "engine identity reported")
    check(r.get("eval") == "handcrafted", "no model -> handcrafted evaluation")
    check(r.get("nnue_loaded") is False, "no model -> not marked as NNUE")
    check("no trained NNUE" in r.get("eval_description", ""),
          "handcrafted eval explicitly says no trained NNUE is loaded")
    check(r.get("nnue_backend") == "none", "no model -> no NNUE backend")


def test_new_game_and_state(client):
    print("[new_game / state]")
    r = client.new_game("renju")
    check(r.get("ok") is True, "new_game -> ok")
    check(r.get("side_to_move") == "black", "black moves first")
    check(r.get("result") == "ongoing", "game ongoing")
    check(r.get("rule") == "renju", "Renju rule active")
    check(len(r.get("cells", [])) == 225 and all(c == 0 for c in r["cells"]),
          "empty 15x15 board")
    check(r.get("move_count") == 0 and r.get("history") == [],
          "no moves or history")
    check(client.state().get("last_move") is None, "no last move")


def test_legal_moves(client):
    print("[complete legal move set]")
    r = client.legal_moves()
    check(r.get("ok") is True and r.get("count") == 225,
          "empty board -> 225 legal points")
    check(len(r.get("moves", [])) == 225, "all legal points listed")
    check(r.get("forbidden") == [], "no forbidden points on empty board")


def test_move_reason_and_illegal(client):
    print("[move reasons / illegal requests do not change state]")
    r = client.move_reason(7, 7)
    check(r.get("legal") is True and r.get("reason") == "legal",
          "empty point -> legal")
    r = client.move_reason(99, 99)
    check(r.get("reason") == "out_of_bounds", "out-of-bounds reason")
    before = client.state()
    r = client.play(99, 99)
    check(r.get("ok") is False and r.get("reason") == "out_of_bounds",
          "out-of-bounds play rejected")
    check(state_equal(before, client.state()),
          "illegal play leaves position unchanged")
    check(client.play(7, 7).get("ok") is True, "legal play accepted")
    after = client.state()
    r = client.play(7, 7)
    check(r.get("ok") is False and r.get("reason") == "occupied",
          "occupied play rejected")
    check(state_equal(after, client.state()),
          "occupied play leaves position unchanged")
    check(len(after["history"]) == 1 and after["history"][0] == [7, 7],
          "move history recorded")


def test_forbidden_reasons(client):
    print("[Renju rule coverage comes from the executor]")
    # Classic shapes from renju_tests.cpp are only laid out here; the verdict
    # (including the reason) is computed by the C++ executor.
    cases = [
        ("overline / 长连",
         [(3, 7), (4, 7), (5, 7), (7, 7), (8, 7)], [(0, 0), (2, 0)],
         (6, 7), "forbidden_overline"),
        ("double four / 四四",
         [(5, 7), (6, 7), (8, 7), (7, 5), (7, 6), (7, 8)], [(0, 0), (2, 0)],
         (7, 7), "forbidden_double_four"),
        ("double three / 三三",
         [(6, 7), (8, 7), (7, 6), (7, 8)], [(0, 0), (2, 0)],
         (7, 7), "forbidden_double_three"),
        ("sleeping three / 假活三 (legal)",
         [(6, 7), (8, 7), (7, 6), (7, 8)], [(9, 7), (0, 0)],
         (7, 7), "legal"),
        ("nested false three / 嵌跳三 (legal)",
         [(6, 7), (8, 7), (7, 6), (7, 8),
          (5, 5), (5, 6), (5, 8), (9, 5), (9, 6), (9, 8)], [(0, 0), (2, 0)],
         (7, 7), "legal"),
        ("five takes precedence / 成五优先",
         [(5, 7), (6, 7), (8, 7), (9, 7), (7, 5), (7, 6), (7, 8),
          (5, 5), (6, 6), (8, 8)], [(0, 0), (2, 0)],
         (7, 7), "legal"),
    ]
    for label, blacks, whites, (x, y), expected in cases:
        r = client.rebuild(cells_from(blacks, whites), "black", "renju")
        check(r.get("ok") is True, f"{label}: snapshot rebuilt")
        rr = client.move_reason(x, y)
        check(rr.get("reason") == expected, f"{label}: reason = {expected}")
        if expected != "legal":
            lm = client.legal_moves()
            check(any(f["x"] == x and f["y"] == y and f["reason"] == expected
                      for f in lm.get("forbidden", [])),
                  f"{label}: reported in forbidden list")
            before = client.state()
            pr = client.play(x, y)
            check(pr.get("ok") is False and pr.get("reason") == expected,
                  f"{label}: forbidden play rejected")
            check(state_equal(before, client.state()),
                  f"{label}: rejected play leaves state unchanged")
    # White has no fouls.
    r = client.rebuild(cells_from([(0, 0), (2, 0)],
                                  [(5, 7), (6, 7), (8, 7), (7, 5), (7, 6), (7, 8)]),
                       "white", "renju")
    check(r.get("ok") is True and client.move_reason(7, 7).get("reason") == "legal",
          "white: same double-four shape is legal")
    # White overline wins.
    r = client.rebuild(cells_from([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)],
                                  [(3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7)]),
                       "black", "renju")
    check(r.get("ok") is True and r.get("result") == "white_win",
          "white six-in-a-row wins in Renju")
    # Board edges are legal.
    client.new_game("renju")
    check(client.move_reason(0, 0).get("reason") == "legal", "corner (0,0) legal")
    check(client.move_reason(14, 14).get("reason") == "legal",
          "corner (14,14) legal")


def test_undo(client):
    print("[undo]")
    client.new_game("renju")
    client.play(7, 7)
    client.play(8, 8)
    r = client.undo()
    check(r.get("ok") is True and len(r.get("history", [])) == 1,
          "undo restores one-move history")
    check(r.get("side_to_move") == "white", "side to move restored")
    r = client.undo()
    check(r.get("ok") is True and r.get("move_count") == 0,
          "second undo returns to empty board")
    r = client.undo()
    check(r.get("ok") is False, "undo on empty history refused")


def test_rebuild(client):
    print("[rebuild validation]")
    client.new_game("renju")
    before = client.state()
    bad = [0] * 225
    bad[0] = 7
    r = client.rebuild(bad, "black", "renju")
    check(r.get("ok") is False and "cell" in r.get("error", ""),
          "corrupt cells rejected")
    check(state_equal(before, client.state()),
          "rejected rebuild leaves position unchanged")
    cells = cells_from([(3, 3), (4, 4)], [(10, 10)])
    r = client.rebuild(cells, "white", "renju")
    check(r.get("ok") is True and r.get("move_count") == 3, "valid snapshot rebuilt")
    check(r.get("history") == [] and r.get("last_move") is None,
          "snapshot has no move history")
    check(r.get("side_to_move_consistent") is True,
          "side_to_move consistency reported")
    five = cells_from([(3, 7), (4, 7), (5, 7), (6, 7), (7, 7)],
                      [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)])
    r = client.rebuild(five, "white", "renju")
    check(r.get("ok") is True and r.get("result") == "black_win",
          "finished snapshot reports its result")


def test_candidates_distinct(client):
    print("[legal set vs search candidate pruning]")
    client.new_game("renju")
    client.play(7, 7)
    client.play(8, 8)
    legal = {tuple(m) for m in client.legal_moves()["moves"]}
    cand = client.candidates(1)
    check(cand.get("ok") is True and len(cand["moves"]) < len(legal),
          "candidate pruning is narrower than the complete legal set")
    check(all(tuple(m) in legal for m in cand["moves"]),
          "every candidate is a legal move")
    check("NOT the complete legal move set" in cand.get("note", ""),
          "protocol explicitly distinguishes the two sets")


def test_engine_response(client):
    print("[engine response]")
    client.new_game("renju")
    before = client.state()
    r = client.think(depth=2)
    check(r.get("ok") is True and len(r.get("move", [])) == 2,
          "think returns a move")
    x, y = r["move"]
    legal = {tuple(m) for m in client.legal_moves()["moves"]}
    check((x, y) in legal, "engine move is legal")
    check(state_equal(before, client.state()),
          "think does not change the position itself")
    r = client.play(x, y)
    check(r.get("ok") is True and r.get("move_count") == 1,
          "GUI applies the engine move through play()")


def test_full_game_replay(client):
    print("[full game / replay consistency / terminal rejection]")
    client.new_game("renju")
    bx = [3, 4, 5, 6, 7]
    wx = [0, 1, 2, 3]
    moves = []
    for i in range(4):
        check(client.play(bx[i], 7).get("ok") is True, f"black move {i + 1} legal")
        moves.append([bx[i], 7])
        check(client.play(wx[i], 0).get("ok") is True, f"white move {i + 1} legal")
        moves.append([wx[i], 0])
    check(client.play(bx[4], 7).get("ok") is True, "black fifth stone legal")
    moves.append([bx[4], 7])
    final = client.state()
    check(final["result"] == "black_win", "game reaches a natural end")
    check(len(final["history"]) == len(moves), "full move record available")

    replayed = client.replay(moves, "renju")
    check(replayed.get("ok") is True and replayed.get("result") == "black_win",
          "replay reaches the same result")
    check(replayed.get("cells") == final["cells"], "replay reproduces the board")
    check(replayed.get("history") == final["history"],
          "replay reproduces the move record")
    check(replayed.get("side_to_move") == final["side_to_move"],
          "replay reproduces side to move")

    r = client.play(10, 10)
    check(r.get("ok") is False and r.get("reason") == "game_over",
          "move after game over rejected")
    check(state_equal(final, client.state()),
          "terminal rejection leaves state unchanged")
    check(client.legal_moves()["count"] == 0, "no legal moves after game over")

    r = client.replay(moves + [[10, 10]], "renju")
    check(r.get("ok") is False and r.get("reason") == "game_over",
          "replay rejects a sequence with a move after the end")
    r = client.replay([], "renju")
    check(r.get("ok") is True and r.get("move_count") == 0,
          "empty replay is valid")


def test_process_cleanup(client):
    print("[process cleanup]")
    proc = client._proc
    client.close()
    check(proc.poll() is not None, "close terminates the engine process")
    client.close()
    check(True, "close is idempotent")


def main():
    watchdog = threading.Timer(90, lambda: os._exit(1))
    watchdog.daemon = True
    watchdog.start()
    client = ExecutorClient(EXE)
    try:
        test_about(client)
        test_new_game_and_state(client)
        test_legal_moves(client)
        test_move_reason_and_illegal(client)
        test_forbidden_reasons(client)
        test_undo(client)
        test_rebuild(client)
        test_candidates_distinct(client)
        test_engine_response(client)
        test_full_game_replay(client)
        test_process_cleanup(client)
    finally:
        client.close()
        watchdog.cancel()
    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
