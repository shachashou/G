#!/usr/bin/env python3
"""Black-box tests for the JSON-lines authoritative executor (ticket 02).

Drives ``gomoku_nnue.exe --executor`` through tools/executor.py. The Python
side never computes Renju legality itself: it only sends boards and asserts
on the verdicts returned by the executor.

Usage: python executor_blackbox.py [gomoku_nnue.exe] [tools/executor.py]
"""

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



def test_vcf_boundary(client):
    print("[VCF boundary: upstream solver, budgets, evidence]")
    about = client.about()
    check(str(about.get("vcf_engine", "")).startswith("quintet@"),
          "about identifies the vendored upstream VCF implementation")
    check(about.get("vcf_policy") in ("off", "root", "root_and_children"),
          "about reports the VCF call-site policy")

    # Counter-four position: Black H8,J8,K8,I7,L7,I6 vs White H9,L8,J7,K6,I5,L5.
    blacks = [(7, 7), (9, 7), (10, 7), (8, 6), (11, 6), (8, 5)]
    whites = [(7, 8), (11, 7), (9, 6), (10, 5), (8, 4), (11, 4)]
    r = client.rebuild(cells_from(blacks, whites), "black", "renju")
    check(r.get("ok") is True, "VCF position rebuilt")
    response = client.vcf(max_nodes=50000, max_attack_plies=10)
    check(response.get("ok") is True, "vcf query accepted")
    report = response.get("vcf", {})
    check(report.get("status") == "proven", "counter-four line is proven")
    check(report.get("proven") is True and report.get("unknown") is False,
          "proven is not reported as unknown")
    check(report.get("first_move") == [8, 7], "first move I8 reported")
    check([tuple(p) for p in report.get("variation", [])]
          == [(8, 7), (6, 7), (8, 9), (8, 8), (9, 8)],
          "variation is the full forced line")
    check(report.get("end", {}).get("kind") == "fours",
          "evidence names a double/straight four ending")
    check(report.get("budget", {}).get("max_nodes") == 50000
          and report.get("budget", {}).get("max_attack_plies") == 10,
          "budget is echoed in the result")
    check(report.get("nodes", 0) > 0, "node cost reported")
    check(response.get("evidence_checked") is True,
          "evidence is marked as re-checked by the executor")
    replayed = client.replay([list(p) for p in report["variation"]], "renju")
    check(replayed.get("ok") is True,
          "the proven variation replays legally through the executor")
    after = client.state()
    check(after["move_count"] == 12 and after["history"] == [],
          "vcf query does not consume the snapshot")

    # Budget exhaustion is unknown, never "no mate".
    tight = client.vcf(max_nodes=1, max_attack_plies=10)["vcf"]
    check(tight.get("status") == "budget_exhausted",
          "tiny node budget -> budget_exhausted")
    check(tight.get("unknown") is True and tight.get("proven") is False,
          "budget exhaustion is reported as unknown")
    check(tight.get("variation") == [], "no variation without a proof")

    # A quiet position has no VCF at all.
    client.new_game("renju")
    client.play(7, 7)
    client.play(8, 8)
    quiet = client.vcf()["vcf"]
    check(quiet.get("status") == "disproven",
          "quiet position -> disproven within the budget")
    check(quiet.get("first_move") is None, "disproven has no first move")

    # Freestyle is not modelled by the upstream solver.
    client.new_game("freestyle")
    free = client.vcf()["vcf"]
    check(free.get("status") == "unsupported_rule",
          "freestyle VCF reports unsupported_rule")

    # Out-of-range parameters are input errors, not tactical verdicts.
    client.new_game("renju")
    bad = client.vcf(max_attack_plies=0)
    check(bad.get("ok") is False and "max_attack_plies" in bad.get("error", ""),
          "invalid parameter rejected with an input error")

    # Finished position: the adapter refuses to answer.
    five = cells_from([(3, 7), (4, 7), (5, 7), (6, 7), (7, 7)],
                      [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)])
    client.rebuild(five, "white", "renju")
    finished = client.vcf()["vcf"]
    check(finished.get("status") == "invalid_input",
          "finished position -> invalid_input")
    check(bool(finished.get("message")), "invalid input explains itself")

    # think reports the VCF state it actually used.
    client.rebuild(cells_from(blacks, whites), "black", "renju")
    thought = client.think(depth=2, vcf_policy="root")
    check(thought.get("vcf_policy") == "root", "think echoes the VCF policy")
    check(thought.get("from_vcf") is True, "think used the proven VCF line")
    check(thought.get("move") == [8, 7], "engine plays the proven first move")
    check(thought.get("vcf", {}).get("status") == "proven",
          "think reports the VCF status")
    thought = client.think(depth=2, vcf_policy="off")
    check(thought.get("vcf") is None, "policy off -> no VCF report")


def test_stop_interrupts_work():
    print("[stop: closing the client kills in-flight engine work]")
    client = ExecutorClient(EXE)
    errors = []

    def worker():
        try:
            # A fixed deep search has no clock, so it cannot finish before the
            # stop; the point is the kill, not the answer.
            client.think(depth=20)
        except Exception as exc:  # noqa: BLE001 - the point is the failure mode
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(0.2)
    check(thread.is_alive(), "long think is still running")
    proc = client._proc
    client.close()
    thread.join(5)
    check(not thread.is_alive(), "closing the client ends the worker")
    check(bool(errors), "in-flight request reports an executor error")
    check(proc.poll() is not None, "engine process is gone after stop")

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
        test_vcf_boundary(client)
        test_stop_interrupts_work()
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
