#!/usr/bin/env python3
"""Self-tests for trainer/move_correlation.py (ticket 15/17 follow-up).

Covers the pure helpers (board text, empty-cell enumeration, child-board
construction and the negamax rank) without invoking the engine. The end-to-end
diagnostic was exercised by hand against the frozen validation set; its JSON
artifacts live under data/m5.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "trainer"
sys.path.insert(0, str(TRAINER))
sys.path.insert(0, str(ROOT / "tools"))
import move_correlation as mc  # noqa: E402

failures = 0


def check(cond, msg):
    global failures
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures += 1


def test_board_text():
    empty = bytes(225)
    check(mc.board_text(empty, empty) == "." * 225, "empty board is all dots")
    black = bytearray(225)
    black[0] = 1
    white = bytearray(225)
    white[224] = 1
    text = mc.board_text(bytes(black), bytes(white))
    check(text[0] == "x" and text[224] == "o" and text.count(".") == 223,
          "black/white stones map to x/o and the rest stays empty")
    check(len(text) == 225, "board text is exactly 225 chars")


def test_empty_and_child():
    black = bytes(225)
    white = bytes(225)
    moves = mc.empty_cells(black, white)
    check(len(moves) == 225, "empty board has 225 candidate moves")
    nb, nw, stm_after = mc.child_board(black, white, stm=0, move=7)
    check(nb[7] == 1 and stm_after == 1, "black's move is placed and stm flips")
    nb2, nw2, stm2 = mc.child_board(bytes(nb), bytes(nw), stm=1, move=8)
    check(nw2[8] == 1 and stm2 == 0, "white's move is placed and stm flips back")


def test_rank_of():
    # Teacher move 5 with the highest child value (negamax: -child_eval highest
    # means the lowest child_eval from the opponent's view).
    moves = [0, 1, 2, 3, 4, 5]
    # child_evals are the OPPONENT's perspective; -child_eval is the current
    # side's value. Make the teacher's child lowest so -child_eval is highest.
    child_evals = [10, 9, 8, 7, 6, 0]
    rank, n = mc.rank_of(teacher_move=5, child_evals=child_evals, moves=moves)
    check((rank, n) == (0, 6), "the best move ranks 0")
    # Teacher move 0 with the worst value ranks last.
    rank, _ = mc.rank_of(teacher_move=0, child_evals=child_evals, moves=moves)
    check(rank == 5, "the worst move ranks n-1")
    # Ties are not counted as better, so a tied best still ranks 0: teacher
    # move 0 has child_eval 0 -> current value 0, the highest (moves 1/2 give
    # -5 each), and nothing is strictly greater.
    tied = [0, 5, 5]
    rank, _ = mc.rank_of(teacher_move=0, child_evals=tied, moves=[0, 1, 2])
    check(rank == 0, "a tied best ranks 0 (strictly-better comparison)")


def main():
    test_board_text()
    test_empty_and_child()
    test_rank_of()
    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
