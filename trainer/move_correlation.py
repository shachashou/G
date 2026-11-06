#!/usr/bin/env python3
"""Ticket 15/17 follow-up: is the NNUE eval aligned with the teacher's moves?

One of the three still-open explanations from tickets 15 and 17 is "the
training target (MSE on tanh(score/400)) is not aligned with strength". This
diagnostic tests that directly, at low cost: for each frozen-validation
position, evaluate every empty cell's child position with the candidate
weights, then report where the teacher's *recorded* move (the move the search
actually played) ranks in that ordering.

A candidate whose eval has learned the teacher's preference will place the
teacher move near the top (top-1 hit rate well above the random baseline
``1 / #moves``). A candidate whose eval is effectively uninformative will rank
near the random baseline. This is a pure read-only diagnostic: no training,
no new data, no formal openings, no budget ledger.

Usage (project venv)::

  .venv\\Scripts\\python.exe trainer\\move_correlation.py ^
      --data data\\raw\\bench-d4\\train.binpack ^
      --frozen-validation data\\m5\\frozen_validation.json ^
      --weights data\\m5\\m5_cmp_large_fidelity\\candidate.nnue ^
      --probe build\\engine\\Release\\nnue_probe.exe ^
      --out data\\m5\\move_correlation.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer"))
sys.path.insert(0, str(ROOT / "tools"))

import binpack  # noqa: E402
import validation_freeze as vf  # noqa: E402
from run_support import file_identity, write_json  # noqa: E402

CELLS = 225


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True,
                        help="frozen source binpack (bench-d4/train.binpack)")
    parser.add_argument("--frozen-validation", type=Path, required=True,
                        help="frozen validation JSON (restricts to val games)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="candidate GMNNUE01 weights (required unless "
                             "--mode handcrafted)")
    parser.add_argument("--probe", type=Path,
                        default=ROOT / "build" / "engine" / "Release" / "nnue_probe.exe")
    parser.add_argument("--backend", choices=("avx2", "scalar"), default="avx2")
    parser.add_argument("--mode", choices=("nnue", "handcrafted"), default="nnue",
                        help="nnue evaluates with --weights; handcrafted uses "
                             "the threat evaluator as a teacher-side baseline")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=None,
                        help="cap the number of positions (default: all val)")
    parser.add_argument("--seed", type=int, default=42,
                        help="sampling seed when --max-positions is set")
    return parser.parse_args(argv)


def board_text(black: bytes, white: bytes) -> str:
    """225-char board string for nnue_probe ('.'/'x'/'o')."""
    return "".join(
        "x" if black[i] else ("o" if white[i] else ".")
        for i in range(CELLS)
    )


def empty_cells(black: bytes, white: bytes) -> list[int]:
    return [i for i in range(CELLS) if not black[i] and not white[i]]


def child_board(black: bytes, white: bytes, stm: int, move: int):
    """Place the side-to-move's stone at `move` and flip the side to move."""
    nb = bytearray(black)
    nw = bytearray(white)
    if stm == 0:
        nb[move] = 1
    else:
        nw[move] = 1
    return bytes(nb), bytes(nw), 1 - stm


def validation_indices(reader: binpack.BinpackReader, frozen: dict) -> list[int]:
    game_map = reader.game_index_map()
    return vf.frozen_validation_record_ids(game_map, frozen)


def sample_indices(indices: list[int], limit: int | None, seed: int) -> list[int]:
    if limit is None or limit >= len(indices):
        return list(indices)
    import random
    rng = random.Random(seed)
    return sorted(rng.sample(indices, limit))


def evaluate_children(probe: Path, weights: Path | None, backend: str,
                      mode: str, boards: list[str]) -> list[int]:
    """Batch-evaluate child boards via one nnue_probe invocation.

    ``subprocess.run(input=...)`` handles the stdin/stdout pairing internally
    (a dedicated reader thread), so a large batch cannot deadlock the way a
    hand-rolled ``write-all-then-read-all`` loop would.
    """
    argv = [str(probe)]
    if mode == "nnue":
        argv += ["--weights", str(weights), "--backend", backend, "board"]
    else:
        argv += ["handcrafted"]
    payload = "".join(b + "\n" for b in boards)
    proc = subprocess.run(argv, input=payload, capture_output=True, text=True,
                          timeout=600)
    if proc.returncode != 0:
        raise SystemExit(f"nnue_probe failed (exit {proc.returncode}): "
                         f"{proc.stderr.strip()}")
    return [int(line) for line in proc.stdout.splitlines()]


def rank_of(teacher_move: int, child_evals: list[int], moves: list[int]) -> tuple[int, int]:
    """Rank of the teacher move under negamax ordering (higher is better).

    Returns (rank, num_moves) with rank 0 = the best move. The recorded move
    is the teacher's move on the CURRENT side-to-move, so its child is
    evaluated from the opponent's perspective; the current side's value is
    ``-child_eval``.
    """
    values = {m: -e for m, e in zip(moves, child_evals)}
    teacher_value = values[teacher_move]
    better = sum(1 for v in values.values() if v > teacher_value)
    return better, len(moves)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frozen = vf.load_frozen(args.frozen_validation)
    with binpack.BinpackReader(args.data) as reader:
        vf.verify_frozen_data(frozen, args.data)
        indices = sample_indices(
            validation_indices(reader, frozen),
            args.max_positions, args.seed)

    # Build all (board, stm, move) child boards first so a single probe
    # process can evaluate them in one pass.
    boards: list[str] = []
    cases: list[tuple[int, list[int]]] = []  # (teacher_move, moves)
    with binpack.BinpackReader(args.data) as reader:
        for index in indices:
            record = reader[index]
            black = binpack.unpack_plane(record.black)
            white = binpack.unpack_plane(record.white)
            moves = empty_cells(black, white)
            if record.move not in moves:
                # Recorded move must be an empty cell; skip malformed/terminal.
                continue
            for move in moves:
                nb, nw, stm_after = child_board(black, white, record.stm, move)
                boards.append(f"{stm_after} {board_text(nb, nw)}")
            cases.append((record.move, moves))

    if not cases:
        raise SystemExit("no validation positions with a recorded move")

    if args.mode == "nnue" and args.weights is None:
        raise SystemExit("--weights is required unless --mode handcrafted")
    child_evals = evaluate_children(args.probe, args.weights, args.backend,
                                    args.mode, boards)

    ranks: list[tuple[int, int]] = []
    offset = 0
    for teacher_move, moves in cases:
        n = len(moves)
        ranks.append(rank_of(teacher_move, child_evals[offset:offset + n], moves))
        offset += n

    # Aggregates.
    total_positions = len(ranks)
    top1 = sum(1 for r, _ in ranks if r == 0)
    top3 = sum(1 for r, _ in ranks if r < 3)
    top5 = sum(1 for r, _ in ranks if r < 5)
    percentiles = [r / max(1, n - 1) for r, n in ranks]
    mean_rank = sum(r for r, _ in ranks) / total_positions
    mean_percentile = sum(percentiles) / total_positions
    # Random baseline: each move uniformly ranked, so expected top-k hit rate
    # is k / #moves and expected percentile is 0.5.
    mean_moves = sum(n for _, n in ranks) / total_positions
    random_top1 = 1.0 / mean_moves
    random_top3 = min(1.0, 3.0 / mean_moves)
    random_top5 = min(1.0, 5.0 / mean_moves)

    report = {
        "schema": 1,
        "mode": args.mode,
        "data": file_identity(args.data),
        "frozen_validation": file_identity(args.frozen_validation),
        "weights": file_identity(args.weights) if args.weights else None,
        "backend": args.backend,
        "positions": total_positions,
        "mean_moves_per_position": round(mean_moves, 3),
        "results": {
            "mean_rank": round(mean_rank, 3),
            "mean_rank_percentile": round(mean_percentile, 4),
            "top1_hit_rate": round(top1 / total_positions, 4),
            "top3_hit_rate": round(top3 / total_positions, 4),
            "top5_hit_rate": round(top5 / total_positions, 4),
        },
        "random_baseline": {
            "top1_hit_rate": round(random_top1, 4),
            "top3_hit_rate": round(random_top3, 4),
            "top5_hit_rate": round(random_top5, 4),
            "mean_rank_percentile": 0.5,
        },
        "rank_distribution": {
            "top1": top1, "top3": top3, "top5": top5,
            "total": total_positions,
        },
    }
    write_json(args.out, report)
    print(f"positions: {total_positions} (mean {mean_moves:.1f} moves each)")
    print(f"model eval: top1={top1}/{total_positions} ({top1 / total_positions:.1%}), "
          f"top3={top3 / total_positions:.1%}, top5={top5 / total_positions:.1%}, "
          f"mean rank={mean_rank:.1f} (percentile {mean_percentile:.3f})")
    print(f"random     : top1={random_top1:.1%}, top3={random_top3:.1%}, "
          f"top5={random_top5:.1%}, percentile 0.500")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
