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
import math
import random
import statistics
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
    parser.add_argument("--move-cases", type=Path, help="schema2 immutable legal/nonterminal cases")
    parser.add_argument("--metric-contract", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
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


def tie_stats(teacher_move, child_evals, moves):
    if len(child_evals) != len(moves) or not moves or len(set(moves)) != len(moves) or teacher_move not in moves:
        raise ValueError("invalid candidate scores/moves")
    if not all(math.isfinite(v) for v in child_evals):
        raise ValueError("non-finite candidate score")
    teacher = child_evals[moves.index(teacher_move)]
    b = sum(v < teacher for v in child_evals)  # child opponent scores, negated for ranking
    t = sum(v == teacher for v in child_evals)
    n = len(moves)
    return {"b": b, "t": t, "n": n, "teacher_child_eval": teacher,
            "top1_credit": 1/t if b == 0 else 0,
            "top3_credit": min(1, max(0, 3-b)/t),
            "top5_credit": min(1, max(0, 5-b)/t),
            "midrank": b + (t-1)/2,
            "max_tie_size": max(child_evals.count(v) for v in set(child_evals))}


def aggregate_cases(cases, evaluations):
    if len(evaluations) != sum(len(c["moves"]) for c in cases):
        raise ValueError("candidate response count mismatch")
    rows, offset = [], 0
    for case in cases:
        n = len(case["moves"])
        row = tie_stats(case["teacher_move"], evaluations[offset:offset+n], case["moves"])
        row.update(record_id=case["record_id"], game_id=case["game_id"], stm=case["stm"], ply=case["ply"])
        rows.append(row)
        offset += n
    if not rows:
        raise ValueError("empty move assessment")
    games = sorted({r["game_id"] for r in rows})
    results = {"top1_tie_adjusted": statistics.mean(r["top1_credit"] for r in rows),
               "top1_credit_sum": sum(r["top1_credit"] for r in rows), "positions": len(rows),
               "top3_tie_adjusted": statistics.mean(r["top3_credit"] for r in rows),
               "top5_tie_adjusted": statistics.mean(r["top5_credit"] for r in rows),
               "unique_top1": sum(r["b"] == 0 and r["t"] == 1 for r in rows)/len(rows),
               "optimistic_top1": sum(r["b"] == 0 for r in rows)/len(rows),
               "random_top1": statistics.mean(1/r["n"] for r in rows),
               "random_top3": statistics.mean(min(3, r["n"])/r["n"] for r in rows),
               "random_top5": statistics.mean(min(5, r["n"])/r["n"] for r in rows),
               "mean_midrank_percentile": statistics.mean(r["midrank"]/max(1, r["n"]-1) for r in rows),
               "all_tied_fraction": sum(r["t"] == r["n"] for r in rows)/len(rows),
               "mean_teacher_tie_size": statistics.mean(r["t"] for r in rows),
               "max_tie_size": max(r["max_tie_size"] for r in rows),
               "games": len(games),
               "game_macro_top1": statistics.mean(statistics.mean(r["top1_credit"] for r in rows if r["game_id"] == g) for g in games)}
    results["per_head"] = {str(h): {"positions": sum(r["stm"] == h for r in rows),
        "top1": statistics.mean(r["top1_credit"] for r in rows if r["stm"] == h)} for h in {r["stm"] for r in rows}}
    results["per_ply"] = {str(p): {"positions": sum(r["ply"]//10 == p for r in rows),
        "top1": statistics.mean(r["top1_credit"] for r in rows if r["ply"]//10 == p)} for p in {r["ply"]//10 for r in rows}}
    return results, rows


def freeze_cases(data: Path, frozen: dict, probe: Path, out: Path):
    from teacher_labels import probe_rows
    with binpack.BinpackReader(data) as reader:
        vf.verify_frozen_data(frozen, data)
        game_map = reader.game_index_map()
        records = [reader[i] for i in validation_indices(reader, frozen)]
    responses = probe_rows(probe, "candidates", records)
    cases = []
    for record, response in zip(records, responses):
        entries = response["candidates"]
        moves = [e["move"] for e in entries]
        if moves != sorted(set(moves)) or any(type(m) is not int or not 0 <= m < CELLS for m in moves):
            raise ValueError("invalid candidate response")
        if response["result"]:
            bucket = "terminal-parent"
        elif record.move not in moves:
            raise ValueError(f"teacher move missing from legal candidates at {record.index}")
        elif any(e["result"] for e in entries):
            bucket = "terminal-tactical"
        elif len(moves) < 2:
            bucket = "single-candidate"
        else:
            bucket = "primary"
        black, white = binpack.unpack_plane(record.black), binpack.unpack_plane(record.white)
        cases.append({"record_id": record.index, "game_id": game_map[record.index], "stm": record.stm,
                      "ply": record.ply, "board": board_text(black, white), "teacher_move": record.move,
                      "moves": moves, "child_results": [e["result"] for e in entries], "bucket": bucket})
    out.write_text("".join(json.dumps(c, separators=(",", ":")) + "\n" for c in cases), encoding="utf-8")
    primary = [c for c in cases if c["bucket"] == "primary"]
    return {"schema": 2, "cases": file_identity(out), "data": file_identity(data),
            "candidate_rule": "renju-radius1-all-children-ongoing", "tie_rule": "uniform-within-equal-scores",
            "total_positions": len(cases), "primary_positions": len(primary),
            "primary_games": len({c["game_id"] for c in primary}),
            "buckets": {b: sum(c["bucket"] == b for c in cases) for b in {c["bucket"] for c in cases}},
            "coverage_passed": len(primary) >= 200 and len({c["game_id"] for c in primary}) >= 20,
            "probe": file_identity(probe), "bootstrap_seed": 20261118, "bootstrap_repeats": 10000,
            "thresholds": {"min_top1": 0.20, "min_improvement": 0.10, "teacher_fraction": 0.50}}


def load_cases(path: Path, metric_path: Path):
    from teacher_labels import verify_identity
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    verify_identity(metric["cases"])
    verify_identity(metric["data"])
    if file_identity(path)["sha256"] != metric["cases"]["sha256"] or metric.get("schema") != 2:
        raise ValueError("move cases/metric contract mismatch")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(cases) != metric["total_positions"] or len({c["record_id"] for c in cases}) != len(cases):
        raise ValueError("move case membership mismatch")
    primary = [c for c in cases if c["bucket"] == "primary"]
    if len(primary) != metric["primary_positions"]:
        raise ValueError("move primary membership mismatch")
    return primary, metric


def child_lines(cases):
    boards = []
    for case in cases:
        for move in case["moves"]:
            board = list(case["board"])
            if board[move] != ".":
                raise ValueError("candidate is occupied")
            board[move] = "x" if case["stm"] == 0 else "o"
            boards.append(f"{1-case['stm']} {''.join(board)}")
    return boards


def assess_cases(probe: Path, weights: Path | None, mode: str, backend: str,
                 cases_path: Path, metric_path: Path, out: Path, timeout=120):
    cases, metric = load_cases(cases_path, metric_path)
    if file_identity(probe)["sha256"] != metric["probe"]["sha256"]:
        raise ValueError("candidate probe differs from frozen metric probe")
    boards = child_lines(cases)
    argv = [str(probe.resolve()), "handcrafted"] if mode == "handcrafted" else [
        str(probe.resolve()), "--weights", str(weights), "--backend", backend, "board"]
    proc = subprocess.run(argv, input="".join(b+"\n" for b in boards), text=True, capture_output=True, timeout=timeout)
    if proc.returncode:
        raise ValueError(f"candidate inference failed: {proc.stderr}")
    scores = [int(v) for v in proc.stdout.splitlines()]
    results, rows = aggregate_cases(cases, scores)
    per_case = out.with_suffix(".cases.jsonl")
    per_case.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    report = {"schema": 2, "complete": True, "mode": mode, "backend": backend,
              "probe": file_identity(probe), "weights": file_identity(weights) if weights else None,
              "move_cases": file_identity(cases_path), "metric_contract": file_identity(metric_path),
              "results": results, "per_case": file_identity(per_case),
              "static_limit_hits": sum(abs(s) >= 900000 for s in scores)}
    write_json(out, report)
    return report


def cluster_interval(candidate_rows, reference_rows, repeats=10000, seed=20261118):
    ids = [(r["record_id"], r["game_id"]) for r in candidate_rows]
    if not reference_rows or any([(r["record_id"], r["game_id"]) for r in rows] != ids for rows in reference_rows):
        raise ValueError("bootstrap reference membership mismatch")
    games = sorted({g for _, g in ids})
    if len(games) < 2:
        raise ValueError("bootstrap needs at least two complete games")
    tables = []
    for rows in [candidate_rows, *reference_rows]:
        tables.append({g: (sum(r["top1_credit"] for r in rows if r["game_id"] == g),
                           sum(r["game_id"] == g for r in rows)) for g in games})
    rng, differences = random.Random(seed), []
    for _ in range(repeats):
        selected = rng.choices(games, k=len(games))
        means = [sum(t[g][0] for g in selected)/sum(t[g][1] for g in selected) for t in tables]
        differences.append(means[0]-max(means[1:]))
    differences.sort()
    return [differences[int((repeats-1)*0.025)], differences[int((repeats-1)*0.975)]]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.move_cases:
        if args.metric_contract is None or args.max_positions is not None:
            raise ValueError("schema2 requires metric contract and complete positions")
        if args.mode == "nnue" and args.weights is None:
            raise ValueError("NNUE requires weights")
        frozen = vf.load_frozen(args.frozen_validation)
        vf.verify_frozen_data(frozen, args.data)
        metric = json.loads(args.metric_contract.read_text(encoding="utf-8"))
        if metric["data"]["sha256"] != file_identity(args.data)["sha256"]:
            raise ValueError("move metric data differs from supplied source")
        if frozen.get("schema") == 2 and frozen["move_metric_contract"]["sha256"] != file_identity(args.metric_contract)["sha256"]:
            raise ValueError("move metric differs from derived freeze")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        assess_cases(args.probe, args.weights, args.mode, args.backend, args.move_cases,
                     args.metric_contract, args.out, args.timeout)
        return 0
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
