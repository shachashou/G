#!/usr/bin/env python3
"""Ticket 14: dataset quality report for any binpack file (no torch needed).

Reports independent-position counts (raw and D4 orbits), train/validation
orbit overlap, ply / side-to-move distributions, score quantiles and
saturated-score ratio. D4 eight-fold augmentation is NOT counted as eight
independent positions.

Example::

  .venv\\Scripts\\python.exe tools\\dataset_profile.py ^
      --data data\\raw\\m5_fresh_200k\\train.binpack ^
      --out data\\m5\\fresh_200k_profile.json
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "trainer"), str(ROOT / "tools")]

import binpack  # noqa: E402
import dataset  # noqa: E402
import validation_freeze as vf  # noqa: E402
from run_support import file_identity, write_json  # noqa: E402


def quantile(sorted_values: list[int], q: float):
    if not sorted_values:
        return None
    return sorted_values[int((len(sorted_values) - 1) * q)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="binpack file to profile")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42,
                        help="only for an informational train/val orbit split; "
                             "the authoritative split is the frozen one")
    args = parser.parse_args(argv)
    path = Path(args.data)

    raw_positions: set[tuple] = set()
    orbits: dict[tuple, list[bool]] = {}
    scores: list[int] = []
    plies: Counter[int] = Counter()
    stm: Counter[int] = Counter()
    saturated = 0
    with binpack.BinpackReader(path) as reader:
        game_map = reader.game_index_map()
        n_games = game_map[-1] + 1 if len(game_map) else 0
        val_games = dataset.choose_validation_games(
            n_games, args.val_fraction, args.seed)
        for index, game in enumerate(game_map):
            record = reader[index]
            sparse = record.feature_indices()
            raw_positions.add((record.stm, tuple(sorted(sparse))))
            key = vf.canonical_position(sparse, record.stm)
            orbits.setdefault(key, []).append(game in val_games)
            scores.append(record.score)
            plies[record.ply] += 1
            stm[record.stm] += 1
            if abs(record.score) == binpack.SCORE_MAX:
                saturated += 1

    train_orbits = {k for k, flags in orbits.items() if any(not f for f in flags)}
    val_orbits = {k for k, flags in orbits.items() if any(f for f in flags)}
    val_records = sum(1 for game in game_map if game in val_games)
    ordered = sorted(scores)
    report = {
        "data": file_identity(path),
        "records": len(scores),
        "games": n_games,
        "distinct_raw_positions": len(raw_positions),
        "distinct_D4_orbits": len(orbits),
        "training_orbits": len(train_orbits),
        "validation_orbits": len(val_orbits),
        "validation_records": val_records,
        "validation_orbits_also_in_training": len(train_orbits & val_orbits),
        "ply_distribution": dict(sorted(plies.items())),
        "side_to_move_distribution": dict(sorted(stm.items())),
        "score_quantiles": {
            "p05": quantile(ordered, 0.05), "p25": quantile(ordered, 0.25),
            "p50": quantile(ordered, 0.50), "p75": quantile(ordered, 0.75),
            "p95": quantile(ordered, 0.95),
        },
        "score_mean": round(statistics.mean(scores), 3) if scores else None,
        "target_space_mean": round(
            statistics.mean(math.tanh(s / 400) for s in scores), 6)
            if scores else None,
        "saturated_scores": saturated,
        "saturated_ratio": round(saturated / len(scores), 6) if scores else None,
        "note": ("informational split only; ticket 13's frozen validation "
                 "JSON is authoritative. D4 augmentation does not create "
                 "independent games."),
    }
    write_json(Path(args.out), report)
    print(f"records={report['records']} games={report['games']} "
          f"raw_positions={report['distinct_raw_positions']} "
          f"D4_orbits={report['distinct_D4_orbits']} "
          f"validation_orbits_also_in_training="
          f"{report['validation_orbits_also_in_training']}")
    print(f"saturated={saturated} ({report['saturated_ratio']}) "
          f"p05={report['score_quantiles']['p05']} "
          f"p95={report['score_quantiles']['p95']}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
