#!/usr/bin/env python3
"""Ticket 13: freeze a leak-free validation split, at the D4 position level.

Splitting by whole games is necessary but NOT sufficient: a validation
position can reappear inside the training split as a rotation/mirror of
itself (same colours, same side to move). This module closes that gap:

* ``canonical_position`` maps every record to the minimum of its eight
  colour-preserving D4 transforms -- two records are D4-equivalent exactly
  when they share that key and their side-to-move;
* ``freeze`` rebuilds the ticket-08 game split, scans the training side for
  equivalent positions, excludes the WHOLE offending training games (the
  validation set itself never changes), fits train-side baselines, and
  writes one frozen-identity JSON that training/evaluation must reference;
* ``check`` re-scans any additional training files against the frozen
  validation tracks and emits a train manifest (include-lists per file) so
  ticket 15 can add data without touching the frozen validation split.

Data-hash equality alone does not prove split independence; the D4 scan and
the hash check are reported as separate facts.

Pure standard library, like the rest of the trainer's split logic.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import binpack  # noqa: E402
import features  # noqa: E402
from dataset import choose_validation_games, clip_score  # noqa: E402
from run_support import file_identity, write_json  # noqa: E402

SCHEMA = 1
EXCLUSION_RULE = (
    "whole training games containing a position D4-equivalent (same colours "
    "and side to move) to any frozen validation position are excluded from "
    "training; the validation set itself is never modified"
)


def canonical_position(indices: list[int], stm: int) -> tuple:
    """(stm, min over the 8 D4 transforms of the sorted index tuple)."""
    best: tuple[int, ...] | None = None
    for t in range(features.NUM_SYMMETRIES):
        candidate = tuple(sorted(features.transform_sparse(indices, t)))
        if best is None or candidate < best:
            best = candidate
    return (stm, best)


def find_transform(source: list[int], target_sorted: tuple[int, ...]) -> int:
    """Which D4 transform maps `source` onto `target_sorted` (-1 if none)."""
    for t in range(features.NUM_SYMMETRIES):
        if tuple(sorted(features.transform_sparse(source, t))) == target_sorted:
            return t
    return -1


def validation_keys(reader: binpack.BinpackReader, game_map, val_games: set[int]):
    """canonical key -> record indices, restricted to validation games."""
    keys: dict[tuple, list[int]] = {}
    for index, game in enumerate(game_map):
        if game not in val_games:
            continue
        record = reader[index]
        key = canonical_position(record.feature_indices(), record.stm)
        keys.setdefault(key, []).append(index)
    return keys


def scan_for_leaks(reader: binpack.BinpackReader, game_map, val_keys: dict,
                   skip_games: set[int] | None = None):
    """D4-equivalent overlaps of THIS file's records with the frozen keys.

    ``skip_games`` excludes the validation games themselves when the scanned
    file is (or contains) the frozen source file -- a validation position
    matching itself is not a training leak. Only the small validation key set
    is held in memory; the scanned file is streamed, so 10M-record shards are
    fine.
    """
    skip = skip_games or frozenset()
    leaks = []
    for index, game in enumerate(game_map):
        if game in skip:
            continue
        record = reader[index]
        indices = record.feature_indices()
        key = canonical_position(indices, record.stm)
        if key in val_keys:
            val_index = val_keys[key][0]
            # val_index refers to the frozen SOURCE, not this training file.
            target = key[1]
            leaks.append({
                "train_record": index,
                "train_game": game,
                "val_record": val_index,
                "transform": find_transform(indices, target),
                "transform_target": "canonical D4 representative",
            })
    return leaks


def frozen_validation_record_ids(game_map, frozen: dict) -> list[int]:
    """Record indices of the frozen validation split, in source-file order."""
    val_games = set(frozen["split"]["validation_games"])
    return [i for i, game in enumerate(game_map) if game in val_games]


def frozen_train_record_ids(game_map, frozen: dict) -> list[int]:
    """Train-side records: drop validation games AND D4-leaking whole games.

    Shared by train.py's dataset construction and export.py's calibration
    sampling so both evaluate against the very same frozen identity.
    """
    val_games = set(frozen["split"]["validation_games"])
    excluded = set(frozen["leakage"]["excluded_train_games"])
    return [i for i, game in enumerate(game_map)
            if game not in val_games and game not in excluded]


def load_frozen(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise SystemExit(f"frozen validation schema {value.get('schema')!r} "
                         f"!= {SCHEMA}: regenerate with "
                         f"validation_freeze.py freeze")
    for section in ("data", "split", "leakage", "label_transform", "baseline"):
        if section not in value:
            raise SystemExit(f"frozen validation file misses section {section!r}")
    return value


def verify_frozen_data(frozen: dict, data_path: str | Path) -> dict:
    """The frozen split only applies to the exact bytes it was built from."""
    identity = file_identity(data_path)
    if identity["sha256"] != frozen["data"]["sha256"]:
        raise SystemExit(
            f"data file {data_path} (sha256 {identity['sha256']}) does not "
            f"match the frozen validation identity "
            f"({frozen['data']['path']}, sha256 {frozen['data']['sha256']}); "
            f"regenerate the freeze or pass the original file")
    return identity


def _tanh_target(score: float, score_clip: float, score_scale: float) -> float:
    return math.tanh(clip_score(float(score), score_clip) / score_scale)


def cmd_freeze(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    identity = file_identity(data_path)
    with binpack.BinpackReader(data_path) as reader:
        game_map = reader.game_index_map()
        n_games = game_map[-1] + 1 if len(game_map) else 0
        val_games = choose_validation_games(n_games, args.val_fraction, args.seed)
        val_keys = validation_keys(reader, game_map, val_games)
        leaks = scan_for_leaks(reader, game_map, val_keys,
                               skip_games=val_games)
        excluded_games = sorted({item["train_game"] for item in leaks})
        excluded_set = set(excluded_games)

        train_sum, train_count = 0.0, 0
        wdl_counts = [0, 0, 0]
        excluded_train_records = 0
        val_sum, val_sq, val_count = 0.0, 0.0, 0
        val_wdl_counts = [0, 0, 0]
        for index, game in enumerate(game_map):
            record = reader[index]
            is_val = game in val_games
            if is_val:
                target = _tanh_target(record.score, args.score_clip,
                                      args.score_scale)
                val_sum += target
                val_sq += target * target
                val_count += 1
                if record.wdl < 3:
                    val_wdl_counts[record.wdl] += 1
                continue
            if game in excluded_set:
                excluded_train_records += 1
                continue
            train_sum += _tanh_target(record.score, args.score_clip,
                                      args.score_scale)
            train_count += 1
            if record.wdl < 3:
                wdl_counts[record.wdl] += 1

    if val_count == 0 or train_count == 0:
        raise SystemExit("split produced an empty train or validation side")
    constant = train_sum / train_count
    constant_mse = val_sq / val_count - 2 * constant * val_sum / val_count \
        + constant * constant
    prior = [(count + 1) / (sum(wdl_counts) + 3) for count in wdl_counts]
    prior_ce = -sum(val_wdl_counts[w] * math.log(prior[w]) for w in range(3)) \
        / max(1, sum(val_wdl_counts))

    report = {
        "schema": SCHEMA,
        "data": identity,
        "split": {
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "games": n_games,
            "validation_games": sorted(val_games),
            "train_games_after_exclusion": sorted(
                set(range(n_games)) - val_games - excluded_set),
            "validation_records": val_count,
            "train_records_after_exclusion": train_count,
        },
        "leakage": {
            "rule": EXCLUSION_RULE,
            "d4_equivalent_positions": len(leaks),
            "excluded_train_games": excluded_games,
            "excluded_train_records": excluded_train_records,
            "details": leaks[:args.max_details],
            "details_truncated": len(leaks) > args.max_details,
        },
        "label_transform": {
            "score_clip": args.score_clip,
            "score_scale": args.score_scale,
            "wdl_temperature": args.wdl_temperature,
        },
        "baseline": {
            "fitted_on": "train split after leakage exclusion",
            "constant_prediction": constant,
            "constant_mse_on_validation": constant_mse,
            "class_prior": prior,
            "class_prior_ce_on_validation": prior_ce,
            "note": "hash equality and D4 position separation are separate "
                    "facts; both are required for split independence",
        },
    }
    write_json(Path(args.out), report)
    print(f"frozen validation: {val_count} records / {len(val_games)} games "
          f"from {identity['path']}")
    print(f"D4-equivalent training positions: {len(leaks)} "
          f"(excluding {excluded_train_records} records in "
          f"{len(excluded_games)} whole train games)")
    print(f"constant baseline: mse {constant_mse:.6f} | "
          f"class-prior CE {prior_ce:.6f}")
    print(f"written: {args.out}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    frozen = load_frozen(args.frozen)
    data_path = Path(args.data) if args.data else Path(frozen["data"]["path"])
    validation_identity = verify_frozen_data(frozen, data_path)
    with binpack.BinpackReader(data_path) as reader:
        game_map = reader.game_index_map()
        val_games = set(frozen["split"]["validation_games"])
        val_keys = validation_keys(reader, game_map, val_games)

    manifest = {
        "schema": SCHEMA,
        "frozen": {"path": str(Path(args.frozen).resolve()),
                   "sha256": file_identity(args.frozen)["sha256"]},
        "validation_data": validation_identity,
        "files": [],
    }
    total_leaks = 0
    for raw_path in args.train_data:
        path = Path(raw_path)
        identity = file_identity(path)
        # If this file IS the frozen source, its validation games are not
        # training material and must not be scanned as leaks.
        skip = (val_games if identity["sha256"] == validation_identity["sha256"]
                else frozenset())
        with binpack.BinpackReader(path) as reader:
            game_map = reader.game_index_map()
            n_games = game_map[-1] + 1 if len(game_map) else 0
            leaks = scan_for_leaks(reader, game_map, val_keys,
                                   skip_games=skip)
            excluded = sorted({item["train_game"] for item in leaks})
            excluded_records = sum(1 for game in game_map if game in set(excluded))
            # The frozen source file additionally loses its validation games;
            # fresh files never contain them by construction.
            drop = set(excluded) | (val_games if skip else set())
            manifest["files"].append({
                "path": str(path.resolve()),
                "games_total": n_games,
                "games": sorted(set(range(n_games)) - drop),
                "excluded_games": excluded,
                "validation_games_in_file": sorted(val_games) if skip else [],
                "excluded_records": excluded_records,
                "d4_equivalent_positions": len(leaks),
                "details": [
                    {**item, "val_record_index_in": str(validation_identity["path"])}
                    for item in leaks[:args.max_details]
                ],
            })
        total_leaks += len(leaks)
        print(f"{path}: {len(leaks)} D4-equivalent positions, "
              f"{excluded_records} records in {len(excluded)} games excluded")

    if args.manifest:
        write_json(Path(args.manifest), manifest)
        print(f"manifest: {args.manifest}")
    if args.out:
        write_json(Path(args.out), manifest)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="freeze the validation identity")
    freeze.add_argument("--data", required=True)
    freeze.add_argument("--val-fraction", type=float, default=0.1)
    freeze.add_argument("--seed", type=int, default=42)
    freeze.add_argument("--score-clip", type=float, default=0.0)
    freeze.add_argument("--score-scale", type=float, default=400.0)
    freeze.add_argument("--wdl-temperature", type=float, default=100.0)
    freeze.add_argument("--max-details", type=int, default=1000)
    freeze.add_argument("--out", required=True)
    freeze.set_defaults(func=cmd_freeze)

    check = sub.add_parser("check", help="scan extra training files for leaks")
    check.add_argument("--frozen", required=True)
    check.add_argument("--data", default=None,
                       help="frozen source file (default: path recorded in "
                            "the freeze)")
    check.add_argument("--train-data", action="append", required=True)
    check.add_argument("--max-details", type=int, default=1000)
    check.add_argument("--out", default=None)
    check.add_argument("--manifest", default=None,
                       help="also write a train manifest for train.py")
    check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
