#!/usr/bin/env python3
"""Sample-pool boundary tests (ticket 05).

No engine and no teacher are needed: the tests write synthetic records and
exercise the public pool boundary - staging/commit atomicity, exact and D4
dedup, capacity eviction, the frozen validation split, D4 leakage, concurrent
read snapshots, the supervision contract and integrity rejection.

Usage: python pool_tests.py [tools dir]
"""
from __future__ import annotations

import math
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

TOOLS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tools")
sys.path.insert(0, str(TOOLS.resolve()))

from sample_pool import (PoolError, SamplePool, _transform,  # noqa: E402
                         d4_key, exact_key)

SIZE = 15
failures = 0


def check(cond, msg):
    global failures
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures += 1


def cells_from(blacks, whites):
    cells = [0] * (SIZE * SIZE)
    for x, y in blacks:
        cells[y * SIZE + x] = 1
    for x, y in whites:
        cells[y * SIZE + x] = 2
    return cells


def position(k: int) -> list[int]:
    """A distinct, quiet, legal-parity board for index k."""
    blacks = [(1 + (k * 3) % 11, 1 + (k * 5) % 11)]
    whites = [(2 + (k * 2) % 10, 3 + (k * 4) % 9)]
    if blacks[0] == whites[0]:
        whites = [(whites[0][0] + 1, whites[0][1])]
    return cells_from(blacks, whites)


def record(cells, side, *, score: int = 120, label_source: str = "teacher",
           result_value: float | None = None) -> dict:
    return {
        "board": bytes(cells),
        "side_to_move": side,
        "rule": "renju",
        "tactical_status": "vcf_clear",
        "label_source": label_source,
        "label_version": "teacher-v1",
        "raw_eval": str(score) if label_source == "teacher" else None,
        "raw_score": score if label_source == "teacher" else None,
        "raw_perspective": "side_to_move",
        "normalization": "tanh",
        "normalization_param": 600.0,
        "value": (math.tanh(score / 600.0)
                  if label_source == "teacher" else None),
        "result_value": result_value,
        "suggested_move": None,
        "depth": 12,
        "budget_ms": 100,
        "threads": 1,
        "teacher_release": "250615",
        "teacher_version": "test",
        "teacher_exe_sha256": "a" * 64,
        "teacher_weights_sha256": "b" * 64,
        "teacher_config_sha256": "c" * 64,
    }


def commit_game(pool, source_id, positions, *, kind="selfplay",
                final_result=None):
    game = pool.begin_game(source_id, kind=kind, model_id="test", rule="renju")
    inserted = 0
    for cells, side, score in positions:
        status = pool.add_sample(game, record(cells, side, score=score))
        if status == "inserted":
            inserted += 1
    result = pool.commit_game(game, final_result=final_result)
    result["inserted"] = inserted
    return result


def main():
    tmpdir = tempfile.mkdtemp(prefix="pool-tests-")
    try:
        return run(Path(tmpdir))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run(tmp: Path) -> int:
    print("[staging / commit / abort]")
    path = tmp / "pool.sqlite3"
    with SamplePool(path, capacity=100) as pool:
        game = pool.begin_game("stage-1")
        pool.add_sample(game, record(position(1), "black"))
        check(pool.read_batch() == [],
              "staged samples are invisible to readers")
        result = pool.commit_game(game, final_result="black_win")
        check(result["split"] == "validation",
              "the first sample-bearing game freezes as validation")
        rows = pool.read_batch(split="validation")
        check(len(rows) == 1 and len(rows[0]["board"]) == SIZE * SIZE,
              "committed sample is readable in full")
        check(rows[0]["final_result"] == "black_win",
              "game result travels with the sample")
        aborted = pool.begin_game("stage-2")
        pool.add_sample(aborted, record(position(2), "black"))
        check(pool.abort_game(aborted) is True, "abort removes a staged game")
        check(len(pool.read_batch(split="validation")) == 1,
              "aborted samples never become readable")
        check(pool.stats()["staged_games"] == 0, "no staged games remain")
        check(pool.validate() == [], "pool validates after staging activity")

    print("[exact and D4 dedup]")
    dedup_path = tmp / "dedup.sqlite3"
    with SamplePool(dedup_path, capacity=100) as pool:
        cells = position(10)
        first = pool.begin_game("dup-a")
        check(pool.add_sample(first, record(cells, "black")) == "inserted",
              "first copy is inserted")
        second = pool.begin_game("dup-b")
        check(pool.add_sample(second, record(cells, "black")) == "duplicate",
              "exact duplicate is refused")
        rotated = list(_transform(cells, 1))
        check(pool.add_sample(second, record(rotated, "black")) == "d4_duplicate",
              "D4-equivalent board is refused as duplicate")
        pool.commit_game(first)
        pool.commit_game(second)
        stats = pool.stats()
        check(stats["duplicate"] == 1 and stats["d4_duplicate"] == 1,
              "duplicate counters are recorded")

    print("[capacity eviction and frozen validation]")
    cap_path = tmp / "capacity.sqlite3"
    with SamplePool(cap_path, capacity=6) as pool:
        pool.set_config("validation_games", 1)
        validation_game = commit_game(
            pool, "val-1", [(position(20 + i), "black", 100) for i in range(3)])
        check(validation_game["split"] == "validation",
              "first game becomes validation")
        frozen = list(pool.stats()["validation_games"])
        check(frozen == ["val-1"], "validation membership is recorded")
        commit_game(pool, "train-1", [(position(30 + i), "black", 100)
                                      for i in range(3)])
        commit_game(pool, "train-2", [(position(40 + i), "black", 100)
                                      for i in range(3)])
        counts = pool.counts()
        stats = pool.stats()
        check(counts["pool_samples"] <= 6, "pool stays within capacity")
        check(stats["evicted_games"] >= 1, "old training games are evicted")
        check(stats["validation_games"] == frozen,
              "eviction does not change validation membership")
        check(pool.counts()["validation_samples"] == 3,
              "validation samples are never evicted")
        commit_game(pool, "train-3", [(position(50 + i), "black", 100)
                                      for i in range(2)])
        check(pool.stats()["validation_games"] == frozen,
              "later appends do not change validation membership")
        check(pool.validate() == [], "evicted pool validates")
        train = pool.read_batch(split="train")
        check(all(row["board"] for row in train), "training samples remain whole")

    print("[D4 leakage into validation]")
    leak_path = tmp / "leak.sqlite3"
    with SamplePool(leak_path, capacity=100) as pool:
        pool.set_config("dedup_d4", False)
        pool.set_config("validation_games", 1)
        cells = position(60)
        commit_game(pool, "val-leak",
                    [(cells, "black", 100), (position(61), "black", 100)])
        rotated = list(_transform(cells, 3))
        commit_game(pool, "train-leak", [(rotated, "black", 100)])
        stats = pool.stats()
        check(stats["d4_leak"] >= 1,
              "D4-equivalent training sample is detected as a leak")
        rows = pool.read_batch(split="train")
        check(all(row["d4_key"] != d4_key(cells, "black") for row in rows),
              "the leaked training sample is removed")
        check(pool.counts()["validation_samples"] == 2,
              "validation membership and samples are untouched")
        check(pool.validate() == [], "leak-free pool validates")

    print("[concurrent reader snapshot]")
    snap_path = tmp / "snapshot.sqlite3"
    writer = SamplePool(snap_path, capacity=8)
    writer.set_config("validation_games", 1)
    commit_game(writer, "snap-val", [(position(70), "black", 100)])
    for i in range(3):
        commit_game(writer, f"snap-train-{i}",
                    [(position(80 + i), "black", 100),
                     (position(90 + i), "black", 100)])
    reader = SamplePool(snap_path, create=False)
    before = None
    with reader.read_snapshot() as snapshot:
        before = snapshot()
        commit_game(writer, "snap-new", [(position(100), "black", 100),
                                         (position(101), "black", 100),
                                         (position(102), "black", 100)])
        during = snapshot()
    check(len(before) == len(during),
          "reader snapshot is stable while another connection evicts/writes")
    check(len(before) > 0 and all(len(r["board"]) == 225 for r in before),
          "snapshot rows are complete")
    check(writer.stats()["evicted_games"] >= 1,
          "the concurrent writer did evict")
    reader.close()
    writer.close()

    print("[supervision contract and validation]")
    contract_path = tmp / "contract.sqlite3"
    with SamplePool(contract_path, capacity=100) as pool:
        game = pool.begin_game("selfplay-1", kind="selfplay")
        result_record = record(position(110), "black",
                               label_source="selfplay_result",
                               result_value=1.0)
        check(pool.add_sample(game, result_record) == "inserted",
              "self-play result supervision is accepted")
        bad = record(position(111), "black", label_source="teacher")
        bad["value"] = None
        try:
            pool.add_sample(game, bad)
            check(False, "teacher sample without value is rejected")
        except PoolError:
            check(True, "teacher sample without value is rejected")
        wrong_side = record(position(112), "white")
        try:
            pool.add_sample(game, wrong_side)
            check(False, "wrong side to move is rejected")
        except PoolError:
            check(True, "wrong side to move is rejected")
        wrong_rule = record(position(113), "black")
        wrong_rule["rule"] = "freestyle"
        try:
            pool.add_sample(game, wrong_rule)
            check(False, "freestyle rule is rejected")
        except PoolError:
            check(True, "freestyle rule is rejected")
        try:
            pool.commit_game(game)
            check(False, "unfinished game cannot commit result supervision")
        except PoolError:
            check(True, "unfinished game cannot commit result supervision")
        pool.commit_game(game, final_result="black_win")
        check(pool.counts()["validation_samples"] == 1,
              "committed self-play sample is readable")

    print("[mixing]")
    with SamplePool(tmp / "mix.sqlite3", capacity=100) as pool:
        for i in range(5):
            commit_game(pool, f"mix-{i}",
                        [(position(120 + i), "black", 100)])
        picked = pool.read_batch(split="train", limit=3, seed=7, new_ratio=0.5)
        check(len(picked) == 3, "batch size is honoured")
        check(len({row["id"] for row in picked}) == 3,
              "mixing never repeats a row")
        again = pool.read_batch(split="train", limit=3, seed=7, new_ratio=0.5)
        check([row["id"] for row in picked] == [row["id"] for row in again],
              "mixing is deterministic for a seed")

    print("[integrity rejection]")
    good = tmp / "integrity.sqlite3"
    with SamplePool(good, capacity=10) as pool:
        commit_game(pool, "i-1", [(position(130), "black", 100)])
    copy = tmp / "truncated.sqlite3"
    shutil.copyfile(good, copy)
    data = copy.read_bytes()
    copy.write_bytes(data[:80])
    try:
        SamplePool(copy, create=False)
        check(False, "truncated pool is rejected")
    except PoolError:
        check(True, "truncated pool is rejected")
    newer = tmp / "newer.sqlite3"
    shutil.copyfile(good, newer)
    conn = sqlite3.connect(newer)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    try:
        SamplePool(newer, create=False)
        check(False, "newer schema is rejected")
    except PoolError:
        check(True, "newer schema is rejected")
    not_a_pool = tmp / "other.sqlite3"
    conn = sqlite3.connect(not_a_pool)
    conn.execute("CREATE TABLE x (y INT)")
    conn.commit()
    conn.close()
    try:
        SamplePool(not_a_pool, create=False)
        check(False, "unrelated SQLite file is rejected")
    except PoolError:
        check(True, "unrelated SQLite file is rejected")

    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
