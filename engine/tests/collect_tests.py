#!/usr/bin/env python3
"""Collection orchestration tests (ticket 05).

Uses the real executor for self-play games and a deterministic stub teacher
(the real Rapfi teacher is exercised by ticket 04's suite), so the test stays
fast and deterministic while still covering: random openings, sampling,
executor/VCF filtering, atomic game commits, pool stats and stop behaviour.

Usage: python collect_tests.py <gomoku_nnue.exe> <tools dir>
"""
from __future__ import annotations

import math
import shutil
import sys
import tempfile
import threading
from pathlib import Path

EXE = sys.argv[1] if len(sys.argv) > 1 else "build/engine/gomoku_nnue.exe"
TOOLS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tools")
sys.path.insert(0, str(TOOLS.resolve()))

from collect import CollectConfig, collect  # noqa: E402
from executor import ExecutorClient  # noqa: E402
from rapfi_teacher import TeacherReport, load_lock  # noqa: E402
from sample_pool import SamplePool  # noqa: E402

SIZE = 15
failures = 0


def check(cond, msg):
    global failures
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures += 1


class StubTeacher:
    """Deterministic score, so the collection chain is testable without Rapfi."""

    def __init__(self, identity):
        self.identity = identity
        self.calls = 0

    def label(self, cells, *, side, budget_ms=None):
        self.calls += 1
        stones = sum(1 for c in cells if c)
        score = ((sum(cells) * 37 + stones * 11) % 601) - 300
        return TeacherReport(status="ok", raw_eval=str(score), raw_score=score,
                             suggested_move=(7, 7), depth=8, elapsed_ms=1,
                             budget_ms=budget_ms or 50, threads=1,
                             message="", identity=self.identity)


def main():
    identity = load_lock()
    tmpdir = tempfile.mkdtemp(prefix="collect-tests-")
    try:
        return run(Path(tmpdir), identity)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run(tmp: Path, identity) -> int:
    pool_path = tmp / "pool.sqlite3"
    config = CollectConfig(games=2, depth=2, max_plies=24, opening_plies=2,
                           opening_region=7, sample_every=3, budget_ms=50,
                           vcf_nodes=5000, vcf_plies=4, seed=5,
                           model_id="test-handcrafted")
    teacher = StubTeacher(identity)
    with SamplePool(pool_path, capacity=60) as pool:
        pool.set_config("validation_games", 1)
        pool.set_config("dedup_d4", True)
        game_client = ExecutorClient(EXE)
        label_client = ExecutorClient(EXE)
        try:
            summary = collect(pool, config, game_client, label_client, teacher)
        finally:
            game_proc = game_client._proc
            label_proc = label_client._proc
            game_client.close()
            label_client.close()
        check(game_proc.poll() is not None,
              "collection closes its game executor process")
        check(label_proc.poll() is not None,
              "collection closes its labelling executor process")
        check(summary["generated"] > 0, "positions were generated")
        check(summary["inserted"] > 0, "accepted samples entered the pool")
        check(summary["generated"] >= summary["valid"] >= summary["inserted"],
              "generated >= valid >= deduped valid")
        check(summary["inserted"] == summary["pool_samples"],
              "pool size matches the deduped inserts")
        check(len(summary["validation_games"]) == 1,
              "validation membership froze at the first sample-bearing game")
        check(summary["staged_games"] == 0, "no staged games were left behind")
        check(summary["runs"] == 1 and summary["effective_per_second"] > 0,
              "the run recorded a positive effective rate")
        check(isinstance(summary["filters"], dict),
              "filter reasons are reported")
        check(summary["d4_augmentations"] == 0,
              "symmetry augmentation is not counted as raw data")
        check(summary["evicted_games"] == 0
              or summary["pool_samples"] <= summary["capacity"],
              "capacity is respected")
        check(pool.validate() == [], "the collected pool validates")

        train = pool.read_batch(split="train")
        validation = pool.read_batch(split="validation")
        check(len(train) + len(validation) == summary["pool_samples"],
              "readers see exactly the committed samples")
        sample = (train or validation)[0]
        check(sample["label_source"] == "teacher",
              "collection samples carry teacher supervision")
        check(isinstance(sample["raw_score"], int) and sample["value"] is not None,
              "teacher score and value are stored")
        check(abs(sample["value"] - math.tanh(sample["raw_score"] / 600.0)) < 1e-9,
              "value follows the recorded teacher contract")
        check(sample["source_kind"] == "selfplay",
              "game provenance is recorded as self-play")
        check(sample["model_id"] == "test-handcrafted",
              "collection model identity is recorded")
        check(sample["final_result"] in (None, "black_win", "white_win", "draw"),
              "final result is a real result or NULL")

        print("[stop during collection]")
        stop = threading.Event()
        second = CollectConfig(games=3, depth=2, max_plies=20, opening_plies=2,
                               sample_every=4, budget_ms=50, vcf_nodes=3000,
                               vcf_plies=3, seed=9)
        frozen = list(pool.stats()["validation_games"])

        def progress(update):
            if update.get("games_done", 0) >= 1:
                stop.set()

        game_client = ExecutorClient(EXE)
        label_client = ExecutorClient(EXE)
        try:
            stopped = collect(pool, second, game_client, label_client, teacher,
                              stop_event=stop, progress=progress)
        finally:
            game_proc = game_client._proc
            label_proc = label_client._proc
            game_client.close()
            label_client.close()
        check(stopped["stopped"] is True, "stop is reported to the caller")
        check(stopped["staged_games"] == 0, "stop leaves no staged game")
        check(pool.stats()["validation_games"] == frozen,
              "stopping does not change validation membership")
        check(game_proc.poll() is not None and label_proc.poll() is not None,
              "stop closes both executor processes")
        check(pool.validate() == [], "pool still validates after a stop")
        check(pool.stats()["runs"] == 2, "the stopped run is recorded")

    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
