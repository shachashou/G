#!/usr/bin/env python3
"""Ticket 12: pause, restart and compatibility checks for the task scheduler."""
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("tools", "trainer", "engine/tests")]

from history_evaluation import evaluate as history_evaluate  # noqa: E402
from model_registry import ModelRegistry  # noqa: E402
from sample_pool import SamplePool  # noqa: E402
from training_task import TaskConfig, TrainingTask  # noqa: E402
from value_fixture import build_pool, make_board  # noqa: E402
from value_fixture import record as sample_record  # noqa: E402

EXE = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else str(
    ROOT / "build/engine/gomoku_nnue.exe")


class EmptySource:
    label_source = "teacher"

    def __init__(self, task):
        self.task = task

    def preflight(self):
        return {"label_source": "teacher", "identity": "no new commits"}

    def produce(self, pool, index):
        return {**pool.stats(), "inserted": 0, "source_ids": []}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="task-recovery-")
        self.root = Path(self.tmp.name)
        self.pool = self.root / "pool.sqlite3"
        build_pool(self.pool, validation_games=1)
        self.cfg = TaskConfig(
            engine=EXE, pool=str(self.pool), registry=str(self.root / "registry"),
            output_root=str(self.root / "tasks"),
            evaluations=str(self.root / "evaluations"), capacity=1000,
            validation_games=1, min_train_samples=1, min_validation_samples=1,
            min_new_samples=1, max_pending_samples=4,
            training_sample_limit=64, train_steps=400, batch_size=4,
            lr=0.0001, device="cpu", max_rounds=1, max_games=1,
            max_minutes=5, max_idle_s=10, min_round_interval_s=0,
            collection_delay_ms=10,
            evaluation={"pairs": 1, "movetime_ms": 10, "vcf_policy": "off"})

    def tearDown(self):
        self.tmp.cleanup()

    def wait(self, predicate, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def start_paused_task(self, config=None):
        task = TrainingTask(config or self.cfg, source_factory=EmptySource)
        result = []
        thread = threading.Thread(target=lambda: result.append(task.run()),
                                  daemon=True)
        thread.start()
        self.assertTrue(self.wait(
            lambda: task.snapshot()["status"] == "running"
            and task.snapshot().get("trainer", {}).get("progress", {}).get(
                "step", 0) >= 2), task.snapshot())
        self.assertTrue(task.pause())
        self.assertTrue(self.wait(
            lambda: task.snapshot()["status"] == "paused", 60),
            task.snapshot())
        return task, thread, result

    def test_pause_at_checkpoint_and_resume_in_process(self):
        task, thread, result = self.start_paused_task()
        state = task.snapshot()
        directory = Path(state["directory"])
        record = json.loads((directory / "round-00000.json").read_text("utf-8"))
        self.assertEqual(record["status"], "paused")
        self.assertTrue((directory / "candidates/round-00000/last.pt").is_file())
        self.assertTrue(record["bundle"] and Path(record["bundle"]).is_file())
        self.assertTrue(record["training"]["stopped"])
        self.assertEqual(record["training"]["stop_reason"], "paused")
        frozen_identity = record["data_identity"]

        self.assertTrue(task.resume())
        thread.join(180)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0]["status"], "completed", result[0]["reason"])
        self.assertEqual(result[0]["rounds"][0]["training"]["steps"],
                         self.cfg.train_steps)
        self.assertEqual(result[0]["rounds"][0]["data_identity"],
                         frozen_identity)
        self.assertEqual(result[0]["rounds"][0]["status"], "published")
        self.assertTrue(result[0]["cleanup_complete"])
        self.assertIsNotNone(ModelRegistry(self.cfg.registry).resolve())
        with SamplePool(self.cfg.pool, create=False) as pool:
            self.assertEqual(pool.stats()["staged_games"], 0)
            self.assertFalse(pool.validate())

    def test_restart_resume_reclaims_lease_and_staging(self):
        task, thread, _ = self.start_paused_task()
        directory = Path(task.snapshot()["directory"])
        task.stop()
        thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertEqual(task.snapshot()["status"], "stopped")

        # Simulate a process that died while a game was staged and while it
        # still held the pool lease.
        lease = Path(str(self.pool) + ".training-task.lock")
        lease.write_text(json.dumps({"task_id": task.task_id, "pid": 999999}),
                         encoding="utf-8")
        with SamplePool(self.pool) as pool:
            staged = pool.begin_game("stale-staged-game")
            cells, side = make_board(random.Random(7), 6)
            pool.add_sample(staged, sample_record(cells, side))
            self.assertEqual(pool.stats()["staged_games"], 1)

        with self.assertRaises(ValueError):
            TrainingTask.resume_from_directory(
                directory, source_factory=EmptySource,
                config=replace(self.cfg, train_steps=self.cfg.train_steps + 10))

        resumed = TrainingTask.resume_from_directory(
            directory, source_factory=EmptySource)
        snapshot = resumed.run()
        self.assertEqual(snapshot["status"], "completed", snapshot["reason"])
        self.assertTrue(snapshot["cleanup_complete"])
        self.assertFalse(lease.exists())
        with SamplePool(self.cfg.pool, create=False) as pool:
            self.assertEqual(pool.stats()["staged_games"], 0)
            self.assertFalse(pool.validate())
        events = [json.loads(line)["event"] for line in
                  (directory / "events.jsonl").read_text("utf-8").splitlines()]
        self.assertIn("stale_lease_reclaimed", events)
        self.assertIn("stale_staging_cleaned", events)
        self.assertEqual(len(ModelRegistry(self.cfg.registry).read()["versions"]), 1)

    def test_resume_keeps_frozen_validation_and_data_identity(self):
        task, thread, _ = self.start_paused_task()
        directory = Path(task.snapshot()["directory"])
        record = json.loads((directory / "round-00000.json").read_text("utf-8"))
        frozen = record["data_identity"]
        task.stop()
        thread.join(30)
        self.assertFalse(thread.is_alive())

        # New committed data and a changed pool must not alter the resumed
        # round's frozen bundle or validation split.
        with SamplePool(self.pool) as pool:
            game = pool.begin_game("post-pause-commit")
            cells, side = make_board(random.Random(55), 8)
            pool.add_sample(game, sample_record(cells, side))
            pool.commit_game(game)

        resumed = TrainingTask.resume_from_directory(
            directory, source_factory=EmptySource)
        snapshot = resumed.run()
        self.assertEqual(snapshot["status"], "completed", snapshot["reason"])
        round_record = json.loads(
            (directory / "round-00000.json").read_text("utf-8"))
        self.assertEqual(round_record["data_identity"], frozen)
        self.assertTrue(Path(round_record["bundle"]).is_file())

    def test_pause_during_evaluation_reuses_frozen_run(self):
        pause_once = {"done": False}
        holder = {}

        def pausing_evaluator(candidate, engine, **kwargs):
            if not pause_once["done"]:
                original_progress = kwargs.get("progress")

                def progress(item):
                    if original_progress is not None:
                        original_progress(item)
                    if not pause_once["done"]:
                        pause_once["done"] = True
                        holder["task"].pause()
                kwargs["progress"] = progress
            return history_evaluate(candidate, engine, **kwargs)

        task = TrainingTask(self.cfg, source_factory=EmptySource,
                            evaluator=pausing_evaluator)
        holder["task"] = task
        result = []
        thread = threading.Thread(target=lambda: result.append(task.run()),
                                  daemon=True)
        thread.start()
        self.assertTrue(self.wait(
            lambda: task.snapshot()["status"] == "paused"
            and task.snapshot().get("evaluation", {}).get("phase") == "evaluating",
            120), task.snapshot())
        state = task.snapshot()
        round_record = json.loads(
            (Path(state["directory"]) / "round-00000.json").read_text("utf-8"))
        interrupted_run = Path(round_record["evaluation"]["run"])
        self.assertEqual(round_record["evaluation"]["status"], "interrupted")
        self.assertTrue((interrupted_run / "manifest.json").is_file())

        self.assertTrue(task.resume())
        thread.join(180)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0]["status"], "completed", result[0]["reason"])
        final_round = json.loads(
            (Path(state["directory"]) / "round-00000.json").read_text("utf-8"))
        self.assertEqual(Path(final_round["evaluation"]["run"]), interrupted_run)
        self.assertEqual(final_round["evaluation"]["status"], "completed")
        self.assertTrue(final_round["evaluation"]["publication"])
        self.assertEqual(len(ModelRegistry(self.cfg.registry).read()["versions"]), 1)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
