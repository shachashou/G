#!/usr/bin/env python3
"""Ticket 09: scheduling, real training/evaluation, backpressure and cleanup."""
from dataclasses import replace
import json
import os
import random
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("tools", "trainer", "engine/tests")]
from training_task import TaskConfig, TrainingTask, TeacherDataSource
from sample_pool import SamplePool, PoolError
from value_fixture import build_pool, make_board, record
from value_train import TrainConfig, train_value
from collect_tests import StubTeacher
from executor import ExecutorClient
from model_registry import ModelRegistry
import training_task

EXE = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else str(ROOT / "build/engine/gomoku_nnue.exe")


class ManagedTeacher(StubTeacher):
    def __enter__(self):
        return self

    def close(self):
        pass


class DeterministicSource(TeacherDataSource):
    def preflight(self):
        return {"label_source": "teacher", "identity": "controlled test labels"}

    def teacher(self):
        return ManagedTeacher(self.identity)


class EmptySource:
    label_source = "teacher"

    def __init__(self, task):
        self.task = task

    def preflight(self):
        return {"label_source": "teacher", "identity": "no new commits"}

    def produce(self, pool, index):
        return {**pool.stats(), "inserted": 0, "source_ids": []}


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="teacher-task-")
        self.root = Path(self.tmp.name)
        self.cfg = TaskConfig(engine=EXE, pool=str(self.root / "pool.sqlite3"),
            registry=str(self.root / "registry"), output_root=str(self.root / "tasks"),
            evaluations=str(self.root / "evaluations"), capacity=120, validation_games=1,
            min_train_samples=1, min_validation_samples=1, min_new_samples=1,
            max_pending_samples=2, training_sample_limit=100, train_steps=4, batch_size=4,
            device="cpu", max_rounds=2, max_games=30, max_minutes=2, max_idle_s=10,
            min_round_interval_s=0, collection_delay_ms=10,
            collection={"depth":2,"max_plies":24,"opening_plies":2,"opening_region":7,
                "sample_every":3,"budget_ms":50,"vcf_nodes":1000,"vcf_plies":4},
            evaluation={"pairs":1,"movetime_ms":10,"vcf_policy":"off"})

    def tearDown(self):
        self.tmp.cleanup()

    def task(self, config=None, **kw):
        return TrainingTask(config or self.cfg, source_factory=DeterministicSource, **kw)

    def events(self, result):
        return [json.loads(s) for s in (Path(result["directory"]) / "events.jsonl").read_text().splitlines()]

    def test_parallel_rounds_and_model_boundaries(self):
        observed = []
        def trainer(config, *, data_bundle, **kw):
            seq = data_bundle.identity["commit_sequence"]
            cells = tuple(s.cells for s in data_bundle.train)
            # Data can commit while this immutable training bundle is held.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with SamplePool(self.cfg.pool, create=False) as pool:
                    new_seq = pool.stats()["commit_sequence"]
                if new_seq > seq:
                    break
                time.sleep(.02)
            observed.append(new_seq > seq)
            self.assertEqual(cells, tuple(s.cells for s in data_bundle.train))
            return train_value(config, data_bundle=data_bundle, **kw)
        task = self.task(replace(self.cfg, collector_workers=3), trainer=trainer)
        result = task.run()
        self.assertEqual(result["status"], "completed", result["reason"])
        self.assertEqual(len(result["rounds"]), 2)
        self.assertTrue(any(observed))
        self.assertTrue(result["cleanup_complete"])
        first, second = result["rounds"]
        self.assertEqual(first["evaluation"]["publication"]["version"], "v0")
        self.assertIsNone(second["evaluation"]["publication"])
        self.assertEqual(second["evaluation"]["statistics"]["v0"]["conclusion"], "insufficient_pairs")
        self.assertGreater(second["data_identity"]["committed_training_samples"], first["data_identity"]["committed_training_samples"])
        events = self.events(result)
        snapshots = [e for e in events if e["event"] == "game_snapshot"]
        self.assertEqual(snapshots[0]["model_id"], "handcrafted")
        self.assertTrue(any(e["version"] == "v0" for e in snapshots))
        self.assertTrue(any(e.get("phase") == "backpressure" for e in events))
        self.assertTrue(any(e["event"] == "data_commit" and e["source_ids"] for e in events))
        with SamplePool(self.cfg.pool, create=False) as pool:
            self.assertFalse(pool.validate())
            self.assertEqual(pool.stats()["staged_games"], 0)
            self.assertLessEqual(pool.stats()["pool_samples"], self.cfg.capacity)
            self.assertTrue(all(r["label_source"] == "teacher" for r in pool.read_batch()))
        model = ModelRegistry(self.cfg.registry).resolve()
        self.assertEqual(model["version"], "v0")

    def test_concurrent_collectors_share_global_game_budget(self):
        barrier = threading.Barrier(3)
        seen = []
        seen_lock = threading.Lock()

        class ConcurrentSource(EmptySource):
            def produce(self, pool, index):
                with seen_lock:
                    seen.append(index)
                if index < 3:
                    barrier.wait(timeout=10)
                game = pool.begin_game(f"parallel-{index}")
                cells, side = make_board(random.Random(200 + index), 8)
                pool.add_sample(game, record(cells, side))
                pool.commit_game(game)
                return {**pool.stats(), "inserted": 1, "source_ids": [f"parallel-{index}"]}

        cfg = replace(self.cfg, collector_workers=3, max_games=7,
                      min_train_samples=100, max_pending_samples=100)
        result = TrainingTask(cfg, source_factory=ConcurrentSource).run()
        self.assertEqual(result["status"], "completed", result["reason"])
        self.assertEqual(sorted(seen), list(range(7)))
        self.assertEqual(result["games"], 7)
        self.assertEqual(result["resources"]["collector_workers"], 3)
        self.assertTrue(result["cleanup_complete"])
        with SamplePool(cfg.pool, create=False) as pool:
            self.assertEqual(pool.stats()["committed_games"], 7)
            self.assertEqual(pool.stats()["staged_games"], 0)
            self.assertFalse(pool.validate())

    def test_progress_writes_are_coalesced_and_terminal_is_immediate(self):
        task = self.task(replace(self.cfg, state_write_interval_s=3600))
        task.started = time.monotonic()
        task.directory.mkdir(parents=True)
        with patch.object(training_task, 'write_json', wraps=training_task.write_json) as writes:
            for step in range(100):
                task.update('trainer', phase='training', progress={'step':step})
            self.assertEqual(writes.call_count, 1)
            task.update(status='stopped', reason='user_stop')
            self.assertEqual(writes.call_count, 2)
        stored = json.loads((task.directory/'state.json').read_text('utf-8'))
        self.assertEqual(stored['trainer']['progress']['step'], 99)
        self.assertEqual(stored['status'], 'stopped')

    def test_checkpoint_rate_limit_preserves_best_and_final_steps(self):
        import torch
        build_pool(self.cfg.pool)
        with patch.object(torch, 'save', wraps=torch.save) as saves:
            result = train_value(TrainConfig(pool=self.cfg.pool,
                output_dir=str(self.root/'limited-writes'), steps=8, batch_size=8,
                val_every=1, checkpoint_every=1, checkpoint_min_interval_s=3600,
                log_every=1, device='cpu', export=False))
            self.assertEqual(saves.call_count, 2)  # Best and last, only at completion.
        last = torch.load(result['checkpoint']['last'], weights_only=False)
        best = torch.load(result['checkpoint']['best'], weights_only=False)
        history = [json.loads(line) for line in Path(result['history']).read_text().splitlines()]
        lowest = min((h for h in history if 'val_loss' in h), key=lambda h:h['val_loss'])
        self.assertEqual(best['step'], lowest['step'])
        self.assertEqual(last['step'], 8)

    def test_batch_staging_uses_one_transaction_and_rolls_back_errors(self):
        with SamplePool(self.cfg.pool) as pool:
            game = pool.begin_game('batch-staging')
            records = [record(*make_board(random.Random(i + 400), 8)) for i in range(8)]
            statements = []
            pool._conn.set_trace_callback(statements.append)
            self.assertEqual(pool.add_samples(game, records), ['inserted'] * 8)
            self.assertEqual(sum(s == 'COMMIT' for s in statements), 1)
            self.assertFalse(pool.read_batch())  # Staged is still invisible.
            fresh = record(*make_board(random.Random(900), 10))
            with self.assertRaises(PoolError):
                pool.add_samples(game, [fresh, {**fresh, 'board':b'bad'}])
            count = pool._conn.execute('SELECT COUNT(*) FROM samples WHERE game_id=?', (game,)).fetchone()[0]
            self.assertEqual(count, 8)
            pool.abort_game(game)

    def test_cached_batches_preserve_rng_d4_targets_and_weights(self):
        import torch
        from value_data import load_pool, tensors, CachedTrainingBatches
        build_pool(self.cfg.pool)
        samples = load_pool(self.cfg.pool).train
        samples = [replace(s, weight=.25 if i % 2 else 1.) for i, s in enumerate(samples)]
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            cached = CachedTrainingBatches(samples, device)
            reference_rng, cached_rng = random.Random(91), random.Random(91)
            for batch_size in (1, 8, len(samples) + 1):
                chosen = [samples[reference_rng.randrange(len(samples))]
                          for _ in range(min(batch_size, len(samples)))]
                x, y = tensors(chosen, augment=True, rng=reference_rng)
                actual_x, actual_y, actual_weights = cached.sample(batch_size, cached_rng)
                self.assertTrue(torch.equal(x, actual_x.cpu()))
                self.assertTrue(torch.equal(y, actual_y.cpu()))
                self.assertTrue(torch.equal(torch.tensor([s.weight for s in chosen]), actual_weights.cpu()))
                self.assertEqual(reference_rng.getstate(), cached_rng.getstate())

    def test_no_unbounded_training_without_new_commits(self):
        build_pool(self.cfg.pool)
        config = replace(self.cfg, capacity=1000, max_rounds=5, max_games=2)
        task = TrainingTask(config, source_factory=EmptySource)
        result = task.run()
        self.assertEqual(result["status"], "completed", result["reason"])
        self.assertEqual(len(result["rounds"]), 1)
        self.assertEqual(result["current"]["version"], "v0")
        # A fresh task can warm-start from the validated deployment model.
        second = TrainingTask(replace(config, max_rounds=1), source_factory=EmptySource).run()
        self.assertEqual(second["status"], "completed", second["reason"])
        self.assertEqual(second["current"]["sha256"], result["current"]["sha256"])
        self.assertEqual(second["rounds"][0]["lineage"]["parent"]["version"], "v0")
        self.assertIsNone(second["rounds"][0]["evaluation"]["publication"])

    def test_numerical_failure_never_calls_evaluator(self):
        build_pool(self.cfg.pool)
        def rejected(config, **kw):
            result = train_value(config, **kw)
            result["export"]["passed"] = False
            return result
        def forbidden(*args, **kwargs):
            self.fail("numerical rejection must not enter evaluation")
        task = TrainingTask(replace(self.cfg, capacity=1000, max_rounds=1, max_games=1),
            source_factory=EmptySource, trainer=rejected, evaluator=forbidden)
        result = task.run()
        self.assertEqual(result["status"], "completed", result["reason"])
        self.assertEqual(result["trainer"]["phase"], "numerical_rejected")
        self.assertIsNone(result["current"])
        self.assertIsNone(result["rounds"][0]["evaluation"])

    def test_validated_promotion_flows_back_to_task(self):
        build_pool(self.cfg.pool)
        config = replace(self.cfg, capacity=1000, max_rounds=1, max_games=1)
        first = TrainingTask(config, source_factory=EmptySource).run()
        self.assertEqual(first["current"]["version"], "v0")
        # Same public scheduler and evaluator, but a controlled transport
        # rival makes the positive statistical branch reproducible. The real
        # C++ referee and publication replay still validate every full game.
        import history_evaluation as evaluation
        from history_evaluation_tests import ScriptedEngine
        ScriptedEngine.mode = "win"
        with patch.object(evaluation, "MatchEngine", ScriptedEngine):
            second = TrainingTask(replace(config, evaluation={"pairs":32,"movetime_ms":10,"vcf_policy":"off"}),
                                  source_factory=EmptySource).run()
        self.assertEqual(second["status"], "completed", second["reason"])
        self.assertEqual(second["current"]["version"], "v1")
        self.assertEqual(second["evaluation"]["phase"], "published")
        self.assertEqual(second["rounds"][0]["evaluation"]["publication"]["previous"], "v0")

    def test_stop_reaps_owned_processes_and_discards_staging(self):
        processes = []
        def client(*args, **kw):
            value = ExecutorClient(*args, **kw)
            processes.append(value._proc)
            return value
        task = self.task(replace(self.cfg, collector_workers=4, min_train_samples=100, max_pending_samples=100))
        result = []
        with patch.object(training_task, "ExecutorClient", client):
            thread = threading.Thread(target=lambda: result.append(task.run()))
            thread.start()
            deadline = time.monotonic() + 10
            while not processes and time.monotonic() < deadline:
                time.sleep(.01)
            task.stop()
            thread.join(15)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0]["status"], "stopped", result[0]["reason"])
        self.assertTrue(result[0]["cleanup_complete"])
        self.assertTrue(processes and all(p.poll() is not None for p in processes))
        with SamplePool(self.cfg.pool, create=False) as pool:
            self.assertEqual(pool.stats()["staged_games"], 0)
            self.assertFalse(pool.validate())
        self.assertFalse(Path(self.cfg.pool + ".training-task.lock").exists())

    def test_dependency_failure_and_pool_lease(self):
        class Missing(EmptySource):
            def preflight(self):
                raise ValueError("teacher unavailable: missing pinned engine")
        failed = TrainingTask(self.cfg, source_factory=Missing).run()
        self.assertEqual(failed["status"], "failed")
        self.assertIn("teacher unavailable", failed["reason"])
        lease = Path(self.cfg.pool + ".training-task.lock")
        lease.write_text('{"task_id":"other"}')
        failed = self.task().run()
        self.assertEqual(failed["status"], "failed")
        self.assertIn("owns this pool", failed["reason"])
        self.assertTrue(lease.exists())
        self.assertIsNone(ModelRegistry(self.cfg.registry).resolve())

    def test_time_limit_and_config_rejection(self):
        config = replace(self.cfg, max_minutes=0.003, max_games=100,
                         min_train_samples=100, max_pending_samples=100)
        result = TrainingTask(config, source_factory=EmptySource).run()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["reason"], "time_limit")
        with self.assertRaises(ValueError):
            replace(self.cfg, capacity=1).validate()

    def test_stop_checkpoint_counts_completed_optimizer_steps(self):
        build_pool(self.cfg.pool)
        stop = threading.Event()
        result = train_value(TrainConfig(pool=self.cfg.pool,
            output_dir=str(self.root / "stopped-training"), steps=10, batch_size=8,
            log_every=1, device="cpu"), stop_event=stop,
            progress=lambda p: stop.set() if p.get("step") == 1 else None)
        self.assertTrue(result["stopped"])
        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["samples_exposed"], 8)

    def test_small_pool_records_actual_sample_exposure(self):
        from value_data import load_pool
        build_pool(self.cfg.pool)
        bundle = load_pool(self.cfg.pool)
        bundle = replace(bundle, train=bundle.train[:1])
        result = train_value(TrainConfig(pool=self.cfg.pool,
            output_dir=str(self.root / "tiny-training"), steps=3, batch_size=8, device="cpu"),
            data_bundle=bundle)
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["samples_exposed"], 3)

    def test_recent_mix_uses_insertion_order_not_source_name(self):
        with SamplePool(self.cfg.pool) as pool:
            pool.set_config("validation_games", 0)
            for index, source in enumerate(("z-old", "y-old", "x-old", "w-old", "a-new")):
                game = pool.begin_game(source)
                board, side = make_board(random.Random(100 + index), 6)
                pool.add_sample(game, record(board, side))
                pool.commit_game(game)
            rows = pool.read_batch()
            newest = pool.read_batch(limit=1, seed=3, new_ratio=1)
            self.assertEqual(newest[0]["id"], rows[-1]["id"])

    @unittest.skipUnless(os.name == "nt", "Windows sharing semantics")
    def test_state_reader_during_training_does_not_fail_task(self):
        from run_support_tests import windows_reader
        build_pool(self.cfg.pool)
        locked = False
        def trainer(config, *, progress, **kw):
            def notify(value):
                nonlocal locked
                if not locked:
                    locked = True
                    close = windows_reader(task.directory / "state.json")
                    reader = threading.Timer(.15, close)
                    reader.start()
                    try:
                        progress(value)
                    finally:
                        reader.join()
                else:
                    progress(value)
            return train_value(config, progress=notify, **kw)
        task = TrainingTask(replace(self.cfg, capacity=1000, max_rounds=1, max_games=1),
                            source_factory=EmptySource, trainer=trainer)
        result = task.run()
        self.assertTrue(locked)
        self.assertEqual(result["status"], "completed", result["reason"])
        self.assertTrue(result["cleanup_complete"])
        self.assertEqual(result["current"]["version"], "v0")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
