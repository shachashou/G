#!/usr/bin/env python3
"""Public evaluation/publication boundary: real engines plus scripted rivals."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "trainer"), str(ROOT / "engine/tests")]
import history_evaluation as he
from model_registry import ModelRegistry, RegistryError
from run_support import file_identity, write_json
from value_fixture import build_pool
from value_train import TrainConfig, train_value
import value_format as vf

EXE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "build/engine/gomoku_nnue.exe"


class ScriptedEngine:
    """Controlled responses at the same transport boundary as real rivals.

    The production C++ referee still validates every move and every release
    transcript. Passing these matches is a branch test, not strength evidence.
    """
    mode = "win"
    budgets = []

    def __init__(self, spec, timeout, stop):
        self.spec, self.stop = spec, stop
        self.candidate = "candidate" in Path(spec.argv[2]).parts
        _, self.meta = vf.read_value_weights(spec.argv[2])
        self.lines = []
        self.budgets.append((timeout, tuple(spec.argv[3:])))

    def start(self, rule, timeout):
        assert rule == 2 and timeout > 0

    def send(self, line):
        self.lines.append(line)

    def recv_line(self):
        ident = "wrong" if self.mode == "identity" else self.meta["model_id"]
        return (f'eval="nnue", eval_engine="gmvalue1-scalar-full", value_model_id="{ident}", '
                'value_model_arch="450-1024-64-32-1", value_model_rule="renju", '
                'value_model_feature_spec="absolute-color-450", '
                'value_model_contract="black-absolute-tanh-v1"')

    def move(self, cells):
        if self.mode in ("timeout", "crash"):
            raise he.EngineFailure(self.mode, "controlled transport failure")
        if self.mode == "illegal":
            return 99, 99
        if self.mode == "foul":
            return 7, 7
        black = cells.count(1) == cells.count(2)
        win = self.candidate == (self.mode != "lose")
        if win:
            choices = [(x, 0 if black else 14) for x in range(5)]
        else:
            choices = [(x, y) for y in ((2, 1) if black else (12, 13))
                       for x in (0, 3, 6, 9, 12)]
        return next((x, y) for x, y in choices if cells[y * 15 + x] == 0)

    def close(self):
        pass


class HistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="history-evaluation-")
        cls.root = Path(cls.temp.name)
        pool = build_pool(cls.root / "pool.sqlite3")
        cls.models = []
        for seed in (11, 12, 13):
            out = cls.root / f"model-{seed}"
            result = train_value(TrainConfig(pool=str(pool), output_dir=str(out),
                seed=seed, steps=8, batch_size=8, device="cpu", val_every=4))
            assert result["export"]["passed"]
            cls.models.append(out / "model.gmvalue")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.work = self.root / self._testMethodName
        self.work.mkdir()
        self.registry = self.work / "registry"
        self.output = self.work / "evaluations"
        self.cfg = he.EvaluationConfig(pairs=1, movetime_ms=10, vcf_policy="off", seed=42)
        ScriptedEngine.mode = "win"
        ScriptedEngine.budgets = []

    def run_eval(self, model=0, bootstrap=False, config=None, **kw):
        return he.evaluate(self.models[model], EXE, registry=self.registry,
            output_root=self.output, bootstrap=bootstrap, config=config or self.cfg, **kw)

    def initial(self):
        with patch.object(he, "MatchEngine", ScriptedEngine):
            report = self.run_eval(bootstrap=True)
        self.assertEqual(report["publication"]["version"], "v0", report.get("publication_error"))
        return report

    def pointer(self):
        return (self.registry / "state.json").read_bytes()

    def test_real_initial_and_small_challenge(self):
        initial = self.run_eval(bootstrap=True)
        self.assertIsNotNone(initial["publication"], initial.get("publication_error"))
        self.assertTrue(initial["initial_registration_only"])
        games = initial["matches"]["initial"]
        self.assertTrue(all(g["valid"] and g["terminal"] != "ongoing" for g in games))
        self.assertEqual(games[0]["opening"], games[1]["opening"])
        before = self.pointer()
        challenge = self.run_eval(model=1)
        self.assertEqual(challenge["statistics"]["v0"]["conclusion"], "insufficient_pairs")
        self.assertIsNone(challenge["publication"])
        self.assertEqual(before, self.pointer())

    def test_pass_failure_and_historical_challenge(self):
        self.initial()
        config = replace(self.cfg, pairs=32)
        with patch.object(he, "MatchEngine", ScriptedEngine):
            passed = self.run_eval(model=1, config=config)
        self.assertEqual(passed["publication"]["version"], "v1", passed.get("publication_error"))
        self.assertEqual(passed["statistics"]["v0"]["wins"], 64)
        self.assertGreater(passed["statistics"]["v0"]["one_sided_lower"], 0.5)
        self.assertEqual(len({budget for budget in ScriptedEngine.budgets}), 1)
        self.assertEqual(ModelRegistry(self.registry).resolve("v0")["version"], "v0")
        # The same winning run cannot race or replace a later incumbent.
        with self.assertRaises(RegistryError):
            ModelRegistry(self.registry).publish(passed["run"], expected_current="v0",
                expected_manifest_sha=passed["manifest_sha256"], expected_report_sha=passed["report_sha256"])
        # A third candidate is unnecessary: challenge v0 against current v1,
        # with losses scripted through real authoritative adjudication.
        ScriptedEngine.mode = "lose"
        before = self.pointer()
        with patch.object(he, "MatchEngine", ScriptedEngine):
            failed = self.run_eval(model=0, config=config, opponents=["v1"])
        self.assertFalse(failed["eligible"])
        self.assertEqual(failed["statistics"]["v1"]["conclusion"], "below_threshold")
        self.assertEqual(before, self.pointer())

    def test_anomalies_never_promote(self):
        self.initial()
        before = self.pointer()
        for mode in ("illegal", "timeout", "crash", "identity"):
            ScriptedEngine.mode = mode
            with patch.object(he, "MatchEngine", ScriptedEngine):
                result = self.run_eval(model=1)
            self.assertEqual(result["status"], "aborted")
            self.assertFalse(result["eligible"])
            game = result["matches"]["v0"][0]
            self.assertFalse(game["valid"])
            self.assertTrue(game["anomalies"])
            self.assertEqual(result["statistics"]["v0"]["valid_games"], 0)
            self.assertEqual(before, self.pointer())

    def test_multiple_history_opponents_and_reuse_control(self):
        initial = self.initial()
        config = replace(self.cfg, pairs=32)
        with patch.object(he, "MatchEngine", ScriptedEngine):
            promoted = self.run_eval(model=1, config=config)
            compared = self.run_eval(model=2, config=config, opponents=["v0"])
        self.assertEqual(compared["publication"]["version"], "v2")
        frozen = he.read_json(Path(compared["run"]) / "manifest.json")
        self.assertEqual(set(frozen["opponents"]), {"v0", "v1"})
        for stats in compared["statistics"].values():
            self.assertEqual(stats["comparisons"], 2)
            self.assertTrue(stats["passed"])
        old = he.read_json(Path(promoted["run"]) / "manifest.json")
        self.assertTrue(set(old["opening_d4_keys"]).isdisjoint(frozen["opening_d4_keys"]))
        self.assertTrue(set(old["opening_d4_keys"]).issubset(frozen["prior_evaluation_d4_keys"]))
        original = ModelRegistry(self.registry).resolve("v0")
        saved = Path(original["path"]).read_bytes()
        try:
            Path(original["path"]).write_bytes(b"bad")
            with self.assertRaises(RegistryError):
                ModelRegistry(self.registry).resolve("v0")
        finally:
            Path(original["path"]).write_bytes(saved)

    def test_truncation_and_cancellation_keep_current(self):
        self.initial()
        before = self.pointer()
        with patch.object(he, "MatchEngine", ScriptedEngine):
            truncated = self.run_eval(model=1, config=replace(self.cfg, max_plies=1))
        game = truncated["matches"]["v0"][0]
        self.assertIsNone(game["result"])
        self.assertEqual(game["anomalies"][0]["kind"], "truncated")
        stop = threading.Event()
        with patch.object(he, "MatchEngine", ScriptedEngine):
            interrupted = self.run_eval(model=1, stop_event=stop,
                                       progress=lambda _: stop.set())
        self.assertEqual(interrupted["status"], "interrupted")
        self.assertEqual(interrupted["statistics"]["v0"]["completed_pairs"], 0)
        self.assertEqual(before, self.pointer())

    def test_freezing_and_tamper_rejection(self):
        self.initial()
        before = self.pointer()
        def mutate(update):
            path = Path(update["run"]) / "manifest.json"
            manifest = he.read_json(path)
            manifest["config"]["score_threshold"] = 0.99
            write_json(path, manifest)
        with patch.object(he, "MatchEngine", ScriptedEngine):
            result = self.run_eval(model=1, progress=mutate)
        self.assertEqual(result["status"], "identity-mismatch")
        self.assertEqual(before, self.pointer())
        run = Path(result["run"])
        with self.assertRaises(RegistryError):
            ModelRegistry(self.registry).publish(run, expected_current="v0",
                expected_manifest_sha=result["manifest_sha256"], expected_report_sha=result["report_sha256"])
        # Modifying a source candidate after freezing cannot change the match.
        original = self.models[1].read_bytes()
        try:
            with patch.object(he, "MatchEngine", ScriptedEngine):
                safe = self.run_eval(model=1, progress=lambda _: self.models[1].write_bytes(b"changed"))
            frozen = he.read_json(Path(safe["run"]) / "manifest.json")
            self.assertEqual(frozen["candidate"]["sha256"], file_identity(Path(safe["run"]) / "candidate/model.gmvalue")["sha256"])
            self.assertEqual(safe["status"], "completed")
        finally:
            self.models[1].write_bytes(original)

    def test_numeric_and_training_identity_required(self):
        self.initial()
        before = self.pointer()
        model = self.models[1]
        fidelity = model.parent / "fidelity.json"
        original = fidelity.read_bytes()
        try:
            report = he.read_json(fidelity)
            report["fidelity"]["float_vs_integer"]["max_abs"] = 1
            write_json(fidelity, report)
            with self.assertRaises(he.EvaluationError):
                self.run_eval(model=1)
        finally:
            fidelity.write_bytes(original)
        manifest_path = model.parent / "training_positions.json"
        original = manifest_path.read_bytes()
        try:
            manifest_path.write_text("{}")
            with self.assertRaises(he.EvaluationError):
                self.run_eval(model=1)
        finally:
            manifest_path.write_bytes(original)
        self.assertEqual(before, self.pointer())

    def test_authoritative_double_three_and_four(self):
        candidate = he.checked_bundle(self.models[0])
        for blacks, reason in (([(6,7),(8,7),(7,6),(7,8)], "double_three"),
                ([(5,7),(6,7),(8,7),(7,5),(7,6),(7,8)], "double_four")):
            whites = [(x,0) for x in (0,2,4,6,8,10)][:len(blacks)]
            opening = [list(move) for pair in zip(blacks, whites) for move in pair]
            ScriptedEngine.mode = "foul"
            with patch.object(he, "MatchEngine", ScriptedEngine):
                game = he.play_pair_game(EXE, candidate, candidate, opening,
                                         self.cfg, 0, True, threading.Event())
            self.assertFalse(game["valid"])
            self.assertEqual(game["anomalies"][0]["kind"], "illegal-move")
            self.assertIn(reason, game["anomalies"][0]["detail"])
            self.assertEqual(game["rejected_move"], [7, 7])

    def test_transport_total_deadline_and_stop(self):
        fake = self.work / "chatty.py"
        fake.write_text("import time\nwhile True:\n print('MESSAGE busy',flush=True)\n time.sleep(.01)\n")
        spec = he.EngineSpec("A", "chatty", [sys.executable, str(fake)])
        stop = threading.Event()
        engine = he.MatchEngine(spec, 0.1, stop)
        try:
            started = time.monotonic()
            with self.assertRaises(he.EngineFailure) as caught:
                engine.move([0] * 225)
            self.assertEqual(caught.exception.kind, "timeout")
            self.assertLess(time.monotonic() - started, 0.6)
            stop.set()
            with self.assertRaises(InterruptedError):
                engine.move([0] * 225)
        finally:
            engine.close()
        self.assertIsNotNone(engine.p.poll())

    def test_training_exclusion_and_resume_union(self):
        first = he.read_json(self.models[0].parent / "training_positions.json")
        pool = build_pool(self.work / "new-pool.sqlite3", seed=109)
        out = self.work / "resumed"
        result = train_value(TrainConfig(pool=str(pool), output_dir=str(out),
            steps=10, batch_size=8, device="cpu", resume=str(self.models[0].parent / "last.pt")))
        union = he.read_json(out / "training_positions.json")
        self.assertTrue(set(first["d4_keys"]).issubset(union["d4_keys"]))
        self.assertGreater(len(union["d4_keys"]), len(first["d4_keys"]))
        # Real held-out opening identities are disjoint from the frozen set.
        self.initial()
        with patch.object(he, "MatchEngine", ScriptedEngine):
            evaluated = self.run_eval(model=1)
        frozen = he.read_json(Path(evaluated["run"]) / "manifest.json")
        training = he.read_json(self.models[1].parent / "training_positions.json")
        self.assertTrue(set(frozen["opening_d4_keys"]).isdisjoint(training["d4_keys"]))
        self.assertEqual(frozen["purpose"], "evaluation-only")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
