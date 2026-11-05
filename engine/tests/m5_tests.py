"""Ticket 12 public report/CLI contracts; bounded synthetic inputs only."""
from __future__ import annotations

import sys
import json
import subprocess
import tempfile
import os
import time
import argparse
import importlib.util
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "trainer")]
import match_runner as mr

EXE = ROOT / "build/engine/Release/gomoku_nnue.exe"
PROBE = ROOT / "build/engine/Release/nnue_probe.exe"


class GateTests(unittest.TestCase):
    def test_formal_gate_uses_strict_unrounded_score(self):
        self.assertFalse(mr.gate_result(130, 0, 200, 0.65, strict=True)["passed"])
        self.assertTrue(mr.gate_result(131, 0, 200, 0.65, strict=True)["passed"])
        self.assertTrue(mr.gate_result(130, 1, 200, 0.65, strict=True)["passed"])
        self.assertTrue(mr.gate_result(200, 0, 200, 1.0)["passed"])

    def test_cli_replays_frozen_openings_in_color_pairs(self):
        exe = EXE
        with tempfile.TemporaryDirectory() as tmp:
            schedule = Path(tmp) / "openings.json"
            output = Path(tmp) / "games.json"
            base = [sys.executable, str(ROOT / "tools/match_runner.py"),
                    "--games", "4", "--paired-openings", "--seed", "42"]
            frozen = subprocess.run(base + ["--freeze-openings", str(schedule)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(frozen.returncode, 0, frozen.stderr)
            command = f'"{exe}" --depth 1'
            result = subprocess.run(base + ["--openings", str(schedule),
                "--engine-a", command, "--engine-b", command,
                "--max-plies", "1", "--output", str(output)],
                capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            games = report["games"]
            for first, second in ((games[0], games[1]), (games[2], games[3])):
                self.assertEqual(first["moves"][:2], second["moves"][:2])
                self.assertEqual((first["a_color"], second["a_color"]),
                                 ("black", "white"))
            self.assertEqual(report["openings"]["sha256"],
                             mr.file_identity(schedule)["sha256"])

    def test_formal_gate_rejects_silent_handcrafted_fallback(self):
        exe = EXE
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            schedule, output, bad = tmp / "openings.json", tmp / "out.json", tmp / "bad.nnue"
            bad.write_bytes(b"bad weights")
            base = [sys.executable, str(ROOT / "tools/match_runner.py"),
                    "--games", "200", "--paired-openings"]
            subprocess.run(base + ["--freeze-openings", str(schedule)], check=True,
                           capture_output=True, timeout=10)
            result = subprocess.run(base + ["--m5-gate", "--openings", str(schedule),
                "--engine-a", f'"{exe}" --depth 2 --nnue "{bad}"',
                "--engine-b", f'"{exe}" --depth 2', "--output", str(output)],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertFalse(report["complete"])
            self.assertFalse(report.get("gate", {}).get("passed", False))
            self.assertIn("handcrafted", report["invalid_reason"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "training requirements not installed")
class DiagnosticTests(unittest.TestCase):
    def test_baseline_is_fitted_only_on_training_labels(self):
        from diagnose import baseline_report
        report = baseline_report([0, 0], [0, 2], [400], [2], [400])
        self.assertEqual(report["constant_target"], 0)
        self.assertAlmostEqual(report["baseline_mse"], 0.5800256584)
        self.assertAlmostEqual(report["baseline_ce"], 0.69314718056)
        self.assertEqual(report["model_mse"], 0)
        self.assertEqual(report["mse_ratio"], 0)

    def test_diagnostic_clis_produce_disjoint_fidelity_and_fixed_fit_reports(self):
        import torch
        import nnue
        from training_tests import write_synthetic
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data, checkpoint = tmp / "data.binpack", tmp / "original.pt"
            write_synthetic(data, games=10, per_game=4)
            model = nnue.NNUEModel(seed=3)
            # Fractional trained bias must not disappear into the rebuilt reference.
            with torch.no_grad():
                model.b2.add_(0.4)
            torch.save({"model_state": model.state_dict(), "config": vars(model.config),
                        "data_identity": mr.file_identity(data),
                        "train_args": {"seed": 42, "val_fraction": 0.3, "limit": 8}}, checkpoint)
            base = [sys.executable, str(ROOT / "trainer/diagnose.py"),
                    "--data", str(data), "--samples", "4"]
            output = tmp / "fidelity"
            result = subprocess.run(base + ["fidelity", "--checkpoint", str(checkpoint),
                "--probe", str(PROBE),
                "--out-dir", str(output)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "report.json").read_text())
            self.assertTrue(report["complete"])
            self.assertTrue(report["cpp"]["exact"])
            self.assertEqual(report["manifest"]["baseline_train_records"], 8)
            self.assertTrue(report["manifest"]["data_identity"]["verified"])
            self.assertIn("ce_ratio", report["baseline"]["original_float"])
            manifest = report["manifest"]
            self.assertFalse(set(manifest["calibration_games"]) & set(manifest["assessment_games"]))
            self.assertGreater(report["comparisons"]["original_vs_final"]["score_units"]["mae"], 0)
            for name in ("fit1", "fit2"):
                result = subprocess.run(base + ["overfit", "--steps", "2", "--device", "cpu", "--hidden-lr", "0.001",
                    "--out-dir", str(tmp / name)], capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
            first = json.loads((tmp / "fit1/report.json").read_text())
            second = json.loads((tmp / "fit2/report.json").read_text())
            self.assertEqual(first["manifest"], second["manifest"])
            self.assertFalse(first["manifest"]["augmentation"])
            self.assertEqual(first["arms"]["float"]["curve"], second["arms"]["float"]["curve"])
            trained = torch.load(tmp / "fit1/float.pt", weights_only=False)["model_state"]
            self.assertTrue(torch.load(tmp / "fit1/ste.pt", weights_only=False)["train_args"]["quant_forward"])
            initial = nnue.NNUEModel(seed=20261106).state_dict()
            self.assertLess(float((trained["W1"] - initial["W1"]).abs().max()), .0021)
            self.assertGreater(float((trained["W2"] - initial["W2"]).abs().max()), .01)
            train_result = subprocess.run([sys.executable, str(ROOT / "trainer/train.py"),
                "--data", str(data), "--epochs", "2", "--batch-size", "1024", "--device", "cpu",
                "--hidden-lr", "0.001", "--seed", "20261106", "--output", str(tmp / "trained.pt"),
                "--history", str(tmp / "training.json")], capture_output=True, text=True, timeout=30)
            self.assertEqual(train_result.returncode, 0, train_result.stderr)
            training_report = json.loads((tmp / "training.json").read_text())
            self.assertEqual(training_report["config"]["hidden_lr"], .001)
            trained_state = torch.load(tmp / "trained.pt", weights_only=False)["model_state"]
            self.assertLess(float((trained_state["W1"] - initial["W1"]).abs().max()), .0021)
            # QAT exports must retain their trained integer forward; calibrating
            # toward unrounded master weights would change the trained model.
            import weights_format as wf
            with torch.no_grad():
                model.W2.add_(.49)
            qat_checkpoint = tmp / "qat.pt"
            payload = torch.load(checkpoint, weights_only=False)
            payload["model_state"] = model.state_dict()
            payload["train_args"]["quant_forward"] = True
            torch.save(payload, qat_checkpoint)
            expected_weights = model.quantized_weights()
            for tool in ("export", "fidelity"):
                if tool == "export":
                    args = [str(ROOT / "trainer/export.py"), "--output", str(tmp / "qat.nnue"),
                            "--report", str(tmp / "qat_export.json")]
                    weights_file = tmp / "qat.nnue"
                else:
                    args = [str(ROOT / "trainer/diagnose.py"), "fidelity", "--out-dir", str(tmp / "qat_fidelity")]
                    weights_file = tmp / "qat_fidelity/candidate.nnue"
                result = subprocess.run([sys.executable, *args, "--checkpoint", str(qat_checkpoint),
                    "--data", str(data)], capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(mr.file_identity(weights_file)["sha256"],
                                 hashlib.sha256(wf.pack_weights(expected_weights)).hexdigest())
            # A replaced dataset must not masquerade as checkpoint holdout data.
            data.write_bytes(data.read_bytes() + b"changed")
            invalid = subprocess.run(base + ["fidelity", "--checkpoint", str(checkpoint),
                "--out-dir", str(tmp / "replaced")], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(invalid.returncode, 0)


class BudgetTests(unittest.TestCase):
    def test_bare_python_keeps_the_parent_virtual_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            job = tmp / "run"
            probe = "import sys; print(sys.prefix, flush=True)"
            if importlib.util.find_spec("torch"):
                probe += "; import torch"
            result = subprocess.run([sys.executable, str(ROOT / "tools/budget_run.py"),
                "--ledger", str(tmp / "budget.json"), "--seconds", "10", "--out-dir", str(job),
                "--", "python", "-c", probe],
                capture_output=True, text=True, timeout=15,
                env={**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]})
            log = (job / "console.log").read_text()
            self.assertEqual(result.returncode, 0, log)
            self.assertEqual(Path(log.splitlines()[0]).resolve(), Path(sys.prefix).resolve())

    def test_deadline_kills_descendants_and_keeps_cumulative_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            script = tmp / "tree.py"
            heartbeat = tmp / "heartbeat.txt"
            script.write_text('''import subprocess, sys, time
if len(sys.argv) == 2:
    child = subprocess.Popen([sys.executable, __file__, sys.argv[1], "child"])
    time.sleep(30)
else:
    while True:
        with open(sys.argv[1], "a") as f: f.write("alive\\n")
        time.sleep(.05)
''')
            ledger = tmp / "budget.json"
            base = [sys.executable, str(ROOT / "tools/budget_run.py"),
                    "--ledger", str(ledger)]
            job = tmp / "timeout"
            result = subprocess.run(base + ["--seconds", "1", "--out-dir", str(job),
                "--", sys.executable, str(script), str(heartbeat)],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
            report = json.loads((job / "run.json").read_text())
            self.assertEqual(report["status"], "timeout")
            self.assertFalse(report["complete"])
            self.assertTrue(heartbeat.exists())
            size = heartbeat.stat().st_size
            time.sleep(.15)
            self.assertEqual(heartbeat.stat().st_size, size, "descendant must stop too")
            used = json.loads(ledger.read_text())["used_seconds"]
            self.assertGreaterEqual(used, 1)
            result = subprocess.run(base + ["--seconds", "2", "--out-dir", str(tmp / "ok"),
                "--", sys.executable, "-c", "print('done')"],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            final = json.loads(ledger.read_text())
            self.assertGreater(final["used_seconds"], used)
            self.assertEqual(len(final["runs"]), 2)
            # Reservations that cannot fit the remaining budget never launch.
            result = subprocess.run(base + ["--seconds", "7200", "--out-dir", str(tmp / "refuse"),
                "--", sys.executable, "-c", "raise Exception('must not run')"],
                capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((tmp / "refuse").exists())

    def test_early_successful_parent_does_not_hide_interrupted_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            script = tmp / "early.py"
            script.write_text('''import subprocess, sys
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
''')
            job = tmp / "run"
            result = subprocess.run([sys.executable, str(ROOT / "tools/budget_run.py"),
                "--ledger", str(tmp / "budget.json"), "--seconds", "5", "--out-dir", str(job),
                "--", sys.executable, str(script)], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((job / "run.json").read_text())
            self.assertFalse(report["complete"])
            self.assertEqual(report["status"], "incomplete_descendants")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine", type=Path, default=EXE)
    parser.add_argument("--probe", type=Path, default=PROBE)
    options, tests = parser.parse_known_args()
    EXE, PROBE = options.engine.resolve(), options.probe.resolve()
    unittest.main(argv=[sys.argv[0], *tests])
