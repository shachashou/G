#!/usr/bin/env python3
"""Offline acceptance for the historical-data planner (tools/storage_cleanup.py)."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import storage_cleanup  # noqa: E402
from storage_cleanup import delete, scan, summary  # noqa: E402


def make_task(path: Path, status: str, *, rounds=None,
              stamp: float = 0.0) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    state = {"status": status, "rounds": rounds or []}
    (path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (path / "manifest.json").write_text("{}", encoding="utf-8")
    (path / "games").mkdir(exist_ok=True)
    (path / "games" / "game-00001.json").write_bytes(b"g" * 128)
    if stamp:
        import os
        os.utime(path, (stamp, stamp))
    return path


def make_run(path: Path, stamp: float = 0.0) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "report.json").write_text("{}", encoding="utf-8")
    (path / "manifest.json").write_text("{}", encoding="utf-8")
    if stamp:
        import os
        os.utime(path, (stamp, stamp))
    return path


class HistoryFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="storage-cleanup-")
        base = Path(self._tmp.name)
        self.weights = base / "weights"
        self.evaluations = base / "data" / "evaluations"
        self.registry = self.weights / "registry"
        self.registry.mkdir(parents=True)
        self.referenced = make_run(self.evaluations / "run-ref")
        self.interrupted = make_run(self.evaluations / "run-interrupted")
        self.stale_a = make_run(self.evaluations / "run-stale-a")
        self.stale_b = make_run(self.evaluations / "run-stale-b")
        (self.registry / "state.json").write_text(json.dumps({
            "format": "GMVALUE-REGISTRY1", "current": "v1",
            "versions": [
                {"version": "v0", "model": "versions/v0/model.gmvalue",
                 "run": str(self.referenced)},
                {"version": "v1", "model": "versions/v1/model.gmvalue",
                 "run": str(self.referenced)}]}), encoding="utf-8")
        self.running = make_task(
            self.weights / "teacher-tasks" / ("a" * 32), "running")
        self.done = make_task(
            self.weights / "selfplay-tasks" / ("b" * 32), "completed")
        self.failed = make_task(
            self.weights / "teacher-tasks" / ("f" * 32), "failed")
        self.stopped = make_task(
            self.weights / "transition-tasks" / ("c" * 32), "stopped",
            rounds=[{"round": 0, "evaluation":
                     {"status": "interrupted", "run": str(self.interrupted)}}])

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def entries(self) -> dict:
        return {entry.id: entry
                for entry in scan(self.weights, self.evaluations,
                                  registry_root=self.registry)}


class ScanTests(HistoryFixture):
    def test_running_task_is_protected(self) -> None:
        entry = self.entries()["task:" + "a" * 32]
        self.assertEqual(entry.protection, "任务正在运行")
        self.assertFalse(entry.removable)
        self.assertFalse(entry.default_checked)

    def test_completed_and_failed_tasks_are_default_targets(self) -> None:
        entries = self.entries()
        for name in ("b" * 32, "f" * 32):
            entry = entries["task:" + name]
            self.assertTrue(entry.removable)
            self.assertTrue(entry.default_checked)

    def test_stopped_task_is_removable_but_warned_and_unchecked(self) -> None:
        entry = self.entries()["task:" + "c" * 32]
        self.assertTrue(entry.removable)
        self.assertIn("可恢复任务", entry.warning)
        self.assertFalse(entry.default_checked)

    def test_registry_run_is_evidence(self) -> None:
        entries = self.entries()
        self.assertEqual(entries["evaluation:run-ref"].protection,
                         "登记版本证据")
        self.assertFalse(entries["evaluation:run-ref"].removable)

    def test_interrupted_task_run_is_evidence(self) -> None:
        entries = self.entries()
        self.assertEqual(entries["evaluation:run-interrupted"].protection,
                         "登记版本证据")

    def test_stale_runs_are_removable_and_unchecked(self) -> None:
        entries = self.entries()
        for name in ("run-stale-a", "run-stale-b"):
            entry = entries["evaluation:" + name]
            self.assertTrue(entry.removable)
            self.assertIn("旧测评记录", entry.warning)
            self.assertFalse(entry.default_checked)

    def test_summary_counts(self) -> None:
        data = summary(scan(self.weights, self.evaluations,
                            registry_root=self.registry))
        self.assertEqual(data["total"], 8)      # 4 tasks + 4 runs
        self.assertEqual(data["protected"], 3)  # running + 2 evidence runs
        self.assertEqual(data["removable"], 5)
        self.assertEqual(data["default_selected"], 2)  # completed + failed tasks
        self.assertGreater(data["removable_bytes"], 0)

    def test_missing_roots_are_tolerated(self) -> None:
        entries = scan(self.weights / "none", self.evaluations / "none")
        self.assertEqual(entries, [])


class DeleteTests(HistoryFixture):
    def test_completed_task_and_stale_run_are_deleted(self) -> None:
        results = delete(self.weights, self.evaluations, [
            "task:" + "b" * 32, "evaluation:run-stale-a"])
        self.assertTrue(all(result.ok for result in results))
        self.assertFalse(self.done.exists())
        self.assertFalse(self.stale_a.exists())

    def test_running_task_and_evidence_are_refused(self) -> None:
        results = delete(self.weights, self.evaluations, [
            "task:" + "a" * 32, "evaluation:run-ref",
            "evaluation:run-interrupted"])
        self.assertFalse(any(result.ok for result in results))
        self.assertTrue(self.running.exists())
        self.assertTrue(self.referenced.exists())
        self.assertTrue(self.interrupted.exists())

    def test_unknown_and_escaping_ids_are_refused(self) -> None:
        results = delete(self.weights, self.evaluations, [
            "task:..", "evaluation:..", "evaluation:run-ref/../run-stale-a"])
        self.assertFalse(any(result.ok for result in results))
        self.assertTrue(self.referenced.exists())
        self.assertTrue(self.stale_a.exists())

    def test_delete_is_idempotent(self) -> None:
        self.assertTrue(delete(self.weights, self.evaluations,
                               ["task:" + "b" * 32])[0].ok)
        again = delete(self.weights, self.evaluations,
                       ["task:" + "b" * 32])[0]
        self.assertFalse(again.ok)
        self.assertIn("未找到", again.message)


class CliTests(HistoryFixture):
    def test_list_reports_plan(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = storage_cleanup.main([
                "--weights", str(self.weights),
                "--evaluations", str(self.evaluations),
                "--registry", str(self.registry)])
        self.assertEqual(code, 0)
        output = buffer.getvalue()
        self.assertIn("任务正在运行", output)
        self.assertIn("登记版本证据", output)
        self.assertIn("共 8 项", output)

    def test_apply_without_selection_fails(self) -> None:
        code = storage_cleanup.main([
            "--weights", str(self.weights),
            "--evaluations", str(self.evaluations), "--apply"])
        self.assertEqual(code, 2)

    def test_apply_all_keeps_resumable_and_evidence(self) -> None:
        code = storage_cleanup.main([
            "--weights", str(self.weights),
            "--evaluations", str(self.evaluations),
            "--registry", str(self.registry), "--apply", "--all"])
        self.assertEqual(code, 0)
        self.assertFalse(self.done.exists())
        self.assertFalse(self.failed.exists())
        self.assertFalse(self.stale_a.exists())
        self.assertTrue(self.stopped.exists())   # resumable needs opt-in
        self.assertTrue(self.running.exists())
        self.assertTrue(self.referenced.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
