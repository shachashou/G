#!/usr/bin/env python3
"""Offline acceptance for the model cleanup planner (tools/model_cleanup.py)."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import model_cleanup  # noqa: E402
from model_cleanup import delete, human_size, scan, summary  # noqa: E402


def make_bundle(path: Path, size: int = 128) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.gmvalue").write_bytes(b"m" * size)
    (path / "last.pt").write_bytes(b"p" * size)
    return path


def set_mtime(path: Path, stamp: float) -> None:
    os.utime(path / "model.gmvalue", (stamp, stamp))
    os.utime(path, (stamp, stamp))


class CleanupFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="model-cleanup-")
        self.root = Path(self._tmp.name) / "weights"
        registry = self.root / "registry"
        (registry / "versions").mkdir(parents=True)
        make_bundle(registry / "versions" / "v0-abcd")
        (registry / "state.json").write_text(json.dumps({
            "format": "GMVALUE-REGISTRY1", "current": "v0",
            "versions": [{"version": "v0",
                          "model": "versions/v0-abcd/model.gmvalue"}]}),
            encoding="utf-8")
        # A running teacher task: nothing in it may be removed.
        self.running = self.root / "teacher-tasks" / ("a" * 32)
        self._task(self.running, "running", rounds=3, stamp=1000)
        # A finished self-play task: old rounds are cleanup targets.
        self.finished = self.root / "selfplay-tasks" / ("b" * 32)
        self._task(self.finished, "completed", rounds=3, stamp=2000)
        # A stopped, still resumable task.
        self.stopped = self.root / "transition-tasks" / ("c" * 32)
        self._task(self.stopped, "stopped", rounds=1, stamp=3000)
        self.manual = make_bundle(self.root / "value-run")
        set_mtime(self.manual, 4000)
        self.fixture = make_bundle(self.root / "value-demo")
        set_mtime(self.fixture, 5000)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _task(self, task: Path, status: str, *, rounds: int,
              stamp: float) -> None:
        (task / "candidates").mkdir(parents=True)
        (task / "state.json").write_text(json.dumps({"status": status}),
                                         encoding="utf-8")
        for index in range(rounds):
            folder = make_bundle(task / "candidates" / f"round-{index:05d}")
            set_mtime(folder, stamp + index)

    def ids(self, bundles) -> dict:
        return {bundle.id: bundle for bundle in bundles}


class ScanTests(CleanupFixture):
    def test_registry_is_never_a_candidate(self) -> None:
        bundles = scan(self.root)
        self.assertFalse(any("registry" in b.id for b in bundles))
        self.assertEqual(len(bundles), 9)  # 3 + 3 + 1 + 2 manual/fixture

    def test_running_task_rounds_are_all_protected(self) -> None:
        bundles = self.ids(scan(self.root, keep=1))
        for index in range(3):
            key = f"teacher-tasks/{'a' * 32}/candidates/round-{index:05d}"
            self.assertEqual(bundles[key].protection, "任务正在运行")
            self.assertFalse(bundles[key].removable)

    def test_newest_round_per_task_is_kept(self) -> None:
        bundles = self.ids(scan(self.root, keep=1))
        newest = f"selfplay-tasks/{'b' * 32}/candidates/round-00002"
        self.assertEqual(bundles[newest].protection, "任务保留最近 1 轮")
        for index in (0, 1):
            old = f"selfplay-tasks/{'b' * 32}/candidates/round-{index:05d}"
            self.assertTrue(bundles[old].removable)

    def test_keep_above_one_protects_more(self) -> None:
        bundles = self.ids(scan(self.root, keep=2))
        for index in (1, 2):
            key = f"selfplay-tasks/{'b' * 32}/candidates/round-{index:05d}"
            self.assertIsNotNone(bundles[key].protection)
        self.assertTrue(
            bundles[f"selfplay-tasks/{'b' * 32}/candidates/round-00000"].removable)

    def test_resumable_task_old_round_warns_but_is_removable(self) -> None:
        bundles = self.ids(scan(self.root, keep=0))
        key = f"transition-tasks/{'c' * 32}/candidates/round-00000"
        self.assertTrue(bundles[key].removable)
        self.assertIn("任务可恢复", bundles[key].warning)

    def test_manual_output_is_removable_but_warned(self) -> None:
        bundles = self.ids(scan(self.root))
        self.assertTrue(bundles["value-run"].removable)
        self.assertIn("手动输出", bundles["value-run"].warning)
        self.assertEqual(bundles["value-demo"].protection, "本地测试样例")

    def test_loaded_model_is_protected(self) -> None:
        loaded = str(self.manual / "model.gmvalue")
        bundles = self.ids(scan(self.root, loaded_model=loaded))
        self.assertEqual(bundles["value-run"].protection, "当前已加载")

    def test_summary_counts_sizes_and_defaults(self) -> None:
        data = summary(scan(self.root))
        self.assertEqual(data["total"], 9)
        self.assertEqual(data["protected"], 6)  # 3 running + newest + fixture
        self.assertEqual(data["removable"], 3)  # 2 old rounds + manual + ?
        self.assertEqual(data["default_selected"], 2)
        self.assertGreater(data["removable_bytes"], 0)

    def test_missing_root_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            scan(self.root / "nope")


class DeleteTests(CleanupFixture):
    def test_old_round_is_deleted_and_newest_survives(self) -> None:
        old = f"selfplay-tasks/{'b' * 32}/candidates/round-00000"
        newest = f"selfplay-tasks/{'b' * 32}/candidates/round-00002"
        results = delete(self.root, [old])
        self.assertTrue(results[0].ok)
        self.assertFalse((self.root / old).exists())
        self.assertTrue((self.root / newest / "model.gmvalue").is_file())

    def test_protected_selection_is_refused(self) -> None:
        newest = f"selfplay-tasks/{'b' * 32}/candidates/round-00002"
        results = delete(self.root, [newest])
        self.assertFalse(results[0].ok)
        self.assertIn("受保护", results[0].message)
        self.assertTrue((self.root / newest / "model.gmvalue").is_file())

    def test_unknown_and_escaping_ids_are_refused(self) -> None:
        results = delete(self.root, [
            "registry/versions/v0-abcd",
            "../outside",
            "selfplay-tasks/../../escape",
        ])
        self.assertFalse(any(result.ok for result in results))
        self.assertTrue((self.root / "registry" / "state.json").is_file())

    def test_delete_is_idempotent(self) -> None:
        old = f"selfplay-tasks/{'b' * 32}/candidates/round-00001"
        self.assertTrue(delete(self.root, [old])[0].ok)
        again = delete(self.root, [old])[0]
        self.assertFalse(again.ok)
        self.assertIn("未找到", again.message)

    def test_manual_bundle_needs_explicit_selection(self) -> None:
        self.assertTrue(delete(self.root, ["value-run"])[0].ok)
        self.assertFalse((self.root / "value-run").exists())
        # The fixture is never removable, even when asked for by name.
        refused = delete(self.root, ["value-demo"])[0]
        self.assertFalse(refused.ok)
        self.assertIn("本地测试样例", refused.message)


class FormatTests(unittest.TestCase):
    def test_human_size_units(self) -> None:
        self.assertEqual(human_size(0), "0 B")
        self.assertEqual(human_size(999), "999 B")
        self.assertEqual(human_size(2048), "2.0 KB")
        self.assertEqual(human_size(5 * 1024 * 1024), "5.0 MB")


class CliTests(CleanupFixture):
    def test_list_reports_summary_and_plan(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = model_cleanup.main(["--weights", str(self.root)])
        self.assertEqual(code, 0)
        output = buffer.getvalue()
        self.assertIn("可清理", output)
        self.assertIn("受保护", output)
        self.assertIn("共 9 个候选", output)

    def test_apply_without_selection_fails(self) -> None:
        code = model_cleanup.main(["--weights", str(self.root), "--apply"])
        self.assertEqual(code, 2)

    def test_apply_all_removes_old_task_rounds_only(self) -> None:
        code = model_cleanup.main(["--weights", str(self.root), "--apply", "--all"])
        self.assertEqual(code, 0)
        # Old rounds of the finished task are gone; running task untouched.
        self.assertFalse((self.finished / "candidates" / "round-00000").exists())
        self.assertTrue((self.finished / "candidates" / "round-00002").exists())
        self.assertTrue((self.running / "candidates" / "round-00000").exists())
        # Manual outputs require --include-manual.
        self.assertTrue((self.root / "value-run").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
