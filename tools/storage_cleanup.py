#!/usr/bin/env python3
"""Find and remove historical training data that no longer earns its disk.

Two kinds of history are handled, both with hard protections:

* **task directories** (``weights/teacher-tasks/<id>`` and friends): a task
  that is running is off-limits; ``completed`` / ``failed`` tasks are safe to
  drop because they are not offered for resume; paused / stopped tasks are
  removable only explicitly, because deleting them loses the ability to resume.
* **evaluation runs** (``data/evaluations/<run>``): a run referenced by the
  model registry (formal or historical version) *or* by an interrupted
  evaluation inside a resumable task is evidence and never removable. Other
  runs are old reports; removing them loses the historical "why rejected"
  record but does not affect the current version.

Candidate model bundles are handled separately by ``tools/model_cleanup.py``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from model_cleanup import human_size

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weights"
DEFAULT_EVALUATIONS = ROOT / "data" / "evaluations"
DEFAULT_REGISTRY = DEFAULT_WEIGHTS / "registry"
TASK_ROOTS = ("teacher-tasks", "selfplay-tasks", "transition-tasks")
TASK_LABELS = {
    "teacher-tasks": "教师训练任务",
    "selfplay-tasks": "自主训练任务",
    "transition-tasks": "自主过渡任务",
}
RUNNING_STATUSES = ("running", "pausing", "resuming", "stopping")
FINISHED_STATUSES = ("completed", "failed")
TASK_PREFIX = "task:"
EVAL_PREFIX = "evaluation:"


@dataclass(frozen=True)
class DataEntry:
    id: str                  # prefixed id: "task:<rel>" / "evaluation:<name>"
    path: Path
    kind: str                # "task" or "evaluation"
    group: str
    group_label: str
    label: str
    size: int
    mtime: float
    protection: str | None = None
    warning: str | None = None

    @property
    def removable(self) -> bool:
        return self.protection is None

    @property
    def default_checked(self) -> bool:
        return self.removable and self.kind == "task" and self.warning is None


@dataclass(frozen=True)
class DeleteResult:
    id: str
    ok: bool
    freed: int
    message: str = ""


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _directory_size(path: Path) -> int:
    total = 0
    try:
        entries = list(path.rglob("*"))
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _resolved(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return str(Path(value).resolve())
    except OSError:
        return None


def _task_status(task_dir: Path) -> str:
    status = _read_json(task_dir / "state.json").get("status")
    return status if isinstance(status, str) and status else "unknown"


def registry_runs(registry_root: str | Path) -> set[str]:
    """Evaluation runs referenced by the model registry (any version)."""
    state = _read_json(Path(registry_root) / "state.json")
    runs: set[str] = set()
    for entry in state.get("versions", []) or []:
        if isinstance(entry, dict):
            run = _resolved(entry.get("run"))
            if run:
                runs.add(run)
    return runs


def task_interrupted_runs(weights_root: str | Path) -> set[str]:
    """Runs a resumable task would continue instead of starting over."""
    runs: set[str] = set()
    root = Path(weights_root)
    for task_root in TASK_ROOTS:
        base = root / task_root
        if not base.is_dir():
            continue
        for task_dir in base.iterdir():
            if not task_dir.is_dir():
                continue
            status = _task_status(task_dir)
            if status in RUNNING_STATUSES or status in FINISHED_STATUSES:
                continue
            state = _read_json(task_dir / "state.json")
            for record in state.get("rounds", []) or []:
                if not isinstance(record, dict):
                    continue
                evaluation = record.get("evaluation")
                if not isinstance(evaluation, dict):
                    continue
                if evaluation.get("status") not in ("interrupted", "evaluating"):
                    continue
                run = _resolved(evaluation.get("run"))
                if run:
                    runs.add(run)
    return runs


def scan(weights_root: str | Path = DEFAULT_WEIGHTS,
         evaluations_root: str | Path = DEFAULT_EVALUATIONS, *,
         registry_root: str | Path | None = None) -> list[DataEntry]:
    """List task directories and evaluation runs with their protection state."""
    weights = Path(weights_root).resolve()
    evaluations = Path(evaluations_root).resolve()
    registry = Path(registry_root or weights / "registry").resolve()
    protected_runs = registry_runs(registry) | task_interrupted_runs(weights)
    entries: list[DataEntry] = []

    if weights.is_dir():
        for task_root in TASK_ROOTS:
            base = weights / task_root
            if not base.is_dir():
                continue
            for task_dir in sorted(base.iterdir()):
                if not task_dir.is_dir() or task_dir.is_symlink():
                    continue
                status = _task_status(task_dir)
                entry_id = TASK_PREFIX + task_dir.name
                if status in RUNNING_STATUSES:
                    protection, warning = "任务正在运行", None
                elif status in FINISHED_STATUSES:
                    protection, warning = None, None
                else:
                    protection = None
                    warning = "可恢复任务：删除后无法继续恢复"
                entries.append(DataEntry(
                    id=entry_id, path=task_dir.resolve(), kind="task",
                    group=task_root, group_label=TASK_LABELS[task_root],
                    label=f"{task_dir.name[:8]}（{status}）",
                    size=_directory_size(task_dir),
                    mtime=task_dir.stat().st_mtime,
                    protection=protection, warning=warning))

    if evaluations.is_dir():
        for run in sorted(evaluations.iterdir()):
            if not run.is_dir() or run.is_symlink():
                continue
            resolved = str(run.resolve())
            entry_id = EVAL_PREFIX + run.name
            if resolved in protected_runs:
                protection = "登记版本证据"
                warning = None
            else:
                protection = None
                warning = "旧测评记录：删除后失去历史证据"
            entries.append(DataEntry(
                id=entry_id, path=run.resolve(), kind="evaluation",
                group="evaluations", group_label="测评记录",
                label=run.name, size=_directory_size(run),
                mtime=run.stat().st_mtime,
                protection=protection, warning=warning))
    return entries


def summary(entries: list[DataEntry]) -> dict:
    removable = [entry for entry in entries if entry.removable]
    default = [entry for entry in entries if entry.default_checked]
    return {
        "total": len(entries),
        "total_bytes": sum(entry.size for entry in entries),
        "removable": len(removable),
        "removable_bytes": sum(entry.size for entry in removable),
        "protected": len(entries) - len(removable),
        "default_selected": len(default),
        "default_selected_bytes": sum(entry.size for entry in default),
    }


def delete(weights_root: str | Path, evaluations_root: str | Path,
           ids: list[str], *, registry_root: str | Path | None = None
           ) -> list[DeleteResult]:
    """Delete the selected history after re-scanning and re-checking safety."""
    weights = Path(weights_root).resolve()
    evaluations = Path(evaluations_root).resolve()
    registry = Path(registry_root or weights / "registry").resolve()
    fresh = {entry.id: entry for entry in scan(
        weights, evaluations, registry_root=registry)}
    results: list[DeleteResult] = []
    for entry_id in ids:
        entry = fresh.get(entry_id)
        if entry is None:
            results.append(DeleteResult(entry_id, False, 0,
                                        "未找到（可能已被删除）"))
            continue
        if entry.protection:
            results.append(DeleteResult(entry_id, False, 0,
                                        "受保护：" + entry.protection))
            continue
        root = weights if entry.kind == "task" else evaluations
        target = entry.path
        if target == root or not target.is_relative_to(root):
            results.append(DeleteResult(entry_id, False, 0, "路径不在允许范围内"))
            continue
        if target.is_symlink():
            results.append(DeleteResult(entry_id, False, 0, "拒绝删除符号链接"))
            continue
        if entry.kind == "task" and target.is_relative_to(registry):
            results.append(DeleteResult(entry_id, False, 0, "拒绝删除登记表目录"))
            continue
        try:
            shutil.rmtree(target)
        except OSError as exc:
            results.append(DeleteResult(entry_id, False, 0, str(exc)))
            continue
        results.append(DeleteResult(entry_id, True, entry.size))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--evaluations", default=str(DEFAULT_EVALUATIONS))
    parser.add_argument("--registry", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--select", action="append", default=[],
                        help='entry id, e.g. "task:<id>" or "evaluation:<run>"')
    parser.add_argument("--all", action="store_true",
                        help="delete finished tasks and old evaluation runs")
    parser.add_argument("--include-resumable", action="store_true",
                        help="with --all, also delete paused/stopped tasks")
    args = parser.parse_args(argv)

    try:
        entries = scan(args.weights, args.evaluations,
                       registry_root=args.registry)
    except ValueError as exc:
        print(f"storage_cleanup: {exc}", file=sys.stderr)
        return 2

    if not args.apply:
        if args.json:
            print(json.dumps({
                "entries": [asdict(entry) | {"path": str(entry.path)}
                            for entry in entries],
                "summary": summary(entries)}, ensure_ascii=False, indent=2))
        else:
            for entry in entries:
                state = ("受保护：" + entry.protection if entry.protection
                         else "可清理" + ("（" + entry.warning + "）"
                                          if entry.warning else ""))
                print(f"{entry.id}  {human_size(entry.size)}  {state}")
            data = summary(entries)
            print(f"\n共 {data['total']} 项 · {human_size(data['total_bytes'])}；"
                  f"可清理 {data['removable']} 项 · "
                  f"{human_size(data['removable_bytes'])}"
                  "（--all 删除已完成/失败任务与旧测评；"
                  "可恢复任务需 --include-resumable）")
        return 0

    selected = list(dict.fromkeys(args.select))
    if args.all:
        for entry in entries:
            if not entry.removable:
                continue
            if (entry.kind == "task" and entry.warning
                    and not args.include_resumable):
                continue
            selected.append(entry.id)
    if not selected:
        print("storage_cleanup: nothing selected; pass --select ID or --all",
              file=sys.stderr)
        return 2
    results = delete(args.weights, args.evaluations, selected,
                     registry_root=args.registry)
    freed = 0
    for result in results:
        if result.ok:
            freed += result.freed
            print(f"已删除 {result.id} ({human_size(result.freed)})")
        else:
            print(f"跳过 {result.id}：{result.message}", file=sys.stderr)
    print(f"共释放 {human_size(freed)}")
    return 0 if any(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
