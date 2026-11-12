#!/usr/bin/env python3
"""Find and remove unused candidate model bundles under ``weights/``.

A *bundle* is a directory that contains ``model.gmvalue`` — one trained
candidate with its checkpoint, fidelity report and training-position list.
The cleanup only ever removes whole bundles; a lone ``model.gmvalue`` is not
useful for play, so partial deletion is not offered.

Hard protections (never removable, reported as ``protection``):

* the registered formal and frozen versions under ``weights/registry``
* the model the client currently has loaded (``--loaded``)
* local fixtures such as ``weights/value-demo`` (used by parity tests)
* every candidate of a task that is still running
* the newest ``--keep`` candidates of each task (default 1) so a paused or
  stopped task can still be resumed from its latest checkpoint

Everything else is a candidate for deletion. Bundles from paused/stopped
tasks carry a ``warning`` because removing them shrinks what that task can
resume from, but the newest checkpoint is always kept.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weights"
DEFAULT_REGISTRY = DEFAULT_WEIGHTS / "registry"
DEFAULT_FIXTURES = ("value-demo",)
DEFAULT_KEEP = 1
TASK_ROOTS = ("teacher-tasks", "selfplay-tasks", "transition-tasks")
TASK_LABELS = {
    "teacher-tasks": "教师训练",
    "selfplay-tasks": "自主训练",
    "transition-tasks": "自主过渡",
}
RUNNING_STATUSES = ("running", "pausing", "resuming", "stopping")
FINISHED_STATUSES = ("completed", "failed")
MODEL_NAME = "model.gmvalue"


@dataclass(frozen=True)
class Bundle:
    """One removable candidate model directory and why it may be removed."""

    id: str                  # forward-slash path relative to the weights root
    directory: Path
    model: Path
    kind: str                # "task" (a training round) or "manual" (an output)
    group: str               # task id, or "manual" for standalone outputs
    group_label: str         # human label for the group header
    size: int
    mtime: float
    protection: str | None = None
    warning: str | None = None

    @property
    def removable(self) -> bool:
        return self.protection is None


@dataclass(frozen=True)
class DeleteResult:
    id: str
    ok: bool
    freed: int
    message: str = ""


def human_size(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


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


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _task_status(task_dir: Path) -> str:
    status = _read_json(task_dir / "state.json").get("status")
    return status if isinstance(status, str) and status else "unknown"


def _registered_paths(registry_root: Path) -> set[str]:
    """Resolved model paths recorded by the registry (formal + historical)."""
    state = _read_json(Path(registry_root) / "state.json")
    paths: set[str] = set()
    for entry in state.get("versions", []) or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("model"), str):
            continue
        try:
            paths.add(str((Path(registry_root) / entry["model"]).resolve()))
        except OSError:
            continue
    return paths


def _same_path(left: Path, right: str | None) -> bool:
    if not right:
        return False
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return False


def _bundle(weights_root: Path, directory: Path, *, kind: str, group: str,
            group_label: str) -> Bundle | None:
    model = directory / MODEL_NAME
    if not model.is_file():
        return None
    try:
        mtime = model.stat().st_mtime
    except OSError:
        mtime = 0.0
    return Bundle(
        id=directory.resolve().relative_to(weights_root).as_posix(),
        directory=directory.resolve(),
        model=model.resolve(),
        kind=kind,
        group=group,
        group_label=group_label,
        size=_directory_size(directory),
        mtime=mtime,
    )


def scan(weights_root: str | Path = DEFAULT_WEIGHTS, *,
         registry_root: str | Path | None = None,
         loaded_model: str | None = None,
         keep: int = DEFAULT_KEEP,
         fixtures: tuple[str, ...] = DEFAULT_FIXTURES) -> list[Bundle]:
    """List candidate bundles with their protection status.

    ``keep`` is the minimum number of newest candidates preserved per task;
    ``fixtures`` are weights-relative paths (or prefixes) that are never
    treated as cleanup targets.
    """
    root = Path(weights_root).resolve()
    if not root.is_dir():
        raise ValueError(f"weights directory not found: {root}")
    if keep < 0:
        raise ValueError("keep must be >= 0")
    registry = Path(registry_root or root / "registry").resolve()
    registered = _registered_paths(registry)
    bundles: list[Bundle] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if entry.resolve() == registry:
            continue
        if entry.name in TASK_ROOTS:
            for task in sorted(p for p in entry.iterdir() if p.is_dir()):
                status = _task_status(task)
                running = status in RUNNING_STATUSES
                resumable = not running and status not in FINISHED_STATUSES
                candidates = sorted(
                    p for p in (task / "candidates").glob("round-*")
                    if p.is_dir()) if (task / "candidates").is_dir() else []
                found = [b for b in (
                    _bundle(root, folder, kind="task",
                            group=f"{entry.name}/{task.name}",
                            group_label=f"{TASK_LABELS.get(entry.name, entry.name)}"
                                        f" {task.name[:8]}（{status}）")
                    for folder in candidates) if b is not None]
                found.sort(key=lambda b: b.mtime, reverse=True)
                for rank, bundle in enumerate(found):
                    if running:
                        protection = "任务正在运行"
                    elif rank < keep:
                        protection = f"任务保留最近 {keep} 轮"
                    else:
                        protection = None
                    warning = None
                    if protection is None and resumable:
                        warning = "任务可恢复：只删旧轮，最近检查点仍在"
                    bundles.append(_replace(bundle, protection=protection,
                                            warning=warning))
            continue
        bundle = _bundle(root, entry, kind="manual", group="manual",
                         group_label="手动训练输出")
        if bundle is None:
            continue
        relative = bundle.id
        if any(relative == fixture or relative.startswith(fixture + "/")
               for fixture in fixtures):
            bundles.append(_replace(bundle, protection="本地测试样例"))
        elif _same_path(bundle.model, loaded_model):
            bundles.append(_replace(bundle, protection="当前已加载"))
        elif str(bundle.model) in registered:
            bundles.append(_replace(bundle, protection="已登记版本"))
        else:
            bundles.append(_replace(bundle, warning="手动输出（默认不勾选）"))
    return bundles


def _replace(bundle: Bundle, **changes) -> Bundle:
    values = asdict(bundle)
    values.update(changes)
    return Bundle(**values)


def summary(bundles: list[Bundle]) -> dict:
    removable = [b for b in bundles if b.removable]
    return {
        "total": len(bundles),
        "total_bytes": sum(b.size for b in bundles),
        "removable": len(removable),
        "removable_bytes": sum(b.size for b in removable),
        "protected": len(bundles) - len(removable),
        "default_selected": sum(1 for b in removable if b.kind == "task"),
        "default_selected_bytes": sum(
            b.size for b in removable if b.kind == "task"),
    }


def delete(weights_root: str | Path, ids: list[str], *,
           registry_root: str | Path | None = None,
           loaded_model: str | None = None,
           keep: int = DEFAULT_KEEP,
           fixtures: tuple[str, ...] = DEFAULT_FIXTURES) -> list[DeleteResult]:
    """Delete the selected bundles after re-scanning and re-checking safety."""
    root = Path(weights_root).resolve()
    registry = Path(registry_root or root / "registry").resolve()
    fresh = {bundle.id: bundle for bundle in scan(
        root, registry_root=registry, loaded_model=loaded_model,
        keep=keep, fixtures=fixtures)}
    results: list[DeleteResult] = []
    for bundle_id in ids:
        bundle = fresh.get(bundle_id)
        if bundle is None:
            results.append(DeleteResult(bundle_id, False, 0, "未找到（可能已被删除）"))
            continue
        if bundle.protection:
            results.append(DeleteResult(bundle_id, False, 0,
                                        "受保护：" + bundle.protection))
            continue
        target = bundle.directory
        if target == root or not target.is_relative_to(root):
            results.append(DeleteResult(bundle_id, False, 0, "路径不在 weights 内"))
            continue
        if target.is_symlink():
            results.append(DeleteResult(bundle_id, False, 0, "拒绝删除符号链接"))
            continue
        if target.is_relative_to(registry):
            results.append(DeleteResult(bundle_id, False, 0, "拒绝删除登记表版本"))
            continue
        try:
            shutil.rmtree(target)
        except OSError as exc:
            results.append(DeleteResult(bundle_id, False, 0, str(exc)))
            continue
        results.append(DeleteResult(bundle_id, True, bundle.size))
    return results


def describe_plan(bundles: list[Bundle]) -> str:
    lines = []
    for bundle in bundles:
        state = ("受保护：" + bundle.protection if bundle.protection
                 else "可清理" + ("（" + bundle.warning + "）" if bundle.warning else ""))
        lines.append(f"{bundle.id}  {human_size(bundle.size)}  {state}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--registry", default=None)
    parser.add_argument("--loaded", default=None,
                        help="currently loaded model path; never removed")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP,
                        help="newest candidates kept per task (default 1)")
    parser.add_argument("--json", action="store_true", help="machine-readable list")
    parser.add_argument("--apply", action="store_true", help="delete instead of list")
    parser.add_argument("--select", action="append", default=[],
                        help="bundle id to delete (repeatable)")
    parser.add_argument("--all", action="store_true",
                        help="delete every removable task candidate not warned about")
    parser.add_argument("--include-manual", action="store_true",
                        help="with --all, also delete manual outputs")
    args = parser.parse_args(argv)

    try:
        bundles = scan(args.weights, registry_root=args.registry,
                       loaded_model=args.loaded, keep=args.keep)
    except ValueError as exc:
        print(f"model_cleanup: {exc}", file=sys.stderr)
        return 2

    if not args.apply:
        if args.json:
            print(json.dumps({"bundles": [asdict(b) | {"directory": str(b.directory),
                                                       "model": str(b.model)}
                                          for b in bundles],
                              "summary": summary(bundles)}, ensure_ascii=False,
                             indent=2))
        else:
            print(describe_plan(bundles) or "没有找到候选模型。")
            data = summary(bundles)
            print(f"\n共 {data['total']} 个候选 · {human_size(data['total_bytes'])}；"
                  f"可清理 {data['removable']} 个 · "
                  f"{human_size(data['removable_bytes'])}")
        return 0

    selected = list(dict.fromkeys(args.select))
    if args.all:
        for bundle in bundles:
            if not bundle.removable:
                continue
            if bundle.kind == "manual" and not args.include_manual:
                continue
            selected.append(bundle.id)
    if not selected:
        print("model_cleanup: nothing selected; pass --select ID or --all",
              file=sys.stderr)
        return 2
    results = delete(args.weights, selected, registry_root=args.registry,
                     loaded_model=args.loaded, keep=args.keep)
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
