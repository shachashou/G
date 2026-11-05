"""Small, dependency-free provenance/report utilities for offline experiments."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_identity(path: str | Path) -> dict:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                  allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def source_identity() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, timeout=5,
                                       stderr=subprocess.DEVNULL).decode("utf-8").strip()
    try:
        diff = git("diff", "HEAD")
        return {"revision": git("rev-parse", "HEAD"),
                "status": git("status", "--short"),
                "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "status": "unavailable"}
