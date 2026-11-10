"""Small, dependency-free provenance/report utilities for offline experiments."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
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
    """Publish a whole document, tolerating brief Windows sharing locks.

    Each writer owns its temporary file. Never remove the current document
    to work around a lock: readers must keep seeing the last valid version.
    Permanent permission errors still propagate after a bounded retry.
    """
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    pending = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        deadline = time.monotonic() + 2.0
        delay = 0.01
        while True:
            try:
                pending.replace(path)
                break
            except PermissionError as exc:
                remaining = deadline - time.monotonic()
                if os.name != "nt" or exc.winerror not in (5, 32, 33) or remaining <= 0:
                    raise
                time.sleep(min(delay, remaining))
                delay = min(delay * 2, 0.1)
    finally:
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            # Preserve the original error if Windows also holds the temp file.
            pass


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
