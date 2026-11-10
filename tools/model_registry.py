"""Atomic, local history of validated GMVALUE1 releases (ticket 08)."""
from __future__ import annotations

from contextlib import contextmanager
import json
import shutil
import uuid
from pathlib import Path

from run_support import file_identity, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "weights" / "registry"


class RegistryError(ValueError):
    pass


class ModelRegistry:
    def __init__(self, root: str | Path = DEFAULT_REGISTRY):
        self.root = Path(root).resolve()

    def read(self) -> dict:
        path = self.root / "state.json"
        if not path.exists():
            return {"format": "GMVALUE-REGISTRY1", "current": None, "versions": []}
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("format") != "GMVALUE-REGISTRY1":
            raise RegistryError("unsupported model registry")
        versions = state.get("versions")
        if (not isinstance(versions, list) or not versions
                or any(not isinstance(v, dict) or v.get("version") != f"v{i}"
                       or not all(isinstance(v.get(k), str) for k in ("model", "sha256", "model_id"))
                       for i, v in enumerate(versions))
                or state.get("current") not in [v["version"] for v in versions]):
            raise RegistryError("invalid current version in registry")
        return state

    def resolve(self, version: str | None = None) -> dict | None:
        state = self.read()
        version = version or state["current"]
        if version is None:
            return None
        for entry in state["versions"]:
            if entry["version"] == version:
                entry = dict(entry)
                model = (self.root / entry["model"]).resolve()
                if not model.is_relative_to(self.root):
                    raise RegistryError("model outside registry")
                if file_identity(model)["sha256"] != entry["sha256"]:
                    raise RegistryError(f"{version}: model hash mismatch")
                entry["path"] = str(model)
                return entry
        raise RegistryError(f"unknown historical version {version!r}")

    @contextmanager
    def _lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".publish.lock"
        try:
            handle = lock.open("x")
        except FileExistsError as exc:
            raise RegistryError("another publication is running; current version retained") from exc
        try:
            with handle:
                yield
        finally:
            lock.unlink()

    def publish(self, run: str | Path, *, expected_current: str | None,
                expected_manifest_sha: str, expected_report_sha: str, stop_event=None) -> dict:
        """Publish an evaluator's sealed result, with compare-and-swap semantics.

        Re-read and validate the full evidence before touching the pointer.
        An orphaned version directory after a crash is harmless: only the
        atomically replaced state.json defines the default version.
        """
        from history_evaluation import verify_publication
        run = Path(run).resolve()
        with self._lock():
            state = self.read()
            if state["current"] != expected_current:
                raise RegistryError("incumbent changed during evaluation; current version retained")
            manifest, report = verify_publication(
                run, expected_manifest_sha, expected_report_sha, stop_event)
            if manifest["incumbent"] != expected_current:
                raise RegistryError("frozen incumbent mismatch")
            if expected_current and expected_current not in manifest["opponents"]:
                raise RegistryError("current incumbent was not challenged")
            for name, opponent in manifest["opponents"].items():
                if self.resolve(name)["sha256"] != opponent["sha256"]:
                    raise RegistryError("historical model changed during evaluation")
            version = f"v{len(state['versions'])}"
            folder = self.root / "versions" / f"{version}-{uuid.uuid4().hex}"
            folder.mkdir(parents=True)
            for name in ("model.gmvalue", "fidelity.json", "training_positions.json"):
                shutil.copy2(run / "candidate" / name, folder / name)
            for name in ("manifest.json", "report.json"):
                shutil.copy2(run / name, folder / name)
            model_sha = file_identity(folder / "model.gmvalue")["sha256"]
            if model_sha != manifest["candidate"]["sha256"]:
                raise RegistryError("published model changed while copying")
            entry = {
                "version": version,
                "model": str((folder / "model.gmvalue").relative_to(self.root)),
                "sha256": model_sha,
                "model_id": manifest["candidate"]["model_id"],
                "previous": expected_current,
                "origin": "initial-validated" if expected_current is None else "history-promotion",
                "run": str(run), "manifest_sha256": expected_manifest_sha,
                "report_sha256": expected_report_sha,
                "created_at": report["finished_at"],
            }
            state["versions"].append(entry)
            state["current"] = version
            if stop_event is not None and stop_event.is_set():
                raise RegistryError("stopped before changing current version")
            write_json(self.root / "state.json", state)
            return self.resolve(version)
