#!/usr/bin/env python3
"""Bounded continuous training orchestration, with replaceable data producers."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
import copy
import ctypes
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer"))
from executor import ExecutorClient
from model_registry import DEFAULT_REGISTRY, ModelRegistry
from run_support import file_identity, source_identity, write_json
from sample_pool import SamplePool

TASK_FORMAT = "GMVALUE-TRAINING-TASK1"
# A round reaches one of these states without needing the scheduler again.
FINAL_ROUND_STATUS = ("published", "retained", "numerical_rejected")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_json(config: "TaskConfig") -> dict:
    """Round-trip through JSON so equality matches the frozen manifest."""
    return json.loads(json.dumps(asdict(config), sort_keys=True, allow_nan=False))


def pid_alive(pid: int) -> bool:
    """True when a process with this PID is still running (best effort)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) can terminate a process on Windows; use the API.
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, int(pid))
        if not handle:
            return False
        try:
            return True
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def save_bundle(path: Path, bundle) -> None:
    """Freezes the exact round data so resume never re-samples the pool."""
    payload = {"format": "GMVALUE-TRAINING-BUNDLE1",
               "train": [asdict(sample) for sample in bundle.train],
               "validation": [asdict(sample) for sample in bundle.validation],
               "identity": bundle.identity,
               "previous_positions": list(bundle.previous_positions),
               "previous_sources": list(bundle.previous_sources)}
    write_json(path, payload)


def load_bundle(path: Path):
    from value_data import DataBundle, Sample
    blob = read_json(path)
    if blob.get("format") != "GMVALUE-TRAINING-BUNDLE1":
        raise ValueError("frozen round data is not a training bundle")

    def samples(rows):
        return [Sample(cells=tuple(int(c) for c in row["cells"]),
                       side=row["side"], target=float(row["target"]),
                       source_id=row["source_id"],
                       label_source=row["label_source"],
                       label_version=row["label_version"],
                       weight=float(row.get("weight", 1.0)))
                for row in rows]

    return DataBundle(train=samples(blob["train"]),
                      validation=samples(blob["validation"]),
                      identity=dict(blob["identity"]),
                      previous_positions=tuple(blob.get("previous_positions", [])),
                      previous_sources=tuple(blob.get("previous_sources", [])))


@dataclass(frozen=True)
class TaskConfig:
    mode: str = "teacher"
    engine: str = str(ROOT / "build/engine/gomoku_nnue.exe")
    pool: str = str(ROOT / "data/pool/pool.sqlite3")
    registry: str = str(DEFAULT_REGISTRY)
    output_root: str = str(ROOT / "weights/teacher-tasks")
    evaluations: str = str(ROOT / "data/evaluations")
    capacity: int = 1000
    validation_games: int = 2
    min_train_samples: int = 32
    min_validation_samples: int = 1
    min_train_games: int = 2
    min_new_samples: int = 32
    max_pending_samples: int = 128
    training_sample_limit: int = 512
    new_ratio: float = 0.5
    train_steps: int = 200
    batch_size: int = 32
    lr: float = 0.001
    loss: str = "mse"
    device: str = "auto"
    cpu_threads: int = 2
    collector_workers: int = 1
    state_write_interval_s: float = 2.0
    checkpoint_min_interval_s: float = 60.0
    min_round_interval_s: float = 10.0
    collection_delay_ms: int = 100
    full_pool_delay_ms: int = 500
    max_rounds: int = 10
    max_games: int = 1000
    max_minutes: float = 60.0
    max_idle_s: float = 120.0
    max_empty_games: int = 50
    keep_candidates: int = 3
    seed: int = 1
    collection: dict = field(default_factory=lambda: {
        "depth": 2, "max_plies": 80, "opening_plies": 3, "opening_region": 7,
        "sample_every": 3, "budget_ms": 200, "vcf_nodes": 5000, "vcf_plies": 4})
    evaluation: dict = field(default_factory=dict)
    teacher_weight: float = 0.25
    selfplay: dict = field(default_factory=dict)

    def label_sources(self):
        return ["selfplay_result"] if self.mode == "selfplay" else (
            ["teacher", "selfplay_result"] if self.mode == "transition" else None)

    def source_weights(self):
        if self.mode == "selfplay":
            return {"selfplay_result": 1.0}
        return {"teacher": self.teacher_weight, "selfplay_result": 1.0} if self.mode == "transition" else {}

    def validate(self):
        if self.mode not in ("teacher", "transition", "selfplay") or not 0 < self.teacher_weight <= 1:
            raise ValueError("invalid training mode or teacher history weight")
        positive = ("capacity", "validation_games", "min_train_samples", "min_validation_samples", "min_train_games",
                    "min_new_samples", "max_pending_samples", "training_sample_limit", "train_steps",
                    "batch_size", "cpu_threads", "collector_workers", "max_rounds", "max_games", "max_empty_games", "keep_candidates")
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1 for name in positive):
            raise ValueError("task count/resource limits must be positive integers")
        if (self.collector_workers > 256 or self.cpu_threads > 256
                or self.capacity < self.min_train_samples + self.min_validation_samples
                or self.max_pending_samples < self.min_new_samples
                or self.training_sample_limit < self.min_train_samples
                or (self.mode != "teacher" and self.training_sample_limit < self.min_train_games)
                or not 0 <= self.new_ratio <= 1 or not 0 < self.lr <= 1
                or not 0 < self.max_minutes < float("inf") or not 0 < self.max_idle_s < float("inf")
                or not 0 <= self.min_round_interval_s < float("inf")
                or not 0 <= self.state_write_interval_s <= 3600
                or not 0 <= self.checkpoint_min_interval_s <= 3600
                or self.collection_delay_ms < 0 or self.full_pool_delay_ms < 0
                or self.device not in ("auto", "cpu", "cuda") or self.loss not in ("mse", "smooth_l1")):
            raise ValueError("invalid task capacity, freshness, training or time limits")
        if self.collection.get("games", 1) != 1:
            raise ValueError("task producer binds one snapshot per game")
        allowed = {"depth", "max_plies", "opening_plies", "opening_region", "sample_every",
                   "budget_ms", "vcf_nodes", "vcf_plies"}
        if set(self.collection) - allowed:
            raise ValueError("unsupported collection overrides")
        # Teacher adapter validation/initialization stays outside teacher-free paths.
        if self.mode != "teacher":
            from selfplay_data import SelfPlayConfig
            SelfPlayConfig(**self.selfplay).validate()
        else:
            from collect import CollectConfig
            c = CollectConfig(**self.collection)
            self._validate_teacher_collection(c)
        from history_evaluation import EvaluationConfig
        EvaluationConfig(**self.evaluation).validate()
        paths = [Path(getattr(self, name)).resolve() for name in ("pool", "registry", "output_root", "evaluations")]
        if len(set(paths)) != len(paths):
            raise ValueError("pool, registry, candidates and evaluation paths must be distinct")

    @staticmethod
    def _validate_teacher_collection(c):
        if (not 1 <= c.depth <= 12 or not 1 <= c.max_plies <= 225
                or not 1 <= c.opening_plies <= 8 or c.opening_region not in range(1, 16, 2)
                or c.opening_plies > c.opening_region ** 2 or c.sample_every < 1
                or c.budget_ms < 1 or c.vcf_nodes < 1 or c.vcf_plies < 1):
            raise ValueError("invalid collection resource limits")


class Resources:
    """Only this task's owned engine/teacher processes may be interrupted."""
    def __init__(self, stop):
        self.stop = stop
        self.lock = threading.Lock()
        self.items = []

    @contextmanager
    def own(self, item):
        with self.lock:
            self.items.append(item)
        try:
            if self.stop.is_set():
                raise InterruptedError("task stopped")
            yield item
        finally:
            try:
                item.close()
            finally:
                with self.lock:
                    if item in self.items:
                        self.items.remove(item)

    def close_all(self):
        with self.lock:
            items = list(self.items)
        errors = []
        for item in items:
            try:
                item.close()
            except Exception as exc:
                errors.append(str(exc))
        return errors


class TeacherDataSource:
    """Teacher-specific supervision lives here, outside the scheduler."""
    label_source = "teacher"

    def __init__(self, task):
        from rapfi_teacher import load_lock
        self.task = task
        self.identity = load_lock()

    def preflight(self):
        from rapfi_teacher import check_install
        installed = check_install(self.identity)
        if not installed.ok:
            raise ValueError("teacher unavailable: " + installed.summary())
        return {"label_source": self.label_source, "teacher": self.identity.describe(),
                "lock": file_identity(ROOT / "tools/rapfi.lock.json")}

    def teacher(self):
        from rapfi_teacher import RapfiTeacher
        return RapfiTeacher(identity=self.identity)

    def produce(self, pool, index):
        from collect import CollectConfig, collect
        task, cfg = self.task, self.task.config
        incumbent = ModelRegistry(cfg.registry).resolve()
        args = ["--value-model", incumbent["path"]] if incumbent else []
        model_id = incumbent["model_id"] if incumbent else "handcrafted"
        model_sha = incumbent["sha256"] if incumbent else None
        version = incumbent["version"] if incumbent else None
        config = CollectConfig(**cfg.collection, games=1, seed=cfg.seed + index,
            model_id=model_id, model_sha256=model_sha, parent_version=version, task_id=task.task_id)
        task.update("data", phase="collecting", game=index + 1,
                    model_id=model_id, model_sha256=model_sha, version=version)
        task.event("game_snapshot", game=index + 1, model_id=model_id,
                   sha256=model_sha, version=version)

        def on_position(payload):
            # collect() sees one game per call; replace its local index with
            # this worker's global game index.
            task.publish_preview({**payload, "kind": "collect",
                                  "game": index + 1, "worker": None})

        with task.resources.own(ExecutorClient(task.engine, *args)) as game, \
                task.resources.own(ExecutorClient(task.engine)) as label, \
                task.resources.own(self.teacher()) as teacher:
            about = game.about()
            if incumbent:
                loaded = about.get("value_model", {})
                if (not about.get("value_model_loaded") or loaded.get("model_id") != model_id
                        or file_identity(incumbent["path"])["sha256"] != model_sha):
                    raise ValueError("collector model identity mismatch")
            elif about.get("eval") != "handcrafted":
                raise ValueError("startup collector must identify handcrafted evaluation")
            teacher.__enter__()
            result = collect(pool, config, game, label, teacher, stop_event=task.stop_event,
                progress=lambda progress: task.update("data", phase="collecting", metrics=progress),
                on_position=on_position)
        return {**result, "collection_model_id": model_id, "version": version}


class TrainingTask:
    """Public task boundary: bounded data workers, one trainer and evaluator.

    A replacement data source supplies preflight()/produce() and label_source;
    produce() may be called concurrently with separate pool connections and
    unique game indices; adapters keep each game's state local. It need not
    construct a teacher. Training/evaluation retain their existing
    independent public APIs. Full pause/restart recovery belongs to ticket 12.
    """
    def __init__(self, config: TaskConfig, *, source_factory=None,
                 trainer=None, evaluator=None, progress=None, preview=None):
        self.config = config
        self.source_factory, self.trainer, self.evaluator = source_factory, trainer, evaluator
        self.progress = progress
        # Optional external live-board consumer. Never persisted, never part of
        # state.json/events.jsonl; see publish_preview().
        self.preview = preview
        self.task_id = uuid.uuid4().hex
        self.directory = Path(config.output_root).resolve() / self.task_id
        self.stop_event = threading.Event()
        # Set means "pause requested"; the scheduler clears it only on resume.
        self.pause_event = threading.Event()
        self.resources = Resources(self.stop_event)
        self.lock = threading.RLock()
        self.state = {"format": TASK_FORMAT, "task_id": self.task_id,
            "directory": str(self.directory), "status": "created", "reason": None,
            "data": {}, "trainer": {}, "evaluation": {}, "rounds": [], "games": 0,
            "next_game": 0, "candidates": [], "last_attempt": None,
            "consumed_training_samples": 0, "resume_count": 0,
            "resources": {"cpu_threads": config.cpu_threads,
                "device": config.device, "collector_workers": config.collector_workers,
                "evaluation_workers": config.evaluation.get("workers", 1),
                "state_write_interval_s": config.state_write_interval_s,
                "checkpoint_min_interval_s": config.checkpoint_min_interval_s,
                "teacher_threads": 1 if config.mode == "teacher" else 0}, "mode": config.mode}
        self.producer_done = threading.Event()
        self.producer_error = None
        self.started = None
        self.next_game = 0
        self.empty_games = 0
        self.production_halted = threading.Event()
        self.last_backpressure_report = -float("inf")
        self.last_state_write = -float("inf")
        self.resume_mode = False
        self.last_candidate = None
        self.last_round_time = -float("inf")
        self._run_lock = threading.Lock()
        self._work_started = False

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def publish_preview(self, payload: dict) -> None:
        """Hand one optional live-board snapshot to the external callback.

        Called from data/evaluation worker threads. The payload is copied so a
        consumer cannot mutate producer state, and delivery is best effort: a
        slow or failing callback must never block, re-enter or fail the task.
        Board cells are deliberately absent from state.json and task events.
        """
        callback = self.preview
        if callback is None:
            return
        snapshot = dict(payload)
        cells = snapshot.get("cells")
        if cells is not None:
            snapshot["cells"] = list(cells)
        try:
            callback(snapshot)
        except Exception:
            # Live previews are advisory; authoritative work continues.
            pass

    def event(self, kind, **values):
        with self.lock:
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": kind, "elapsed_s": round(time.monotonic() - self.started, 3),
                    **values}, ensure_ascii=False, allow_nan=False) + "\n")

    def update(self, actor=None, **values):
        with self.lock:
            old_phase = self.state.get(actor, {}).get("phase") if actor else None
            if actor:
                self.state[actor].update(values)
            else:
                self.state.update(values)
            self.state["elapsed_s"] = round(time.monotonic() - self.started, 3)
            if actor and "phase" in values and values["phase"] != old_phase:
                self.event("phase", actor=actor, phase=values["phase"])
            if ("status" in values or values.get("cleanup_complete") is not None
                    or time.monotonic() - self.last_state_write >= self.config.state_write_interval_s):
                write_json(self.directory / "state.json", self.state)
                self.last_state_write = time.monotonic()
                self.event("state", actor=actor, status=self.state["status"], phase=values.get("phase"))
            value = self.snapshot()
        if self.progress:
            self.progress(value)

    def stop(self):
        self.pause_event.clear()
        self.stop_event.set()
        with self.lock:
            if self.state["status"] in ("created", "running", "pausing",
                                        "paused", "resuming"):
                self.update(status="stopping")

    def pause(self):
        """Requests a pause at the next game / optimizer / evaluation boundary."""
        with self.lock:
            if self.state["status"] not in ("running", "resuming"):
                return False
            self.pause_event.set()
            self.update(status="pausing")
            return True

    def resume(self):
        """Clears a pause request; the scheduler thread keeps running."""
        with self.lock:
            if self.state["status"] not in ("pausing", "paused"):
                return False
            self.pause_event.clear()
            self.update(status="resuming", reason=None)
            return True

    def _wait_while_paused(self, set_state: bool = True) -> bool:
        """Blocks while paused. False means a stop arrived first."""
        while self.pause_event.is_set() and not self.stop_event.is_set():
            if set_state and self.state.get("status") != "paused":
                self.update(status="paused")
            self.pause_event.wait(0.1)
        if self.stop_event.is_set():
            return False
        if set_state and self.state.get("status") in (
                "paused", "pausing", "resuming"):
            reason = None if self.state.get("reason") == "user_pause" else self.state.get("reason")
            self.update(status="running", reason=reason)
        return True

    @classmethod
    def resume_from_directory(cls, directory, *, progress=None,
                              source_factory=None, trainer=None,
                              evaluator=None, config: "TaskConfig | None" = None,
                              preview=None):
        """Reopens a persisted task from its manifest/state, without running it."""
        directory = Path(directory).resolve()
        manifest = read_json(directory / "manifest.json")
        if manifest.get("format") != TASK_FORMAT:
            raise ValueError("not a training task directory")
        if manifest.get("task_id") != directory.name:
            raise ValueError("task manifest id does not match its directory")
        if config is not None and config_json(config) != manifest.get("config"):
            raise ValueError(
                "task configuration differs from the frozen manifest; "
                "resume with the original configuration")
        config = TaskConfig(**manifest["config"])
        task = cls(config, source_factory=source_factory, trainer=trainer,
                   evaluator=evaluator, progress=progress, preview=preview)
        task.task_id = manifest["task_id"]
        task.directory = directory
        task.resume_mode = True
        task._restore_state(manifest)
        return task

    def _restore_state(self, manifest) -> None:
        state_path = self.directory / "state.json"
        stored = read_json(state_path) if state_path.is_file() else {}
        if stored:
            if stored.get("task_id") != self.task_id:
                raise ValueError("stored task state belongs to another task")
            with self.lock:
                self.state.update(stored)
                self.state["format"] = TASK_FORMAT
                self.state["resume_count"] = int(stored.get("resume_count", 0)) + 1
            self.next_game = int(stored.get("next_game", stored.get("games", 0)))
            self.empty_games = int(stored.get("empty_games", 0))
        self.last_candidate = None
        candidates = self.state.get("candidates") or []
        if candidates:
            self.last_candidate = candidates[-1]
        status = self.state.get("status")
        if status == "completed":
            raise ValueError("task already completed; nothing to resume")
        if status == "failed":
            raise ValueError("task failed previously; fix the cause or start a new task")

    def data_ready(self, stats):
        return (stats["train_samples"] >= self.config.min_train_samples
                and stats["validation_samples"] >= self.config.min_validation_samples
                and (self.config.mode == "teacher" or stats["train_games"] >= self.config.min_train_games))

    def _producer(self, source):
        workers = [threading.Thread(target=self._producer_worker, args=(source, i),
                    name=f"collector-{i}", daemon=True) for i in range(self.config.collector_workers)]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            self.update("data", phase="stopped" if self.stop_event.is_set() else "finished")
        finally:
            self.producer_done.set()

    def _producer_worker(self, source, worker):
        cfg = self.config
        try:
            with SamplePool(cfg.pool, create=False) as pool:
                while not self.stop_event.is_set() and not self.production_halted.is_set():
                    # A pause takes effect between whole games, never inside a
                    # game being committed. The scheduler owns the paused state.
                    if not self._wait_while_paused(set_state=False):
                        break
                    while not self.stop_event.is_set() and not self.production_halted.is_set():
                        stats = pool.stats(cfg.label_sources())
                        with self.lock:
                            consumed = self.state["consumed_training_samples"]
                        if (self.data_ready(stats)
                                and stats["committed_training_samples"] - consumed >= cfg.max_pending_samples):
                            with self.lock:
                                if time.monotonic() - self.last_backpressure_report >= .5:
                                    self.last_backpressure_report = time.monotonic()
                                    self.update("data", phase="backpressure", metrics=stats)
                            self.stop_event.wait(0.5)
                            continue
                        break
                    if self.stop_event.is_set() or self.production_halted.is_set():
                        break
                    with self.lock:
                        if self.next_game >= cfg.max_games:
                            break
                        index = self.next_game
                        self.next_game += 1
                        self.state["next_game"] = self.next_game
                    # Persist before the game starts: a crash mid-game must not
                    # reuse the same game index (and source id) on resume.
                    self._persist_state()
                    self.event("game_started", worker=worker, game=index + 1)
                    result = source.produce(pool, index)
                    with self.lock:
                        self.state["games"] += 1
                    if result.get("stopped"):
                        self.event("data_interrupted", game=index + 1, run_id=result.get("run_id"))
                        self.update("data", phase="stopped", metrics=result, totals=pool.stats())
                        break
                    complete = result.get("game_status", "completed") == "completed"
                    self.event("data_commit" if complete else "data_incomplete", game=index + 1,
                        game_status=result.get("game_status"), source_ids=result.get("source_ids", []),
                        run_id=result.get("run_id"), model_id=result.get("collection_model_id"),
                        label_source=source.label_source, commit_sequence=result.get("commit_sequence"))
                    self.update("data", phase="committed" if complete else result["game_status"],
                                metrics=result, totals=pool.stats(cfg.label_sources()))
                    if result.get("total_validation_samples", result["validation_samples"]) + cfg.min_train_samples > cfg.capacity:
                        raise ValueError("frozen validation data leaves insufficient training capacity")
                    with self.lock:
                        self.empty_games = self.empty_games + 1 if not result.get("inserted") else 0
                        stalled = self.empty_games >= cfg.max_empty_games
                    if stalled:
                        self.production_halted.set()
                        self.update("data", phase="stalled", reason="no_effective_samples")
                        break
                    delay = cfg.collection_delay_ms
                    if result["utilization"] >= 0.95:
                        delay = max(delay, cfg.full_pool_delay_ms)
                    if self.stop_event.wait(delay / 1000):
                        break
        except Exception as exc:
            if not self.stop_event.is_set():
                with self.lock:
                    self.producer_error = self.producer_error or str(exc)
                self.update("data", phase="failed", error=str(exc))
                self.stop_event.set()
            else:
                self.update("data", phase="stopped")

    def _watchdog(self, finished):
        while not finished.wait(0.1):
            if time.monotonic() - self.started >= self.config.max_minutes * 60:
                self.update(reason="time_limit")
                self.stop_event.set()
                return

    def _prune(self, last_candidate):
        candidates = sorted((self.directory / "candidates").glob("round-*"))
        protected = Path(last_candidate["checkpoint"]).parent if last_candidate else None
        removable = [p for p in candidates if p != protected]
        for folder in removable[:-self.config.keep_candidates]:
            target = folder.resolve()
            parent = (self.directory / "candidates").resolve()
            if target.parent != parent or folder.is_symlink():
                raise ValueError("unsafe candidate retention path")
            shutil.rmtree(target)

    def _persist_state(self) -> None:
        with self.lock:
            self.state["next_game"] = int(self.next_game)
            self.state["empty_games"] = int(self.empty_games)
            write_json(self.directory / "state.json", self.state)
            self.last_state_write = time.monotonic()

    def _write_round(self, record) -> None:
        write_json(self.directory / f"round-{int(record['round']):05d}.json", record)

    def _round_record(self, index):
        for record in self.state["rounds"]:
            if record.get("round") == index:
                return record
        return None

    def _acquire_lease(self, pool_path):
        lease = pool_path.with_suffix(pool_path.suffix + ".training-task.lock")
        reclaimed = None
        for _ in range(2):
            try:
                with lease.open("x") as handle:
                    handle.write(json.dumps({"task_id": self.task_id, "pid": os.getpid()}))
                if reclaimed:
                    self.event("stale_lease_reclaimed", previous=reclaimed)
                return lease
            except FileExistsError:
                try:
                    info = read_json(lease)
                except (OSError, ValueError) as exc:
                    raise ValueError("pool lease is unreadable") from exc
                pid = int(info.get("pid", -1))
                if pid_alive(pid):
                    raise ValueError("another training task owns this pool")
                # Only the same persisted task may take over its own stale
                # lease. An unrelated lease is never removed speculatively.
                if not self.resume_mode or info.get("task_id") != self.task_id:
                    raise ValueError("another training task owns this pool")
                reclaimed = info
                lease.unlink()
        raise ValueError("could not acquire the pool lease")

    def _prepare_frozen_engine(self, manifest, cfg):
        self.engine = self.directory / "engine" / Path(cfg.engine).name
        expected = manifest.get("engine", {})
        ok = (self.engine.is_file()
              and file_identity(self.engine)["sha256"] == expected.get("sha256"))
        if not ok:
            source = Path(cfg.engine)
            if (source.is_file()
                    and file_identity(source)["sha256"] == expected.get("sha256")):
                self.engine.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, self.engine)
                ok = True
        if not ok:
            raise ValueError("frozen task engine is missing or changed")

    def _check_supervision(self, manifest, source):
        frozen = manifest.get("supervision") or {}
        fresh = source.preflight()
        for key in ("label_source", "teacher", "lock", "label_version",
                    "teacher_calls", "teacher_history_used", "exploration",
                    "value_contract"):
            if key in frozen and frozen[key] != fresh.get(key):
                raise ValueError("supervision configuration changed since the task started")
        self.update(supervision=fresh.get("label_source", frozen.get("label_source")))

    def _wait_for_bundle(self, index, cfg, load_pool):
        last_attempt = self.state.get("last_attempt")
        observed_commit = None
        last_change = time.monotonic()
        while not self.stop_event.is_set():
            if not self._wait_while_paused():
                return None, None
            with SamplePool(cfg.pool, create=False) as pool:
                stats = pool.stats(cfg.label_sources())
            if stats["commit_sequence"] != observed_commit:
                observed_commit = stats["commit_sequence"]
                last_change = time.monotonic()
            ready = self.data_ready(stats)
            fresh = (last_attempt is None
                     or stats["committed_training_samples"]
                     - self.state["consumed_training_samples"] >= cfg.min_new_samples)
            interval = (time.monotonic() - self.last_round_time
                        >= cfg.min_round_interval_s)
            if ready and fresh and interval:
                candidate_data = load_pool(
                    cfg.pool, train_limit=cfg.training_sample_limit,
                    new_ratio=cfg.new_ratio, seed=cfg.seed + index,
                    label_sources=cfg.label_sources(),
                    source_weights=cfg.source_weights())
                if (len(candidate_data.train) >= cfg.min_train_samples
                        and len(candidate_data.validation) >= cfg.min_validation_samples
                        and (cfg.mode == "teacher"
                             or candidate_data.identity["train_games"] >= cfg.min_train_games)
                        and candidate_data.identity["samples_sha256"] != last_attempt):
                    return candidate_data, None
            self.update("trainer",
                        phase="waiting_data" if not ready else "waiting_fresh_data",
                        metrics=stats, required_new_samples=cfg.min_new_samples)
            if self.producer_done.is_set() and not (ready and fresh):
                return None, "data_limit_or_insufficient_new_samples"
            if time.monotonic() - last_change > cfg.max_idle_s:
                return None, "data_idle_limit"
            self.stop_event.wait(0.2)
        return None, None

    def _create_round(self, index, bundle, cfg, source):
        incumbent = ModelRegistry(cfg.registry).resolve()
        if cfg.mode != "teacher" and incumbent is None:
            raise ValueError("teacher-free training requires an existing incumbent")
        output = self.directory / "candidates" / f"round-{index:05d}"
        parent = self.last_candidate or incumbent
        checkpoint = self.last_candidate["checkpoint"] if self.last_candidate else None
        start_step = self.last_candidate["steps"] if self.last_candidate else 0
        lineage = {"task_id": self.task_id, "round": index,
                   "supervision": source.label_source, "mode": cfg.mode,
                   "source_weights": cfg.source_weights(), "parent": parent,
                   "incumbent_at_training": incumbent,
                   "data_commit_sequence": bundle.identity["commit_sequence"]}
        bundle_path = self.directory / "rounds" / f"round-{index:05d}-bundle.json"
        save_bundle(bundle_path, bundle)
        record = {"round": index, "status": "new", "output": str(output),
                  "bundle": str(bundle_path), "lineage": lineage,
                  "data_identity": bundle.identity, "training": None,
                  "evaluation": None, "candidate": None,
                  "start_step": start_step,
                  "initial_model": incumbent["path"] if incumbent and not checkpoint else None,
                  "attempts": {"training": 0, "evaluation": 0}}
        with self.lock:
            self.state["rounds"].append(record)
            self.state["rounds"].sort(key=lambda item: item["round"])
            self.state["last_attempt"] = bundle.identity["samples_sha256"]
            self.state["consumed_training_samples"] = bundle.identity["committed_training_samples"]
        self._write_round(record)
        self._persist_state()
        self.update("trainer", phase="data_frozen", round=index,
                    metrics=bundle.summary())
        return record

    def _train_round(self, index, cfg, record, bundle):
        from value_train import TrainConfig, train_value
        training = record.get("training")
        if training and not training.get("stopped"):
            if record.get("candidate"):
                self.last_candidate = record["candidate"]
                with self.lock:
                    candidates = [item for item in self.state.get("candidates", [])
                                  if item.get("round") != index]
                    candidates.append(record["candidate"])
                    candidates.sort(key=lambda item: item.get("round", 0))
                    self.state["candidates"] = candidates
                return "trained"
            if not record.get("candidate"):
                # Crash window after the training record was persisted but
                # before the candidate identity was. Recover from the export.
                output = Path(record["output"])
                model_path = output / "model.gmvalue"
                if not model_path.is_file():
                    raise ValueError("trained round has no exported candidate")
                from value_format import read_value_weights
                _, metadata = read_value_weights(model_path)
                candidate = {"model_id": metadata["model_id"],
                             "sha256": file_identity(model_path)["sha256"],
                             "checkpoint": str(output / "last.pt"),
                             "steps": training["steps"], "round": index}
                record["candidate"] = candidate
                record["status"] = "trained"
                self.last_candidate = candidate
                with self.lock:
                    candidates = [item for item in self.state.get("candidates", [])
                                  if item.get("round") != index]
                    candidates.append(candidate)
                    candidates.sort(key=lambda item: item.get("round", 0))
                    self.state["candidates"] = candidates
                self._write_round(record)
                self._persist_state()
            return "trained"
        checkpoint = None
        if training and training.get("stopped"):
            checkpoint = (training.get("checkpoint") or {}).get("last")
            if not checkpoint or not Path(checkpoint).is_file():
                raise ValueError("paused round has no usable training checkpoint")
        attempts = record.setdefault("attempts", {"training": 0, "evaluation": 0})
        attempts["training"] = int(attempts.get("training", 0)) + 1
        train_config = TrainConfig(
            pool=cfg.pool, output_dir=record["output"], seed=cfg.seed + index,
            steps=int(record["start_step"]) + cfg.train_steps,
            batch_size=cfg.batch_size, lr=cfg.lr, loss=cfg.loss,
            device=cfg.device, resume=checkpoint, strict_resume=True,
            initial_model=None if checkpoint else record.get("initial_model"),
            lineage=record["lineage"], val_every=max(1, cfg.train_steps // 5),
            checkpoint_min_interval_s=cfg.checkpoint_min_interval_s,
            checkpoint_every=max(1, cfg.train_steps // 2),
            log_every=max(1, cfg.train_steps // 10))
        self.update("trainer", phase="training", round=index,
                    attempt=attempts["training"])
        summary = (self.trainer or train_value)(
            train_config, data_bundle=bundle, stop_event=self.stop_event,
            pause_event=self.pause_event,
            progress=lambda progress: self.update(
                "trainer", phase="training", progress=progress))
        with self.lock:
            record["training"] = {
                "steps": summary["steps"], "stopped": summary["stopped"],
                "stop_reason": summary.get("stop_reason"),
                "samples_exposed": summary["samples_exposed"],
                "export": summary.get("export"),
                "checkpoint": summary["checkpoint"],
                "validation": summary["validation"],
                "device": summary["device"]}
            record["checkpoint"] = summary["checkpoint"]["last"]
        if (summary.get("stop_reason") == "paused"
                and not self.stop_event.is_set()):
            record["status"] = "paused"
            self._write_round(record)
            self.update("trainer", phase="paused", summary=record["training"])
            self.update(status="paused", reason="user_pause")
            return "paused"
        if summary.get("stopped") or self.stop_event.is_set():
            record["status"] = "stopped"
            self._write_round(record)
            self.update("trainer", phase="stopped", summary=record["training"])
            return "stopped"
        passed = (summary.get("export") or {}).get("passed", False)
        self.update("trainer",
                    phase="exported" if passed else "numerical_rejected",
                    summary=record["training"])
        self._write_round(record)
        if not passed:
            record["status"] = "numerical_rejected"
            self._write_round(record)
            return "done"
        from value_format import read_value_weights
        _, metadata = read_value_weights(
            Path(record["output"]) / "model.gmvalue")
        candidate = {"model_id": metadata["model_id"],
                     "sha256": file_identity(
                         Path(record["output"]) / "model.gmvalue")["sha256"],
                     "checkpoint": str(Path(record["output"]) / "last.pt"),
                     "steps": summary["steps"], "round": index}
        record["candidate"] = candidate
        record["status"] = "trained"
        self.last_candidate = candidate
        with self.lock:
            candidates = [item for item in self.state.get("candidates", [])
                          if item.get("round") != index]
            candidates.append(candidate)
            candidates.sort(key=lambda item: item.get("round", 0))
            self.state["candidates"] = candidates
        self._write_round(record)
        self._persist_state()
        return "trained"

    def _evaluate_round(self, index, cfg, record):
        from history_evaluation import EvaluationConfig, evaluate
        candidate = record.get("candidate")
        if not candidate:
            raise ValueError("round has no exported candidate to evaluate")
        store = ModelRegistry(cfg.registry)
        current = store.resolve()
        if current and current["sha256"] == candidate["sha256"]:
            record["evaluation"] = {
                "run": current.get("run") or (record.get("evaluation") or {}).get("run"),
                "status": "completed",
                "statistics": (record.get("evaluation") or {}).get("statistics", {}),
                "publication": current,
                "already_current": True}
            record["status"] = "published"
            self._write_round(record)
            self.update("evaluation", phase="already_current", round=index,
                        result=record["evaluation"])
            return "done"
        bootstrap = current is None
        if cfg.mode != "teacher" and bootstrap:
            raise ValueError("incumbent disappeared before teacher-free evaluation")
        eval_config = EvaluationConfig(**cfg.evaluation)
        eval_config = replace(
            eval_config, seed=cfg.seed + 100000 + index,
            pairs=1 if bootstrap else eval_config.pairs)
        previous = record.get("evaluation") or {}
        resume_run = None
        if previous.get("status") == "interrupted":
            candidate_run = Path(previous.get("run", ""))
            if candidate_run.is_dir():
                resume_run = candidate_run
        attempts = record.setdefault("attempts", {"training": 0, "evaluation": 0})
        attempts["evaluation"] = int(attempts.get("evaluation", 0)) + 1
        eval_stop = threading.Event()
        eval_done = threading.Event()

        def watch():
            while not eval_done.wait(0.1):
                if self.stop_event.is_set() or self.pause_event.is_set():
                    eval_stop.set()
                    return

        watcher = threading.Thread(target=watch, daemon=True,
                                   name="evaluation-pause")
        watcher.start()
        self.update("evaluation", phase="evaluating", round=index,
                    bootstrap=bootstrap, attempt=attempts["evaluation"],
                    resumed=bool(resume_run))
        try:
            kwargs = dict(
                registry=cfg.registry, output_root=cfg.evaluations,
                config=eval_config, bootstrap=bootstrap, stop_event=eval_stop,
                progress=lambda progress: self.update(
                    "evaluation", phase="evaluating", progress=progress))
            if resume_run is not None:
                kwargs["resume_run"] = resume_run
            if self.evaluator is None:
                # Only the default history_evaluation.evaluate accepts the
                # optional preview keyword; custom evaluator callables are
                # called exactly as before.
                kwargs["on_position"] = lambda payload: self.publish_preview(
                    {**payload, "kind": "evaluation"})
            result = (self.evaluator or evaluate)(
                Path(record["output"]) / "model.gmvalue", self.engine, **kwargs)
        finally:
            eval_done.set()
            watcher.join(timeout=2)
        with self.lock:
            record["evaluation"] = {
                "run": result["run"], "status": result["status"],
                "statistics": result["statistics"],
                "publication": result.get("publication"),
                "publication_error": result.get("publication_error"),
                "resumed": bool(resume_run),
                "attempt": attempts["evaluation"]}
        self._write_round(record)
        self._persist_state()
        if result["status"] == "interrupted":
            if self.stop_event.is_set():
                return "stopped"
            if self.pause_event.is_set():
                record["status"] = "paused"
                self._write_round(record)
                self.update(status="paused", reason="user_pause")
                return "paused"
            raise ValueError("evaluation interrupted without a pause or stop request")
        if result["status"] != "completed":
            raise ValueError("evaluation aborted: " + result["status"])
        if result.get("publication_error"):
            raise ValueError("publication failed: " + result["publication_error"])
        if cfg.mode != "teacher":
            self.event("teacher_exit_evidence", mode=cfg.mode,
                       task_runs_without_teacher=True,
                       training_sources=record["data_identity"]["label_sources"],
                       history_report=result["run"],
                       promoted=bool(result.get("publication")),
                       relative_progress_proven=bool(result.get("publication")))
        record["status"] = "published" if result.get("publication") else "retained"
        self._write_round(record)
        self.update("evaluation",
                    phase="published" if result.get("publication") else "retained_incumbent",
                    result=record["evaluation"])
        return "done"

    def _run_round(self, index, cfg, source):
        from value_data import load_pool
        record = self._round_record(index)
        if record is None:
            bundle, reason = self._wait_for_bundle(index, cfg, load_pool)
            if bundle is None:
                return reason or "stopped"
            record = self._create_round(index, bundle, cfg, source)
        else:
            bundle = load_bundle(Path(record["bundle"]))
        if record.get("status") in FINAL_ROUND_STATUS:
            return "done"
        trained = self._train_round(index, cfg, record, bundle)
        if trained != "trained":
            return trained
        return self._evaluate_round(index, cfg, record)

    def run(self):
        # One scheduler thread owns one task directory. A duplicate start on
        # the same object waits for that run and returns its final snapshot
        # instead of racing on the directory and state files.
        if not self._run_lock.acquire(blocking=False):
            with self._run_lock:
                return self.snapshot()
        try:
            return self._run()
        finally:
            self._run_lock.release()

    def _run(self):
        from value_data import load_pool
        import torch
        cfg = self.config
        self.started = time.monotonic()
        finished = threading.Event()
        producer = watchdog = None
        lease = None
        threads_before = torch.get_num_threads()
        try:
            cfg.validate()
            pool_path = Path(cfg.pool).resolve()
            pool_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.resume_mode and self.directory.exists():
                # A freshly generated id must never inherit an existing task
                # directory (e.g. an aborted duplicate start). Move to a new id
                # before taking the pool lease.
                previous = self.task_id
                self.task_id = uuid.uuid4().hex
                self.directory = Path(cfg.output_root).resolve() / self.task_id
                with self.lock:
                    self.state["task_id"] = self.task_id
                    self.state["directory"] = str(self.directory)
                self._directory_collision = previous
            lease = self._acquire_lease(pool_path)
            factory = self.source_factory
            if factory is None:
                if cfg.mode == "teacher":
                    factory = TeacherDataSource
                else:
                    from selfplay_data import SelfPlayDataSource
                    factory = SelfPlayDataSource
            source = factory(self)
            if self.resume_mode:
                manifest_path = self.directory / "manifest.json"
                if not manifest_path.is_file():
                    raise ValueError("task manifest is missing; cannot resume")
                manifest = read_json(manifest_path)
                if manifest.get("format") != TASK_FORMAT:
                    raise ValueError("not a training task directory")
                if manifest.get("task_id") != self.task_id:
                    raise ValueError("task manifest id does not match the directory")
                if manifest.get("config") != config_json(cfg):
                    raise ValueError(
                        "task configuration differs from the frozen manifest; "
                        "resume with the original configuration")
                self._prepare_frozen_engine(manifest, cfg)
                self._check_supervision(manifest, source)
                with SamplePool(cfg.pool) as pool:
                    removed = pool.abort_staged_games()
                    if removed:
                        self.event("stale_staging_cleaned", games=removed)
                    for key, value in (("capacity", cfg.capacity),
                                       ("new_ratio", cfg.new_ratio),
                                       ("validation_games", cfg.validation_games)):
                        pool.set_config(key, value)
                    stats = pool.stats(cfg.label_sources())
                    if (stats["total_validation_samples"] + cfg.min_train_samples
                            > cfg.capacity):
                        raise ValueError(
                            "frozen validation data leaves insufficient training capacity")
                self.update(status="resuming", reason=None)
            else:
                self.directory.mkdir(parents=True, exist_ok=False)
                previous = getattr(self, "_directory_collision", None)
                if previous:
                    self.event("directory_collision_reassigned",
                               previous=previous, current=self.task_id)
                provenance = source.preflight()
                (self.directory / "engine").mkdir()
                self.engine = self.directory / "engine" / Path(cfg.engine).name
                shutil.copy2(cfg.engine, self.engine)
                current = ModelRegistry(cfg.registry).resolve()
                write_json(self.directory / "manifest.json", {
                    "format": TASK_FORMAT, "config": asdict(cfg),
                    "task_id": self.task_id, "supervision": provenance,
                    "engine": file_identity(self.engine),
                    "incumbent_at_start": current,
                    "implementation": {path: file_identity(ROOT / path) for path in (
                        "tools/training_task.py", "tools/collect.py",
                        "tools/teacher_label.py", "tools/sample_pool.py",
                        "trainer/value_train.py", "trainer/value_data.py",
                        "trainer/value_model.py", "tools/history_evaluation.py",
                        "tools/selfplay_data.py")},
                    "environment": {"python": sys.version,
                                    "torch": str(torch.__version__),
                                    "hip": torch.version.hip,
                                    "cuda": torch.version.cuda,
                                    "gpu": (torch.cuda.get_device_name(0)
                                            if torch.cuda.is_available() else None)},
                    "source": source_identity()})
                with SamplePool(cfg.pool) as pool:
                    for key, value in (("capacity", cfg.capacity),
                                       ("new_ratio", cfg.new_ratio),
                                       ("validation_games", cfg.validation_games)):
                        pool.set_config(key, value)
                    stats = pool.stats(cfg.label_sources())
                    if (stats["total_validation_samples"] + cfg.min_train_samples
                            > cfg.capacity):
                        raise ValueError(
                            "frozen validation data leaves insufficient training capacity")
            torch.set_num_threads(cfg.cpu_threads)
            self._work_started = True
            self._persist_state()
            self.update(status="running", reason=None,
                        supervision=source.label_source,
                        training_sources=cfg.label_sources(),
                        source_weights=cfg.source_weights())
            watchdog = threading.Thread(target=self._watchdog,
                                        args=(finished,), daemon=True)
            watchdog.start()
            producer = threading.Thread(target=self._producer,
                                        args=(source,), daemon=True)
            producer.start()
            index = 0
            while index < len(self.state["rounds"]):
                record = self.state["rounds"][index]
                if record.get("round") != index:
                    raise ValueError("round records are not contiguous")
                if record.get("status") in FINAL_ROUND_STATUS:
                    index += 1
                    continue
                break
            while index < cfg.max_rounds:
                if self.stop_event.is_set():
                    break
                if not self._wait_while_paused():
                    break
                record = self._round_record(index)
                if record and record.get("status") in FINAL_ROUND_STATUS:
                    index += 1
                    continue
                outcome = self._run_round(index, cfg, source)
                if outcome == "paused":
                    continue
                if outcome == "stopped":
                    break
                if outcome in ("data_limit_or_insufficient_new_samples",
                               "data_idle_limit"):
                    self.update(reason=outcome)
                    break
                self.last_round_time = time.monotonic()
                self._prune(self.last_candidate)
                self.event("round_complete", round=index,
                           record=str(self.directory / f"round-{index:05d}.json"))
                index += 1
            if self.producer_error:
                raise ValueError(self.producer_error)
            self.update(status="stopping",
                        terminal_status="stopped" if self.stop_event.is_set()
                        else "completed",
                        reason=self.state["reason"] or (
                            "user_stop" if self.stop_event.is_set()
                            else "round_or_data_limit"))
        except Exception as exc:
            if self.resume_mode and not self._work_started:
                # A resume that failed during identity/compatibility checks
                # must leave the paused/stopped task intact for a later retry.
                with self.lock:
                    previous_status = self.state.get("status", "paused")
                self.update(status=previous_status,
                            reason="resume failed: " + str(exc),
                            resume_error=str(exc))
            else:
                self.update(status="failed", reason=str(exc))
        finally:
            self.stop_event.set()
            self.pause_event.clear()
            finished.set()
            cleanup_errors = self.resources.close_all()
            if producer:
                producer.join(timeout=15)
                if producer.is_alive():
                    self.update(status="failed", reason="data worker did not stop")
            if watchdog:
                watchdog.join(timeout=2)
            if self.producer_error and self.state["status"] != "failed":
                self.update(status="failed", reason=self.producer_error)
            if lease:
                try:
                    if json.loads(lease.read_text())["task_id"] == self.task_id:
                        lease.unlink()
                    else:
                        cleanup_errors.append("pool lease ownership changed")
                except (OSError, ValueError, KeyError) as exc:
                    cleanup_errors.append(str(exc))
            torch.set_num_threads(threads_before)
            try:
                current = ModelRegistry(cfg.registry).resolve()
            except (ValueError, OSError) as exc:
                cleanup_errors.append(str(exc))
                current = None
            if cleanup_errors:
                self.update(status="failed", reason="; ".join(cleanup_errors))
            final_status = self.state.get("terminal_status", self.state["status"])
            self.update(status="failed" if self.state["status"] == "failed" else final_status,
                        cleanup_complete=not producer or not producer.is_alive(),
                        current=current)
        return self.snapshot()

def main():
    import argparse
    import signal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON object with TaskConfig fields")
    args = parser.parse_args()
    try:
        config = TaskConfig(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else TaskConfig()
        task = TrainingTask(config)
        signal.signal(signal.SIGINT, lambda *_: task.stop())
        result = task.run()
    except (ValueError, OSError, TypeError) as exc:
        print(f"training_task: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
