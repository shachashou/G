#!/usr/bin/env python3
"""Bounded continuous training orchestration, with replaceable data producers."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
import copy
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
                progress=lambda progress: task.update("data", phase="collecting", metrics=progress))
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
                 trainer=None, evaluator=None, progress=None):
        self.config = config
        self.source_factory, self.trainer, self.evaluator = source_factory, trainer, evaluator
        self.progress = progress
        self.task_id = uuid.uuid4().hex
        self.directory = Path(config.output_root).resolve() / self.task_id
        self.stop_event = threading.Event()
        self.resources = Resources(self.stop_event)
        self.lock = threading.RLock()
        self.state = {"format": "GMVALUE-TRAINING-TASK1", "task_id": self.task_id,
            "directory": str(self.directory), "status": "created", "reason": None,
            "data": {}, "trainer": {}, "evaluation": {}, "rounds": [], "games": 0,
            "consumed_training_samples": 0, "resources": {"cpu_threads": config.cpu_threads,
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

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

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
        self.stop_event.set()

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

    def run(self):
        from history_evaluation import EvaluationConfig, evaluate
        from value_data import load_pool
        from value_train import TrainConfig, train_value
        import torch
        cfg = self.config
        self.started = time.monotonic()
        self.directory.mkdir(parents=True, exist_ok=False)
        finished = threading.Event()
        producer = watchdog = None
        lease = None
        threads_before = torch.get_num_threads()
        try:
            cfg.validate()
            pool_path = Path(cfg.pool).resolve()
            pool_path.parent.mkdir(parents=True, exist_ok=True)
            lease = pool_path.with_suffix(pool_path.suffix + ".training-task.lock")
            # Keep a separate ownership flag: a failed acquisition must never
            # remove another task's lease in finally.
            try:
                with lease.open("x") as handle:
                    handle.write(json.dumps({"task_id": self.task_id, "pid": os.getpid()}))
            except FileExistsError:
                lease = None
                raise ValueError("another training task owns this pool")
            factory = self.source_factory
            if factory is None:
                if cfg.mode == "teacher":
                    factory = TeacherDataSource
                else:
                    from selfplay_data import SelfPlayDataSource
                    factory = SelfPlayDataSource
            source = factory(self)
            provenance = source.preflight()
            (self.directory / "engine").mkdir()
            self.engine = self.directory / "engine" / Path(cfg.engine).name
            shutil.copy2(cfg.engine, self.engine)
            current = ModelRegistry(cfg.registry).resolve()
            write_json(self.directory / "manifest.json", {
                "format": "GMVALUE-TRAINING-TASK1", "config": asdict(cfg),
                "task_id": self.task_id, "supervision": provenance,
                "engine": file_identity(self.engine), "incumbent_at_start": current,
                "implementation": {path: file_identity(ROOT / path) for path in (
                    "tools/training_task.py", "tools/collect.py", "tools/teacher_label.py",
                    "tools/sample_pool.py", "trainer/value_train.py", "trainer/value_data.py",
                    "trainer/value_model.py", "tools/history_evaluation.py", "tools/selfplay_data.py")},
                "environment": {"python": sys.version, "torch": str(torch.__version__),
                    "hip": torch.version.hip, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
                "source": source_identity()})
            with SamplePool(cfg.pool) as pool:
                for key, value in (("capacity", cfg.capacity), ("new_ratio", cfg.new_ratio),
                                   ("validation_games", cfg.validation_games)):
                    pool.set_config(key, value)
                stats = pool.stats(cfg.label_sources())
                if stats["total_validation_samples"] + cfg.min_train_samples > cfg.capacity:
                    raise ValueError("frozen validation data leaves insufficient training capacity")
            torch.set_num_threads(cfg.cpu_threads)
            self.update(status="running", supervision=source.label_source,
                        training_sources=cfg.label_sources(), source_weights=cfg.source_weights())
            watchdog = threading.Thread(target=self._watchdog, args=(finished,), daemon=True)
            watchdog.start()
            producer = threading.Thread(target=self._producer, args=(source,), daemon=True)
            producer.start()
            last_candidate = None
            consumed = 0
            last_attempt = None
            last_round = -float("inf")
            last_change = time.monotonic()
            observed_commit = None
            for index in range(cfg.max_rounds):
                bundle = None
                while not self.stop_event.is_set():
                    with SamplePool(cfg.pool, create=False) as pool:
                        stats = pool.stats(cfg.label_sources())
                    if stats["commit_sequence"] != observed_commit:
                        observed_commit = stats["commit_sequence"]
                        last_change = time.monotonic()
                    ready = self.data_ready(stats)
                    fresh = index == 0 or stats["committed_training_samples"] - consumed >= cfg.min_new_samples
                    interval = time.monotonic() - last_round >= cfg.min_round_interval_s
                    if ready and fresh and interval:
                        candidate_data = load_pool(cfg.pool, train_limit=cfg.training_sample_limit,
                            new_ratio=cfg.new_ratio, seed=cfg.seed + index,
                            label_sources=cfg.label_sources(), source_weights=cfg.source_weights())
                        if (len(candidate_data.train) >= cfg.min_train_samples
                                and len(candidate_data.validation) >= cfg.min_validation_samples
                                and (cfg.mode == "teacher" or candidate_data.identity["train_games"] >= cfg.min_train_games)
                                and candidate_data.identity["samples_sha256"] != last_attempt):
                            bundle = candidate_data
                            break
                    self.update("trainer", phase="waiting_data" if not ready else "waiting_fresh_data",
                        metrics=stats, required_new_samples=cfg.min_new_samples)
                    if self.producer_done.is_set() and not (ready and fresh):
                        self.update(reason="data_limit_or_insufficient_new_samples")
                        break
                    if time.monotonic() - last_change > cfg.max_idle_s:
                        self.update(reason="data_idle_limit")
                        self.stop_event.set()
                        break
                    self.stop_event.wait(0.2)
                if bundle is None or self.stop_event.is_set():
                    break
                last_attempt = bundle.identity["samples_sha256"]
                consumed = bundle.identity["committed_training_samples"]
                self.update(consumed_training_samples=consumed)
                incumbent = ModelRegistry(cfg.registry).resolve()
                if cfg.mode != "teacher" and incumbent is None:
                    raise ValueError("teacher-free training requires an existing incumbent")
                output = self.directory / "candidates" / f"round-{index:05d}"
                parent = last_candidate or incumbent
                checkpoint = last_candidate["checkpoint"] if last_candidate else None
                start_step = last_candidate["steps"] if last_candidate else 0
                lineage = {"task_id": self.task_id, "round": index, "supervision": source.label_source,
                    "mode": cfg.mode, "source_weights": cfg.source_weights(),
                    "parent": parent, "incumbent_at_training": incumbent,
                    "data_commit_sequence": bundle.identity["commit_sequence"]}
                train_config = TrainConfig(pool=cfg.pool, output_dir=str(output), seed=cfg.seed + index,
                    steps=start_step + cfg.train_steps, batch_size=cfg.batch_size, lr=cfg.lr,
                    loss=cfg.loss, device=cfg.device, resume=checkpoint,
                    initial_model=incumbent["path"] if incumbent and not checkpoint else None,
                    lineage=lineage, val_every=max(1, cfg.train_steps // 5),
                    checkpoint_min_interval_s=cfg.checkpoint_min_interval_s,
                    checkpoint_every=max(1, cfg.train_steps // 2), log_every=max(1, cfg.train_steps // 10))
                record = {"round": index, "output": str(output), "lineage": lineage,
                          "data_identity": bundle.identity, "training": None, "evaluation": None}
                with self.lock:
                    self.state["rounds"].append(record)
                write_json(self.directory / f"round-{index:05d}.json", record)
                self.update("trainer", phase="training", round=index)
                summary = (self.trainer or train_value)(train_config, data_bundle=bundle,
                    stop_event=self.stop_event,
                    progress=lambda p: self.update("trainer", phase="training", progress=p))
                with self.lock:
                    record["training"] = {"steps": summary["steps"], "stopped": summary["stopped"],
                        "samples_exposed": summary["samples_exposed"],
                        "export": summary.get("export"), "checkpoint": summary["checkpoint"],
                        "validation": summary["validation"], "device": summary["device"]}
                passed = (summary.get("export") or {}).get("passed", False)
                self.update("trainer", phase="stopped" if summary["stopped"] else "exported" if passed else "numerical_rejected",
                            summary=record["training"])
                write_json(self.directory / f"round-{index:05d}.json", record)
                last_round = time.monotonic()
                if self.stop_event.is_set() or summary["stopped"]:
                    break
                if passed:
                    _, metadata = __import__("value_format").read_value_weights(output / "model.gmvalue")
                    last_candidate = {"model_id": metadata["model_id"],
                        "sha256": file_identity(output / "model.gmvalue")["sha256"],
                        "checkpoint": str(output / "last.pt"), "steps": summary["steps"]}
                    current = ModelRegistry(cfg.registry).resolve()
                    bootstrap = current is None
                    if cfg.mode != "teacher" and bootstrap:
                        raise ValueError("incumbent disappeared before teacher-free evaluation")
                    eval_config = EvaluationConfig(**cfg.evaluation)
                    eval_config = replace(eval_config, seed=cfg.seed + 100000 + index,
                                          pairs=1 if bootstrap else eval_config.pairs)
                    self.update("evaluation", phase="evaluating", round=index, bootstrap=bootstrap)
                    result = (self.evaluator or evaluate)(output / "model.gmvalue", self.engine,
                        registry=cfg.registry, output_root=cfg.evaluations, config=eval_config,
                        bootstrap=bootstrap, stop_event=self.stop_event,
                        progress=lambda p: self.update("evaluation", phase="evaluating", progress=p))
                    with self.lock:
                        record["evaluation"] = {"run": result["run"], "status": result["status"],
                            "statistics": result["statistics"], "publication": result.get("publication"),
                            "publication_error": result.get("publication_error")}
                    if cfg.mode != "teacher":
                        self.event("teacher_exit_evidence", mode=cfg.mode,
                            task_runs_without_teacher=True, training_sources=bundle.identity["label_sources"],
                            history_report=result["run"], promoted=bool(result.get("publication")),
                            relative_progress_proven=bool(result.get("publication")))
                    self.update("evaluation", phase="published" if result.get("publication") else "retained_incumbent",
                        result=record["evaluation"])
                    if result["status"] not in ("completed", "interrupted"):
                        raise ValueError("evaluation aborted: " + result["status"])
                    if result.get("publication_error"):
                        raise ValueError("publication failed: " + result["publication_error"])
                write_json(self.directory / f"round-{index:05d}.json", record)
                self._prune(last_candidate)
                self.event("round_complete", round=index, record=str(self.directory / f"round-{index:05d}.json"))
            if self.producer_error:
                raise ValueError(self.producer_error)
            self.update(status="stopping", terminal_status="stopped" if self.stop_event.is_set() else "completed",
                reason=self.state["reason"] or ("user_stop" if self.stop_event.is_set() else "round_or_data_limit"))
        except Exception as exc:
            self.update(status="failed", reason=str(exc))
        finally:
            self.stop_event.set()
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
                        cleanup_complete=not producer or not producer.is_alive(), current=current)
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
