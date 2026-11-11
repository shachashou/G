"""Teacher-free, authoritative self-play with terminal-result supervision."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

from executor import ExecutorClient
from history_evaluation import MatchEngine, position_payload, verify_about
from match_runner import EngineFailure, EngineSpec
from model_registry import ModelRegistry
from run_support import file_identity, write_json

RESULTS = {"black_win": 1.0, "draw": 0.0, "white_win": -1.0}


@dataclass(frozen=True)
class SelfPlayConfig:
    max_plies: int = 225
    opening_plies: int = 4
    opening_region: int = 7
    epsilon: float = 0.1
    exploration_plies: int = 16
    sample_every: int = 2
    movetime_ms: int = 30
    reply_grace_ms: int = 1000
    game_timeout_s: float = 60.0
    vcf_policy: str = "root"
    vcf_nodes: int = 5000
    filter_vcf: bool = True
    filter_nodes: int = 1000
    filter_plies: int = 4

    def validate(self):
        if (not 1 <= self.max_plies <= 225 or not 0 <= self.opening_plies <= 8
                or self.opening_region not in range(1, 16, 2)
                or self.opening_plies > self.opening_region ** 2
                or not 0 <= self.epsilon <= 1 or not 0 <= self.exploration_plies <= 225
                or self.sample_every < 1 or self.movetime_ms < 1 or self.reply_grace_ms < 0
                or not 0 < self.game_timeout_s < float("inf")
                or self.vcf_policy not in ("off", "root", "root_and_children")
                or min(self.vcf_nodes, self.filter_nodes, self.filter_plies) < 1):
            raise ValueError("invalid self-play exploration/search/resource limits")


class SelfPlayDataSource:
    label_source = "selfplay_result"

    def __init__(self, task):
        self.task = task
        self.config = SelfPlayConfig(**task.config.selfplay)

    def preflight(self):
        self.config.validate()
        incumbent = ModelRegistry(self.task.config.registry).resolve()
        if incumbent is None:
            raise ValueError("self-play requires an existing validated incumbent; establish v0 first")
        return {"label_source": self.label_source, "label_version": "selfplay-result-v1",
                "teacher_calls": 0, "teacher_history_used": self.task.config.mode == "transition",
                "exploration": asdict(self.config), "incumbent": incumbent,
                "value_contract": "black-absolute-tanh-v1"}

    def produce(self, pool, index):
        task, cfg = self.task, self.config
        incumbent = ModelRegistry(task.config.registry).resolve()
        if incumbent is None:
            raise ValueError("self-play incumbent is missing")
        source_id = f"selfplay-result-v1-{task.task_id}-{index:06d}"
        path = task.directory / "games" / f"{index:06d}.json"
        record = {"format": "GMVALUE-SELFPLAY-GAME1", "source_id": source_id,
                  "task_id": task.task_id, "model": incumbent, "config": asdict(cfg),
                  "seed": task.config.seed + index, "moves": [], "sampled_plies": [],
                  "filtered_plies": [],
                  "status": "running", "final_result": None, "result_value": None,
                  "teacher_calls": 0, "anomalies": []}
        run_id = pool.start_run({"strategy_version": "selfplay-result-v1",
            "source_id": source_id, "task_id": task.task_id, "model": incumbent,
            "config": asdict(cfg), "seed": record["seed"], "game_record": str(path),
            "mode": task.config.mode, "label_source": self.label_source})
        started = time.monotonic()
        rng = random.Random(record["seed"])
        positions = []
        filters = {}
        generated = inserted = 0
        game_id = None
        record_written = False
        # Live snapshots stay best effort: TrainingTask.publish_preview copies
        # the payload and never stores it. A task-like object without the hook
        # simply gets no previews.
        publish_preview = getattr(task, "publish_preview", None)
        task.event("game_snapshot", game=index + 1, model_id=incumbent["model_id"],
                   version=incumbent["version"], sha256=incumbent["sha256"], teacher_calls=0)
        task.update("data", phase="selfplay", game=index + 1,
            model_id=incumbent["model_id"], version=incumbent["version"], teacher_calls=0)

        def discard(reason):
            filters[reason] = filters.get(reason, 0) + 1

        def stopped():
            if task.stop_event.is_set():
                raise InterruptedError("self-play stopped")
            if time.monotonic() - started >= cfg.game_timeout_s:
                raise EngineFailure("timeout", "self-play game deadline exceeded")

        try:
            args = [str(task.engine), "--value-model", incumbent["path"],
                    "--movetime", str(cfg.movetime_ms), "--vcf-policy", cfg.vcf_policy,
                    "--vcf-nodes", str(cfg.vcf_nodes)]
            spec = EngineSpec("selfplay", " ".join(args), args)
            with task.resources.own(ExecutorClient(task.engine)) as referee, \
                    task.resources.own(MatchEngine(spec, (cfg.movetime_ms + cfg.reply_grace_ms) / 1000,
                                                   task.stop_event)) as engine:
                engine.start(2, cfg.movetime_ms)
                engine.send("INFO thread_num 1")
                engine.send("ABOUT")
                record["about"] = engine.recv_line()
                verify_about(record["about"], incumbent)
                if file_identity(incumbent["path"])["sha256"] != incumbent["sha256"]:
                    raise EngineFailure("identity-mismatch", "incumbent file changed")
                state = referee.new_game("renju")
                for ply in range(cfg.max_plies):
                    stopped()
                    explore = ply < cfg.opening_plies or (ply < cfg.exploration_plies and rng.random() < cfg.epsilon)
                    if explore:
                        legal = referee.legal_moves()["moves"]
                        region = [p for p in legal if abs(p[0]-7) <= cfg.opening_region//2
                                  and abs(p[1]-7) <= cfg.opening_region//2]
                        x, y = rng.choice(region or legal)
                        kind = "opening" if ply < cfg.opening_plies else "epsilon"
                    else:
                        engine.move_timeout = min((cfg.movetime_ms + cfg.reply_grace_ms) / 1000,
                            max(.001, cfg.game_timeout_s - (time.monotonic()-started)))
                        x, y = engine.move(state["cells"])
                        kind = "search"
                    response = referee.play(x, y)
                    if not response.get("ok"):
                        record["rejected_move"] = [x, y]
                        raise EngineFailure("illegal-move", str(response.get("reason", response)))
                    state = response
                    record["moves"].append({"move": [x, y], "kind": kind})
                    if publish_preview is not None:
                        publish_preview(position_payload(
                            "selfplay", state, (x, y), game=index + 1,
                            model_id=incumbent["model_id"]))
                    if state["result"] != "ongoing":
                        record["final_result"] = state["result"]
                        record["result_value"] = RESULTS[state["result"]]
                        record["status"] = "completed"
                        break
                    if ply >= cfg.opening_plies and (ply - cfg.opening_plies) % cfg.sample_every == 0:
                        generated += 1
                        status = "not_filtered"
                        if cfg.filter_vcf:
                            tactical = referee.vcf(max_nodes=cfg.filter_nodes, max_attack_plies=cfg.filter_plies)
                            status = tactical.get("vcf", {}).get("status", "invalid_input")
                            if status == "proven":
                                record["filtered_plies"].append({"ply": ply + 1,
                                    "reason": "vcf_proven", "evidence": tactical.get("vcf")})
                                discard("vcf_proven")
                                continue
                        positions.append((list(state["cells"]), state["side_to_move"], ply + 1, status))
                if record["final_result"] is None:
                    record["status"] = "truncated"
                    discard("truncated")
                else:
                    # The whole terminal game is replayed before any result is
                    # made trainable. No model score is used as supervision.
                    replay = referee.replay([m["move"] for m in record["moves"]])
                    if not replay.get("ok") or replay.get("result") != record["final_result"]:
                        raise ValueError("self-play terminal replay mismatch")
                    stopped()
                    game_id = pool.begin_game(source_id, kind="selfplay", model_id=incumbent["model_id"])
                    records = []
                    for cells, side, ply, tactical in positions:
                        records.append({
                            "board": bytes(cells), "side_to_move": side, "rule": "renju",
                            "label_source": self.label_source, "label_version": "selfplay-result-v1",
                            "tactical_status": tactical, "result_value": record["result_value"],
                            "value": None, "raw_eval": None, "raw_score": None,
                            "raw_perspective": "black_absolute", "normalization": "terminal-result",
                            "normalization_param": 1.0})
                    stopped()
                    statuses = pool.add_samples(game_id, records)
                    for (_, _, ply, _), status in zip(positions, statuses):
                        record["sampled_plies"].append({"ply": ply, "status": status})
                        inserted += status == "inserted"
                    record["elapsed_s"] = round(time.monotonic() - started, 4)
                    write_json(path, record)
                    record_written = True
                    stopped()
                    pool.commit_game(game_id, final_result=record["final_result"], validation_group=self.label_source)
                    game_id = None
        except InterruptedError:
            record.update(status="interrupted", final_result=None, result_value=None)
        except Exception as exc:
            if task.stop_event.is_set():
                record.update(status="interrupted", final_result=None, result_value=None)
            else:
                record.update(status="failed", final_result=None, result_value=None)
                record["anomalies"].append({"kind": getattr(exc, "kind", "error"), "detail": str(exc)})
                raise
        finally:
            if game_id is not None:
                pool.abort_game(game_id)
            if not record_written or record["status"] != "completed":
                record["elapsed_s"] = round(time.monotonic() - started, 4)
                write_json(path, record)
            pool.record_observations(generated, filters)
            pool.finish_run(run_id, generated=generated, valid=len(positions),
                inserted=inserted if record["status"] == "completed" else 0, filters=filters,
                elapsed_seconds=record["elapsed_s"],
                note=__import__('json').dumps({"status": record["status"], "game_record": file_identity(path)}))
        committed = record["status"] == "completed"
        return {**pool.stats(task.config.label_sources()), "run_id": run_id,
                "inserted": inserted if committed else 0, "generated": generated,
                "valid": len(positions) if committed else 0, "filters": filters,
                "stopped": record["status"] == "interrupted", "game_status": record["status"],
                "source_ids": [source_id] if committed else [], "game_record": file_identity(path),
                "collection_model_id": incumbent["model_id"], "version": incumbent["version"],
                "teacher_calls": 0}
