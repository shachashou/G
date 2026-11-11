#!/usr/bin/env python3
"""Frozen paired matches and evidence-based publication of GMVALUE1 models."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import random
import re
import shutil
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer"))
import value_format as vf
from value_data import canonical_key
from executor import ExecutorClient, ExecutorError
from match_runner import Engine, EngineFailure, EngineSpec, MOVE_RE, random_opening
from model_registry import ModelRegistry, RegistryError, DEFAULT_REGISTRY
from run_support import file_identity, source_identity, write_json


class EvaluationError(ValueError):
    pass


def position_payload(kind: str, state: dict, last_move,
                     *, game: int | None = None, pair: int | None = None,
                     engine: str | None = None, model_id: str | None = None,
                     worker: int | None = None) -> dict:
    """One optional live-board snapshot in the shared ticket-14 shape.

    The payload is derived from an accepted referee response only; it is never
    written into the sealed manifest/report or any training data.
    """
    return {"kind": kind, "cells": list(state["cells"]),
            "side_to_move": state["side_to_move"],
            "last_move": (None if last_move is None
                          else [int(last_move[0]), int(last_move[1])]),
            "ply": int(state["move_count"]), "result": state["result"],
            "game": game, "pair": pair, "engine": engine,
            "model_id": model_id, "worker": worker}


def emit_position(on_position, payload: dict) -> None:
    """Best-effort delivery; a preview consumer must never disturb a match."""
    if on_position is None:
        return
    try:
        on_position(payload)
    except Exception:
        pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class EvaluationConfig:
    pairs: int = 64
    min_pairs: int = 32
    score_threshold: float = 0.5
    alpha: float = 0.05
    movetime_ms: int = 200
    timeout_grace_ms: int = 1000
    threads: int = 1
    workers: int = 1
    report_interval_s: float = 2.0
    max_plies: int = 225
    opening_plies: int = 6
    opening_region: int = 7
    seed: int = 20261110
    vcf_policy: str = "root_and_children"
    vcf_nodes: int = 50000

    def validate(self):
        for name in ("pairs", "min_pairs", "movetime_ms", "max_plies"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise EvaluationError(f"{name} must be a positive integer")
        if (self.threads != 1 or type(self.workers) is not int or not 1 <= self.workers <= 256
                or not 0 <= self.report_interval_s <= 3600
                or not 0 <= self.timeout_grace_ms <= 60000
                or not 0 < self.alpha < 1 or not 0.5 <= self.score_threshold < 1
                or not 1 <= self.opening_plies <= 8
                or self.opening_region not in range(1, 16, 2)
                or self.opening_plies > self.opening_region ** 2
                or self.max_plies > 225 or self.vcf_nodes < 1
                or self.vcf_policy not in ("off", "root", "root_and_children")):
            raise EvaluationError("invalid fairness/statistics/search configuration")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_bundle(model: str | Path) -> dict:
    """Reject missing or mismatched numerical and training provenance."""
    model = Path(model).resolve()
    _, metadata = vf.read_value_weights(model)
    identity = file_identity(model)
    report = read_json(model.parent / "fidelity.json")
    if (report.get("format") != "GMVALUE-FIDELITY1"
            or report.get("model", {}).get("sha256") != identity["sha256"]
            or report.get("metadata") != metadata):
        raise EvaluationError("fidelity report does not identify this model")
    fidelity = report.get("fidelity", {})
    for metric, tolerance, field in (
            ("float_vs_integer", vf.VALUE_TOLERANCE, "value_tolerance"),
            ("quantised_float_vs_integer", vf.INTEGER_TOLERANCE, "integer_tolerance")):
        measured = fidelity.get(metric, {})
        count = measured.get("count")
        if fidelity.get(field) != tolerance or type(count) is not int or count < 1:
            raise EvaluationError("missing frozen fidelity contract or measurements")
        for key, limit in tolerance.items():
            value = measured.get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= limit:
                raise EvaluationError(f"numerical fidelity failed: {metric}/{key}")
    exclusion = metadata.get("training", {}).get("evaluation_exclusion", {})
    if exclusion.get("file") != "training_positions.json":
        raise EvaluationError("missing frozen training positions; re-export the model")
    path = model.parent / "training_positions.json"
    if file_identity(path)["sha256"] != exclusion.get("sha256"):
        raise EvaluationError("training position manifest hash mismatch")
    positions = read_json(path)
    if (positions.get("format") != "GMVALUE-TRAINING-POSITIONS1"
            or not isinstance(positions.get("d4_keys"), list)
            or not positions["d4_keys"]
            or any(not re.fullmatch(r"[0-9a-f]{64}", k) for k in positions["d4_keys"])):
        raise EvaluationError("invalid training position manifest")
    return {**identity, "model_id": metadata["model_id"], "metadata": metadata}


def copy_bundle(model: Path, folder: Path) -> dict:
    source = checked_bundle(model)
    folder.mkdir(parents=True)
    for name, original in (("model.gmvalue", model),
                           ("fidelity.json", model.parent / "fidelity.json"),
                           ("training_positions.json", model.parent / "training_positions.json")):
        shutil.copy2(original, folder / name)
    frozen = checked_bundle(folder / "model.gmvalue")
    if frozen["sha256"] != source["sha256"]:
        raise EvaluationError("model changed while freezing")
    return {**frozen, "source_path": str(model.resolve())}


def pair_statistics(games: list[dict], config: EvaluationConfig,
                    comparisons: int = 1) -> dict:
    """One bounded observation per complete opening pair, never per game.

    Hoeffding is conservative and distribution-free. A Bonferroni correction
    makes the one-sided bounds simultaneous over the frozen opponent list.
    """
    valid = [g for g in games if g["valid"]]
    scores = {"A-win": 1.0, "B-win": 0.0, "draw": 0.5}
    pairs = []
    for i in range(config.pairs):
        group = [g for g in games if g["pair"] == i and g["valid"]]
        if len(group) == 2 and {g["a_color"] for g in group} == {"black", "white"}:
            pairs.append(sum(scores[g["result"]] for g in group) / 2)
    n = len(pairs)
    mean = sum(pairs) / n if n else 0.0
    margin = math.sqrt(math.log(max(1, comparisons) / config.alpha) / (2 * n)) if n else 1.0
    lower, upper = max(0.0, mean - margin), min(1.0, mean + margin)
    enough = n >= config.min_pairs
    wins = sum(g["result"] == "A-win" for g in valid)
    losses = sum(g["result"] == "B-win" for g in valid)
    draws = len(valid) - wins - losses
    complete = len(games) == 2 * config.pairs and len(valid) == len(games)
    passed = complete and enough and lower > config.score_threshold
    conclusion = ("passed" if passed else "incomplete" if not complete
                  else "insufficient_pairs" if not enough
                  else "below_threshold" if upper <= config.score_threshold
                  else "insufficient_evidence")
    return {"wins": wins, "draws": draws, "losses": losses,
            "attempted_games": len(games), "requested_games": config.pairs * 2,
            "anomalies": {kind: sum(a["kind"] == kind for g in games for a in g["anomalies"])
                          for kind in sorted({a["kind"] for g in games for a in g["anomalies"]})},
            "valid_games": len(valid), "completed_pairs": n,
            "strict_win_rate": wins / len(valid) if valid else None,
            "score_rate": (wins + 0.5 * draws) / len(valid) if valid else None,
            "paired_score_rate": mean if n else None,
            "one_sided_lower": lower, "one_sided_upper": upper,
            "method": "paired-hoeffding-bonferroni-v1", "alpha": config.alpha,
            "comparisons": comparisons, "threshold": config.score_threshold,
            "passed": passed, "conclusion": conclusion}


class MatchEngine(Engine):
    """Reuse Gomocup transport with a total deadline and prompt cancellation."""
    def __init__(self, spec, timeout, stop):
        self.stop = stop
        self.deadline = None
        super().__init__(spec, timeout)

    def recv_line(self):
        deadline = self.deadline or time.monotonic() + self.move_timeout
        while True:
            if self.stop.is_set():
                raise InterruptedError("evaluation stopped")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineFailure("timeout", "total reply deadline exceeded")
            try:
                line = self._lines.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            if line is None:
                raise EngineFailure("crash", f"stdout closed (exit {self.p.poll()})")
            return line

    def move(self, cells):
        self.deadline = time.monotonic() + self.move_timeout
        try:
            self.sync_board(cells)
            for _ in range(128):
                line = self.recv_line()
                match = MOVE_RE.fullmatch(line)
                if match:
                    return int(match[1]), int(match[2])
                if line.startswith("ERROR") or line == "UNKNOWN":
                    raise EngineFailure("bad-reply", line)
            raise EngineFailure("bad-reply", "too many non-move replies")
        finally:
            self.deadline = None

    def close(self):
        # Stopping a match must not wait for a queued search to finish.
        if self.stop.is_set() and self.p.poll() is None:
            self.p.kill()
        super().close()
        self._reader.join(timeout=1)
        for stream in (self.p.stdin, self.p.stdout):
            if stream is not None:
                stream.close()


def verify_about(line: str, model: dict):
    fields = dict(re.findall(r'(\w+)="([^"\n]*)"', line))
    expected = {"eval": "nnue", "eval_engine": "gmvalue1-incremental-avx2",
                "value_model_id": model["model_id"], "value_model_arch": "450-1024-64-32-1",
                "value_model_rule": "renju", "value_model_feature_spec": "absolute-color-450",
                "value_model_contract": "black-absolute-tanh-v1"}
    if any(fields.get(k) != v for k, v in expected.items()):
        raise EngineFailure("identity-mismatch", f"expected {expected}, received {fields}")


def play_pair_game(engine_path: Path, candidate: dict, opponent: dict,
                   opening: list, config: EvaluationConfig, pair: int,
                   a_black: bool, stop: threading.Event,
                   on_position=None) -> dict:
    """Plays one colour of a frozen opening pair.

    ``on_position`` receives an optional :func:`position_payload` after every
    accepted referee move, including the neutral frozen opening (no engine
    label there). It never persists board cells.
    """
    started = time.monotonic()
    record = {"pair": pair, "a_color": "black" if a_black else "white",
              "opening": opening, "moves": [], "anomalies": [], "about": {},
              "valid": False, "result": None, "terminal": None}
    engines = {}
    active = None
    startup = True
    try:
        with ExecutorClient(engine_path) as referee:
            state = referee.new_game("renju")
            for x, y in opening:
                response = referee.play(x, y)
                if not response.get("ok") or response["result"] != "ongoing":
                    raise EvaluationError("invalid or terminal frozen opening")
                state = response
                record["moves"].append([x, y])
                emit_position(on_position, position_payload(
                    "evaluation", state, (x, y), pair=pair,
                    game=pair * 2 + (0 if a_black else 1)))
            for label, model in (("A", candidate), ("B", opponent)):
                active = label
                args = [str(engine_path), "--value-model", model["path"],
                        "--movetime", str(config.movetime_ms),
                        "--vcf-policy", config.vcf_policy, "--vcf-nodes", str(config.vcf_nodes)]
                spec = EngineSpec(label, " ".join(args), args)
                engine = MatchEngine(spec, (config.movetime_ms + config.timeout_grace_ms) / 1000, stop)
                engines[label] = engine
                engine.start(2, config.movetime_ms)
                engine.send("INFO thread_num 1")
                engine.send("ABOUT")
                line = engine.recv_line()
                verify_about(line, model)
                record["about"][label] = line
            startup = False
            for _ in range(config.max_plies):
                if stop.is_set():
                    raise InterruptedError("evaluation stopped")
                active = "A" if (state["side_to_move"] == "black") == a_black else "B"
                before = time.monotonic()
                x, y = engines[active].move(state["cells"])
                response = referee.play(x, y)
                if not response.get("ok"):
                    record["rejected_move"] = [x, y]
                    raise EngineFailure("illegal-move", f"{x},{y}: {response.get('reason', response)}")
                state = response
                record["moves"].append([x, y])
                emit_position(on_position, position_payload(
                    "evaluation", state, (x, y), pair=pair,
                    game=pair * 2 + (0 if a_black else 1), engine=active,
                    model_id=(candidate if active == "A" else opponent).get("model_id")))
                record.setdefault("move_times_ms", []).append(round((time.monotonic() - before) * 1000, 3))
                if state["result"] != "ongoing":
                    record["terminal"] = state["result"]
                    if state["result"] == "draw":
                        record["result"] = "draw"
                    else:
                        winner_black = state["result"] == "black_win"
                        record["result"] = "A-win" if winner_black == a_black else "B-win"
                    record["valid"] = True
                    break
            if not record["valid"]:
                record["anomalies"].append({"kind": "truncated", "detail": "ply limit; not a draw"})
    except InterruptedError as exc:
        record["anomalies"].append({"kind": "interrupted", "detail": str(exc)})
    except EngineFailure as exc:
        kind = "model-load-failure" if startup and exc.kind == "crash" else exc.kind
        record["anomalies"].append({"kind": kind, "engine": active, "detail": exc.detail})
        record["rule_result"] = "B-win" if active == "A" else "A-win"
    except (OSError, ExecutorError, EvaluationError) as exc:
        record["anomalies"].append({"kind": "startup-or-referee-error", "engine": active, "detail": str(exc)})
    finally:
        for engine in engines.values():
            engine.close()
    record["elapsed_s"] = round(time.monotonic() - started, 4)
    return record


def frozen_files(run: Path) -> dict:
    return {str(p.relative_to(run)): file_identity(p)["sha256"]
            for folder in ("candidate", "opponents", "engine")
            for p in sorted((run / folder).rglob("*")) if p.is_file()}


def frozen_file_changes(run: Path, manifest: dict) -> list[str]:
    """Describe any frozen input that no longer matches the sealed manifest."""
    recorded = manifest.get("files")
    if (not isinstance(recorded, dict) or not recorded
            or any(not isinstance(k, str) or not isinstance(v, str)
                   for k, v in recorded.items())):
        raise EvaluationError(f"cannot resume {run}: frozen file identities are missing or invalid")
    current = frozen_files(run)
    missing = sorted(set(recorded) - set(current))
    unexpected = sorted(set(current) - set(recorded))
    changed = sorted(name for name in set(recorded) & set(current)
                     if recorded[name] != current[name])
    return ([f"missing {name}" for name in missing]
            + [f"changed {name}" for name in changed]
            + [f"unexpected {name}" for name in unexpected])


def load_resume_manifest(run: Path) -> tuple[dict, str, dict]:
    """Return the sealed manifest, its hash and the last report of a run."""
    path = run / "manifest.json"
    if not path.is_file():
        raise EvaluationError(f"cannot resume {run}: manifest.json is missing")
    try:
        manifest = read_json(path)
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"cannot resume {run}: manifest.json is unreadable ({exc})") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != "GMVALUE-EVALUATION1":
        raise EvaluationError(f"cannot resume {run}: unsupported manifest format")
    manifest_sha = file_identity(path)["sha256"]
    report_path = run / "report.json"
    if not report_path.is_file():
        raise EvaluationError(f"cannot resume {run}: report.json is missing")
    try:
        previous = read_json(report_path)
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"cannot resume {run}: report.json is unreadable ({exc})") from exc
    if not isinstance(previous, dict) or previous.get("manifest_sha256") != manifest_sha:
        raise EvaluationError(
            f"cannot resume {run}: manifest.json was rewritten after the run was sealed")
    return manifest, manifest_sha, previous


def frozen_bundle(run: Path, folder: str, entry, label: str) -> dict:
    """Revalidate one frozen model copy in place without re-freezing it."""
    if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
        raise EvaluationError(f"cannot resume {run}: {label} identity is missing from the manifest")
    bundle = checked_bundle(run / folder / "model.gmvalue")
    if bundle["sha256"] != entry["sha256"] or bundle["model_id"] != entry.get("model_id"):
        raise EvaluationError(f"cannot resume {run}: frozen {label} does not match the manifest")
    bundle["source_path"] = entry.get("source_path", str(run / folder / "model.gmvalue"))
    return bundle


def resume_openings(run: Path, manifest: dict, config: EvaluationConfig) -> list:
    """Accept only the originally drawn, structurally complete opening list."""
    openings = manifest.get("openings")
    if (not isinstance(openings, list) or len(openings) != config.pairs
            or any(not isinstance(opening, list) or len(opening) != config.opening_plies
                   or any(not (isinstance(move, list) and len(move) == 2
                               and all(type(value) is int for value in move))
                          for move in opening)
                   for opening in openings)):
        raise EvaluationError(f"cannot resume {run}: frozen openings are invalid")
    return openings


def verify_publication(run: Path, manifest_sha: str, report_sha: str, stop_event=None):
    if (file_identity(run / "manifest.json")["sha256"] != manifest_sha
            or file_identity(run / "report.json")["sha256"] != report_sha):
        raise RegistryError("sealed configuration or report changed")
    manifest, report = read_json(run / "manifest.json"), read_json(run / "report.json")
    if frozen_files(run) != manifest["files"] or report.get("manifest_sha256") != manifest_sha:
        raise RegistryError("frozen evidence identity mismatch")
    if not report.get("eligible") or report.get("status") != "completed":
        raise RegistryError("incomplete, failed or insufficient evidence cannot be published")
    checked_bundle(run / "candidate" / "model.gmvalue")
    config = EvaluationConfig(**manifest["config"])
    config.validate()
    if manifest["bootstrap"] and (config.pairs != 1 or manifest["incumbent"] is not None):
        raise RegistryError("invalid initial registration contract")
    expected = ["initial"] if manifest["bootstrap"] else list(manifest["opponents"])
    if set(report["matches"]) != set(expected):
        raise RegistryError("opponent evidence mismatch")
    for name, games in report["matches"].items():
        if len(games) != 2 * config.pairs or not all(g["valid"] and not g["anomalies"] for g in games):
            raise RegistryError("all frozen games must complete without anomalies")
        if not manifest["bootstrap"] and not pair_statistics(games, config, len(expected))["passed"]:
            raise RegistryError("statistical evidence failed")
        # Recheck the actual legal game records, not a caller's valid flag.
        with ExecutorClient(run / manifest["engine_path"]) as referee:
            for index, game in enumerate(games):
                if stop_event is not None and stop_event.is_set():
                    raise RegistryError("stopped during publication checks")
                try:
                    verify_about(game["about"]["A"], manifest["candidate"])
                    verify_about(game["about"]["B"], manifest["candidate"] if manifest["bootstrap"]
                                 else manifest["opponents"][name])
                except EngineFailure as exc:
                    raise RegistryError(str(exc)) from exc
                if (game["pair"] != index // 2
                        or game["a_color"] != ("black" if index % 2 == 0 else "white")
                        or game["opening"] != manifest["openings"][index // 2]
                        or game["moves"][:len(game["opening"])] != game["opening"]):
                    raise RegistryError("paired game configuration mismatch")
                result = referee.replay(game["moves"])
                state = result.get("state", result)
                if not result.get("ok") or state.get("result") not in ("black_win", "white_win", "draw"):
                    raise RegistryError("game record is not a legal terminal game")
                terminal = state["result"]
                outcome = ("draw" if terminal == "draw" else "A-win"
                           if (terminal == "black_win") == (game["a_color"] == "black") else "B-win")
                if outcome != game["result"]:
                    raise RegistryError("game winner mismatch")
    return manifest, report


def parallel_games(engine, candidate, opponent, openings, config, stop,
                   on_position=None):
    """Bounded independent games; callers alone write the report.

    An abnormal game cancels in-flight siblings without setting the caller's
    stop event (which would incorrectly classify an engine failure as a user stop).
    ``on_position`` is forwarded to every game and may therefore be called
    concurrently from several worker threads.
    """
    halted = threading.Event()

    class MatchStop:
        def is_set(self):
            return stop.is_set() or halted.is_set()

    cancelled = MatchStop()
    jobs = iter((pair, opening, black) for pair, opening in enumerate(openings)
                for black in (True, False))
    pending = set()
    with ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix="evaluation") as executor:
        def submit():
            if cancelled.is_set():
                return
            job = next(jobs, None)
            if job is not None:
                pair, opening, black = job
                pending.add(executor.submit(play_pair_game, engine, candidate, opponent,
                                            opening, config, pair, black, cancelled,
                                            on_position))
        try:
            for _ in range(config.workers):
                submit()
            while pending:
                done, pending = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                for future in done:
                    if future.cancelled():
                        continue
                    game = future.result()
                    if not game["valid"]:
                        halted.set()
                    yield game
                if cancelled.is_set():
                    for future in pending:
                        future.cancel()
                else:
                    for _ in done:
                        submit()
        finally:
            halted.set()
            for future in pending:
                future.cancel()


def evaluate(candidate: str | Path, engine: str | Path, *,
             registry: str | Path = DEFAULT_REGISTRY,
             output_root: str | Path = ROOT / "data" / "evaluations",
             config: EvaluationConfig | None = None, opponents: list[str] | None = None,
             bootstrap: bool = False, stop_event=None, progress=None,
             resume_run: str | Path | None = None, on_position=None) -> dict:
    """Freeze, run, report and atomically publish; never writes training data.

    ``resume_run`` reuses an interrupted run directory verbatim: candidate,
    opponents, engine copy, openings, config, bootstrap flag, incumbent and
    manifest hash all come from its sealed manifest, and every game is rerun.
    In that mode the candidate/engine/config arguments are ignored, and a
    non-empty ``opponents`` list is rejected.  Interrupted games are never
    carried over.

    ``on_position`` receives optional live-board snapshots tagged with
    ``kind="evaluation"``; they are not part of the report or manifest.
    """
    stop = stop_event if stop_event is not None else threading.Event()
    store = ModelRegistry(registry)
    if resume_run is not None:
        run = Path(resume_run).resolve()
        manifest, manifest_sha, previous = load_resume_manifest(run)
        changes = frozen_file_changes(run, manifest)
        if changes:
            listed = "; ".join(changes[:8]) + ("; ..." if len(changes) > 8 else "")
            raise EvaluationError(
                f"cannot resume {run}: frozen files no longer match the manifest ({listed})")
        if opponents:
            raise EvaluationError("resumed runs use the frozen opponent list; do not pass opponents")
        recorded_config = manifest.get("config")
        if not isinstance(recorded_config, dict):
            raise EvaluationError(f"cannot resume {run}: frozen configuration is missing")
        try:
            config = EvaluationConfig(**recorded_config)
        except (KeyError, TypeError) as exc:
            raise EvaluationError(f"cannot resume {run}: invalid frozen configuration ({exc})") from exc
        config.validate()
        bootstrap = manifest.get("bootstrap")
        incumbent = manifest.get("incumbent")
        if type(bootstrap) is not bool or (bootstrap and incumbent is not None):
            raise EvaluationError(f"cannot resume {run}: invalid frozen registration contract")
        recorded_opponents = manifest.get("opponents")
        frozen_candidate = frozen_bundle(run, "candidate", manifest.get("candidate"), "candidate")
        frozen_opponents = {}
        if bootstrap:
            if recorded_opponents:
                raise EvaluationError(f"cannot resume {run}: initial registration cannot freeze opponents")
        else:
            if (not isinstance(incumbent, str) or not isinstance(recorded_opponents, dict)
                    or not recorded_opponents or incumbent not in recorded_opponents):
                raise EvaluationError(f"cannot resume {run}: frozen incumbent or opponents are invalid")
            for version, entry in recorded_opponents.items():
                if not isinstance(version, str) or not re.fullmatch(r"v[0-9]+", version):
                    raise EvaluationError(f"cannot resume {run}: invalid frozen opponent name")
                frozen_opponents[version] = frozen_bundle(
                    run, f"opponents/{version}", entry, f"opponent {version}")
        relative_engine = manifest.get("engine_path")
        if not isinstance(relative_engine, str):
            raise EvaluationError(f"cannot resume {run}: frozen engine path is missing")
        engine_path = (run / relative_engine).resolve()
        if not engine_path.is_relative_to(run) or not engine_path.is_file():
            raise EvaluationError(f"cannot resume {run}: frozen engine is missing")
        if manifest.get("rule") != "renju":
            raise EvaluationError(f"cannot resume {run}: unsupported frozen rule")
        openings = resume_openings(run, manifest, config)
        expected_current = incumbent
        started = time.monotonic()
        prior_count = previous.get("resume_count", 0)
        report = {"format": "GMVALUE-EVALUATION-REPORT1", "run": str(run),
                  "manifest_sha256": manifest_sha, "status": "running", "eligible": False,
                  "matches": {}, "statistics": {}, "publication": None,
                  "resumed": True,
                  "resume_count": (prior_count if type(prior_count) is int else 0) + 1,
                  "previous_status": previous.get("status")}
        write_json(run / "report.json", report)
        last_report_write = started
    else:
        config = config or EvaluationConfig(seed=random.SystemRandom().randrange(2**31))
        config.validate()
        if bootstrap and config.pairs != 1:
            raise EvaluationError("initial registration requires exactly one legal colour-swapped pair")
        current = store.read()["current"]
        if bootstrap != (current is None):
            raise EvaluationError("register v0 first" if current is None else "v0 already exists")
        selected = list(dict.fromkeys([current, *(opponents or [])])) if current else []
        run = Path(output_root).resolve() / (datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:12])
        run.mkdir(parents=True)
        # Each run owns copies; training/export can keep changing the source file.
        frozen_candidate = copy_bundle(Path(candidate).resolve(), run / "candidate")
        frozen_opponents = {}
        for version in selected:
            entry = store.resolve(version)
            bundle = copy_bundle(Path(entry["path"]), run / "opponents" / version)
            if bundle["sha256"] == frozen_candidate["sha256"]:
                raise EvaluationError("candidate is identical to a selected historical model")
            frozen_opponents[version] = bundle
        (run / "engine").mkdir()
        source_engine = Path(engine).resolve()
        engine_path = run / "engine" / source_engine.name
        shutil.copy2(source_engine, engine_path)
        exclusions = set(read_json(run / "candidate" / "training_positions.json")["d4_keys"])
        # Fresh holdouts across repeated local trials reduce tuning to a single
        # benchmark. Freeze the prior identities too, so selection is reproducible.
        prior_openings = set()
        for previous in run.parent.glob("*/manifest.json"):
            previous_manifest = read_json(previous)
            if previous_manifest.get("format") == "GMVALUE-EVALUATION1":
                prior_openings.update(previous_manifest.get("opening_d4_keys", []))
        openings, seen = [], set()
        rng = random.Random(config.seed)
        with ExecutorClient(engine_path) as referee:
            for _ in range(max(1000, config.pairs * 100)):
                opening = [list(p) for p in random_opening(rng, config.opening_plies, config.opening_region)]
                state = referee.new_game("renju")
                for x, y in opening:
                    state = referee.play(x, y)
                    if not state.get("ok") or state["result"] != "ongoing":
                        break
                if not state.get("ok") or state["result"] != "ongoing":
                    continue
                key = canonical_key(state["cells"], state["side_to_move"])
                if key in exclusions or key in seen or key in prior_openings:
                    continue
                seen.add(key)
                openings.append(opening)
                if len(openings) == config.pairs:
                    break
        if len(openings) != config.pairs:
            raise EvaluationError("cannot generate enough distinct held-out openings")
        manifest = {"format": "GMVALUE-EVALUATION1", "purpose": "evaluation-only",
                    "created_at": now(), "bootstrap": bootstrap, "incumbent": current,
                    "candidate": frozen_candidate, "opponents": frozen_opponents,
                    "engine_path": str(engine_path.relative_to(run)),
                    "config": asdict(config), "rule": "renju", "openings": openings,
                    "opening_d4_keys": sorted(seen), "source": source_identity(),
                    "prior_evaluation_d4_keys": sorted(prior_openings),
                    "implementation": {name: file_identity(ROOT / relative)
                        for name, relative in (("evaluation", "tools/history_evaluation.py"),
                            ("registry", "tools/model_registry.py"), ("transport", "tools/match_runner.py"),
                            ("referee_client", "tools/executor.py"), ("format", "trainer/value_format.py"))},
                    "statistics": "paired-hoeffding-bonferroni-v1",
                    "anomaly_policy": "abort-run-no-promotion",
                    "stop_rule": "fixed-pairs-no-early-success", "files": frozen_files(run)}
        write_json(run / "manifest.json", manifest)
        manifest_sha = file_identity(run / "manifest.json")["sha256"]
        expected_current = current
        report = {"format": "GMVALUE-EVALUATION-REPORT1", "run": str(run),
                  "manifest_sha256": manifest_sha, "status": "running", "eligible": False,
                  "matches": {}, "statistics": {}, "publication": None}
        write_json(run / "report.json", report)
        started = time.monotonic()
        last_report_write = started
    aborted = False
    emit = None
    if on_position is not None:
        def emit(payload):
            # Source of truth for evaluation frames is this replay run, never
            # a persisted report field.
            payload["kind"] = "evaluation"
            on_position(payload)
    matches = {"initial": frozen_candidate} if bootstrap else frozen_opponents
    for name, opponent in matches.items():
        games = report["matches"][name] = []
        for game in parallel_games(engine_path, frozen_candidate, opponent, openings,
                                   config, stop, emit):
            games.append(game)
            games.sort(key=lambda g: (g["pair"], g["a_color"] != "black"))
            report["statistics"][name] = pair_statistics(games, config, len(matches))
            if time.monotonic() - last_report_write >= config.report_interval_s:
                write_json(run / "report.json", report)
                last_report_write = time.monotonic()
            if progress:
                progress({"opponent": name, "games": len(games), "total": config.pairs * 2,
                          "statistics": report["statistics"][name], "run": str(run)})
            if not game["valid"]:
                aborted = True
        if aborted or stop.is_set():
            aborted = True
            break
    intact = (file_identity(run / "manifest.json")["sha256"] == manifest_sha
              and frozen_files(run) == manifest["files"])
    complete = not aborted and not stop.is_set() and intact
    report["status"] = ("completed" if complete else "interrupted" if stop.is_set()
                        else "identity-mismatch" if not intact else "aborted")
    report["eligible"] = complete and (bootstrap or all(
        s["passed"] for s in report["statistics"].values()))
    report["finished_at"] = now()
    report["elapsed_s"] = round(time.monotonic() - started, 3)
    report["initial_registration_only"] = bootstrap
    write_json(run / "report.json", report)
    report_sha = file_identity(run / "report.json")["sha256"]
    # Keep the sealed report unchanged; publication has its own receipt.
    if report["eligible"]:
        try:
            if stop.is_set():
                raise RegistryError("stopped before publication")
            # A resume (or a racing publisher) may already have promoted this
            # exact candidate; never create a duplicate version for it.
            current_entry = store.resolve()
            if (current_entry is not None
                    and current_entry["sha256"] == frozen_candidate["sha256"]):
                report["publication"] = current_entry
                report["publication_skipped"] = "candidate already current"
            else:
                report["publication"] = store.publish(run, expected_current=expected_current,
                    expected_manifest_sha=manifest_sha, expected_report_sha=report_sha,
                    stop_event=stop)
        except (RegistryError, EvaluationError, OSError, ValueError) as exc:
            report["publication_error"] = str(exc)
    report["report_sha256"] = report_sha
    write_json(run / "result.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="show current and historical versions")
    listing.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    for name in ("initial", "challenge"):
        p = sub.add_parser(name)
        p.add_argument("candidate")
        p.add_argument("--engine", default=str(ROOT / "build/engine/gomoku_nnue.exe"))
        p.add_argument("--registry", default=str(DEFAULT_REGISTRY))
        p.add_argument("--output-root", default=str(ROOT / "data/evaluations"))
        p.add_argument("--pairs", type=int, default=1 if name == "initial" else 64)
        p.add_argument("--min-pairs", type=int, default=32)
        p.add_argument("--score-threshold", type=float, default=0.5)
        p.add_argument("--alpha", type=float, default=0.05)
        p.add_argument("--movetime-ms", type=int, default=200)
        p.add_argument("--workers", type=int, default=1, help="independent games in parallel; each engine remains single-threaded")
        p.add_argument("--timeout-grace-ms", type=int, default=1000)
        p.add_argument("--max-plies", type=int, default=225)
        p.add_argument("--vcf-policy", choices=("off", "root", "root_and_children"), default="root_and_children")
        p.add_argument("--vcf-nodes", type=int, default=50000)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--opponent", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            result = ModelRegistry(args.registry).read()
        else:
            stop = threading.Event()
            # SIGINT sets a flag so partial reports are saved and processes reaped.
            import signal
            signal.signal(signal.SIGINT, lambda *_: stop.set())
            result = evaluate(args.candidate, args.engine, registry=args.registry,
                output_root=args.output_root, bootstrap=args.command == "initial",
                opponents=args.opponent, stop_event=stop,
                config=EvaluationConfig(pairs=args.pairs, movetime_ms=args.movetime_ms, workers=args.workers,
                    min_pairs=args.min_pairs, score_threshold=args.score_threshold, alpha=args.alpha,
                    timeout_grace_ms=args.timeout_grace_ms, max_plies=args.max_plies,
                    vcf_policy=args.vcf_policy, vcf_nodes=args.vcf_nodes,
                    seed=args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if args.command == "list" or result.get("publication") else 2
    except (EvaluationError, RegistryError, OSError, ValueError) as exc:
        print(f"history_evaluation: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
