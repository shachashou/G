#!/usr/bin/env python3
"""Continuous collection into the rolling sample pool (ticket 05).

One run:

1. plays complete games with the current engine (handcrafted evaluation is a
   valid starter - no trained model needed), using random openings inside the
   configured region and fixed-depth search for the rest;
2. samples positions every ``sample_every`` plies, including midgame;
3. labels each candidate through the shared pipeline (executor validation ->
   VCF filter -> Rapfi teacher), staging accepted samples against the game;
4. commits the game atomically; uncommitted or interrupted games are invisible
   to readers and aborted on stop;
5. records the run's configuration, counts, filter reasons and measured
   effective rate in the pool.

Exploration/collection settings are part of the recorded run config:
``opening_plies``, ``opening_region``, ``depth``, ``sample_every``,
``budget_ms`` and the strategy version.

Usage::

    python tools/collect.py --pool data/pool/pool.sqlite3 \\\\
        --engine build/engine/gomoku_nnue.exe --games 2 --capacity 200 \\\\
        --depth 4 --sample-every 3 --budget-ms 200

Programmatic use (tests inject a stub teacher)::

    from collect import CollectConfig, collect
    summary = collect(pool, config, game_client, label_client, teacher)
"""

import argparse
import json
import random
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from executor import ExecutorClient, ExecutorError  # noqa: E402
from rapfi_teacher import (RapfiTeacher, TeacherError, check_install,  # noqa: E402
                           load_lock)
from sample_pool import PoolError, SamplePool  # noqa: E402
from teacher_label import label_position  # noqa: E402

SIZE = 15


def position_payload(kind: str, state: dict, last_move,
                     *, game: int | None = None, pair: int | None = None,
                     engine: str | None = None, model_id: str | None = None,
                     worker: int | None = None) -> dict:
    """One optional live-board snapshot in the shared ticket-14 shape.

    The payload is derived from an accepted executor response only; it is
    never stored in the sample pool, task state or task events.
    """
    return {"kind": kind, "cells": list(state["cells"]),
            "side_to_move": state["side_to_move"],
            "last_move": (None if last_move is None
                          else [int(last_move[0]), int(last_move[1])]),
            "ply": int(state["move_count"]), "result": state["result"],
            "game": game, "pair": pair, "engine": engine,
            "model_id": model_id, "worker": worker}


def emit_position(on_position, payload: dict) -> None:
    """Best-effort delivery; a preview consumer must never disturb collection."""
    if on_position is None:
        return
    try:
        on_position(payload)
    except Exception:
        pass


@dataclass
class CollectConfig:
    games: int = 4
    depth: int = 4
    max_plies: int = 80
    opening_plies: int = 3
    opening_region: int = 7
    sample_every: int = 2
    budget_ms: int = 300
    vcf_nodes: int = 50000
    vcf_plies: int = 10
    seed: int = 1
    model_id: str = "handcrafted"
    strategy_version: str = "collect-v1"
    throttle_ms: int = 0
    task_id: str | None = None
    model_sha256: str | None = None
    parent_version: str | None = None

    def snapshot(self) -> dict:
        data = asdict(self)
        data["validation"] = "first committed games"
        return data


def play_game(game_client: ExecutorClient, config: CollectConfig,
              rng: random.Random,
              stop_event: threading.Event | None,
              on_position=None) -> tuple[list[tuple], str | None]:
    """Plays one engine game; returns sampled positions and the result.

    ``on_position`` receives an optional :func:`position_payload` after every
    accepted move. It is only called from this thread, after the executor has
    already replied, so it can never re-enter an executor request. The game
    index is unknown here; :func:`collect` adds it.
    """
    state = game_client.new_game("renju")
    positions: list[tuple] = []
    legal = [tuple(p) for p in game_client.legal_moves().get("moves", [])]
    for ply in range(config.max_plies):
        if stop_event is not None and stop_event.is_set():
            return positions, None
        if state["result"] != "ongoing":
            break
        if ply < config.opening_plies:
            region = [p for p in legal
                      if abs(p[0] - SIZE // 2) <= config.opening_region // 2
                      and abs(p[1] - SIZE // 2) <= config.opening_region // 2]
            x, y = rng.choice(region or legal)
        else:
            response = game_client.think(depth=config.depth)
            if not response.get("ok"):
                raise ExecutorError(f"engine think failed: {response}")
            x, y = response["move"]
        response = game_client.play(int(x), int(y))
        if not response.get("ok"):
            raise ExecutorError(f"engine move rejected: {response}")
        state = response
        if on_position is not None:
            emit_position(on_position, position_payload(
                "collect", state, (x, y), model_id=config.model_id))
        legal = [tuple(p) for p in game_client.legal_moves().get("moves", [])]
        if (ply >= config.opening_plies
                and (ply - config.opening_plies) % max(1, config.sample_every) == 0):
            positions.append((list(state["cells"]), state["side_to_move"], ply + 1))
    result = state["result"] if state["result"] != "ongoing" else None
    return positions, result


def collect(pool: SamplePool, config: CollectConfig,
            game_client: ExecutorClient, label_client: ExecutorClient,
            teacher, stop_event: threading.Event | None = None,
            pause_event: threading.Event | None = None,
            progress=None, on_position=None) -> dict:
    """Runs one collection run; caller owns the clients/teacher lifecycle.

    ``on_position`` receives one :func:`position_payload` per accepted move,
    tagged with this run's 1-based game index. Snapshots are optional and are
    never written to the pool, task state or task events.
    """
    run_id = pool.start_run(config.snapshot())
    started = time.monotonic()
    generated = 0
    valid = 0
    inserted = 0
    rejected_games = 0
    filters: dict[str, int] = {}
    source_ids: list[str] = []
    pending_generated = 0
    pending_filters = {}
    rng = random.Random(config.seed)

    def bump(reason: str) -> None:
        filters[reason] = filters.get(reason, 0) + 1

    def flush_observations():
        nonlocal pending_generated
        pool.record_observations(pending_generated, pending_filters)
        pending_generated = 0
        pending_filters.clear()

    def report(stopped: bool = False) -> dict:
        flush_observations()
        elapsed = time.monotonic() - started
        summary = pool.finish_run(
            run_id, generated=generated, valid=valid, inserted=inserted,
            filters=filters, elapsed_seconds=elapsed,
            note="stopped" if stopped else "")
        stats = pool.stats()
        if progress is not None:
            progress({"generated": generated, "valid": valid,
                      "inserted": inserted, "filters": dict(filters),
                      "stopped": stopped, **summary, **stats})
        return {**stats, "run_id": run_id, "generated": generated,
                "valid": valid, "inserted": inserted, "filters": dict(filters),
                "rejected_games": rejected_games, "stopped": stopped,
                "source_ids": list(source_ids),
                **summary}

    for index in range(config.games):
        # A pause takes effect between whole games; the current game either
        # commits or is aborted, never half-written.
        while (pause_event is not None and pause_event.is_set()
               and not (stop_event is not None and stop_event.is_set())):
            time.sleep(0.1)
        if stop_event is not None and stop_event.is_set():
            return report(stopped=True)
        source_id = (f"{config.strategy_version}-{config.seed}-"
                     f"{index:04d}-{uuid.uuid4().hex[:8]}")
        game_id = pool.begin_game(source_id, kind="selfplay",
                                  model_id=config.model_id, rule="renju")
        game_preview = None
        if on_position is not None:
            def game_preview(payload, _game=index + 1):
                # play_game only knows the board; the run knows the game.
                payload["game"] = _game
                on_position(payload)
        try:
            positions, result = play_game(game_client, config, rng, stop_event,
                                          game_preview)
            accepted = []
            for cells, side, ply in positions:
                generated += 1
                pending_generated += 1
                if stop_event is not None and stop_event.is_set():
                    break
                report_label = label_position(
                    executor=label_client, teacher=teacher, store=None,
                    cells=cells, side=side, source_kind="game",
                    source_id=source_id, ply=ply, model_id=config.model_id,
                    final_result=result, budget_ms=config.budget_ms,
                    vcf_nodes=config.vcf_nodes, vcf_plies=config.vcf_plies,
                    return_record=True)
                if not report_label.get("accepted"):
                    reason = str(report_label.get("filter_reason", "rejected"))
                    bump(reason)
                    pending_filters[reason] = pending_filters.get(reason, 0) + 1
                    continue
                valid += 1
                pool_record = dict(report_label["record"])
                pool_record["label_source"] = "teacher"
                pool_record["result_value"] = None
                accepted.append(pool_record)
            if stop_event is not None and stop_event.is_set():
                pool.abort_game(game_id)
                return report(stopped=True)
            for status in pool.add_samples(game_id, accepted):
                if status == "inserted":
                    inserted += 1
                elif status == "duplicate":
                    bump("duplicate")
                elif status == "d4_duplicate":
                    bump("d4_duplicate")
                else:
                    bump(status)
            if stop_event is not None and stop_event.is_set():
                pool.abort_game(game_id)
                return report(stopped=True)
            pool.commit_game(game_id, final_result=result)
            flush_observations()
            source_ids.append(source_id)
        except Exception:
            flush_observations()
            pool.abort_game(game_id)
            rejected_games += 1
            if stop_event is not None and stop_event.is_set():
                return report(stopped=True)
            raise
        if progress is not None:
            progress({"generated": generated, "valid": valid,
                      "inserted": inserted, "filters": dict(filters),
                      "games_done": index + 1, **pool.stats()})
        if config.throttle_ms > 0:
            utilization = pool.stats().get("utilization", 0.0)
            if utilization >= 0.95:
                time.sleep(config.throttle_ms / 1000.0)
    return report(stopped=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", default="build/engine/gomoku_nnue.exe")
    parser.add_argument("--pool", default="data/pool/pool.sqlite3")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=1000)
    parser.add_argument("--validation-games", type=int, default=1)
    parser.add_argument("--dedup-d4", action="store_true", default=True)
    parser.add_argument("--no-dedup-d4", dest="dedup_d4", action="store_false")
    parser.add_argument("--new-ratio", type=float, default=0.5)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--opening-plies", type=int, default=3)
    parser.add_argument("--opening-region", type=int, default=7)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--budget-ms", type=int, default=300)
    parser.add_argument("--vcf-nodes", type=int, default=50000)
    parser.add_argument("--vcf-plies", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--throttle-ms", type=int, default=0)
    parser.add_argument("--model-id", default="handcrafted")
    args = parser.parse_args(argv)

    identity = load_lock()
    install = check_install(identity)
    if not install.ok:
        print(f"collect: teacher unavailable: {install.summary()}",
              file=sys.stderr)
        return 2
    config = CollectConfig(
        games=args.games, depth=args.depth, max_plies=args.max_plies,
        opening_plies=args.opening_plies, opening_region=args.opening_region,
        sample_every=args.sample_every, budget_ms=args.budget_ms,
        vcf_nodes=args.vcf_nodes, vcf_plies=args.vcf_plies, seed=args.seed,
        model_id=args.model_id, throttle_ms=args.throttle_ms)
    pool_config = {
        "capacity": args.capacity,
        "validation_games": args.validation_games,
        "validation_seed": args.seed,
        "dedup_d4": args.dedup_d4,
        "new_ratio": args.new_ratio,
        "strategy_version": config.strategy_version,
    }
    pool = SamplePool(args.pool, create=True)
    try:
        for key, value in pool_config.items():
            pool.set_config(key, value)
        with ExecutorClient(args.engine) as game_client, \
                ExecutorClient(args.engine) as label_client, \
                RapfiTeacher(identity=identity) as teacher:
            summary = collect(pool, config, game_client, label_client, teacher)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (TeacherError, ExecutorError, PoolError) as exc:
        print(f"collect: {exc}", file=sys.stderr)
        return 2
    finally:
        pool.close()


if __name__ == "__main__":
    sys.exit(main())
