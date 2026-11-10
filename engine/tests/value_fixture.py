#!/usr/bin/env python3
"""Shared fixture for the ticket-06 tests: a small, learnable sample pool.

The black-absolute target is +0.8 when the centre point is black, -0.8 when it
is white and 0.0 otherwise, plus a tiny stone-count term. That makes the
network able to learn it in a few dozen steps, so the tests can assert that
validation error actually drops instead of only that the loop runs.

No engine and no teacher are involved: records are written through the public
pool boundary (ticket 05).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from sample_pool import SamplePool  # noqa: E402

SIZE = 15
CENTRE = 7 * SIZE + 7


def target_black(cells: list[int]) -> float:
    sign = 0.0
    if cells[CENTRE] == 1:
        sign = 0.8
    elif cells[CENTRE] == 2:
        sign = -0.8
    stones = sum(1 for c in cells if c) / 225.0
    return max(-1.0, min(1.0, sign + stones))


def stm_value(cells: list[int], side: str) -> float:
    value = target_black(cells)
    return value if side == "black" else -value


def record(cells: list[int], side: str) -> dict:
    value = stm_value(cells, side)
    return {
        "board": bytes(cells),
        "side_to_move": side,
        "rule": "renju",
        "tactical_status": "vcf_clear",
        "label_source": "teacher",
        "label_version": "teacher-v1",
        "raw_eval": str(int(value * 600)),
        "raw_score": int(value * 600),
        "raw_perspective": "side_to_move",
        "normalization": "tanh",
        "normalization_param": 600.0,
        "value": value,
        "result_value": None,
        "suggested_move": None,
        "depth": 10,
        "budget_ms": 800,
        "threads": 1,
        "teacher_release": "250615",
        "teacher_version": "fixture",
        "teacher_exe_sha256": "a" * 64,
        "teacher_weights_sha256": "b" * 64,
        "teacher_config_sha256": "c" * 64,
    }


def force_centre(cells: list[int], color: int) -> None:
    """Puts `color` on the centre without changing the stone counts."""
    if cells[CENTRE] == color:
        return
    if cells[CENTRE] != 0:
        for index, value in enumerate(cells):
            if value == color:
                cells[index] = cells[CENTRE]
                cells[CENTRE] = color
                return
        return
    for index, value in enumerate(cells):
        if value == color:
            cells[index] = 0
            cells[CENTRE] = color
            return


def make_board(rng: random.Random, stones: int) -> tuple[list[int], str]:
    cells = [0] * 225
    placed = 0
    while placed < stones:
        x = rng.randrange(2, 13)
        y = rng.randrange(2, 13)
        index = y * SIZE + x
        if cells[index]:
            continue
        cells[index] = 1 if placed % 2 == 0 else 2
        placed += 1
    black = cells.count(1)
    white = cells.count(2)
    side = "black" if black == white else "white"
    return cells, side


def build_pool(path: str | Path, *, seed: int = 11, validation_games: int = 5,
               train_games: int = 16, samples_per_game: int = 4) -> Path:
    path = Path(path)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()
    rng = random.Random(seed)
    with SamplePool(path, capacity=1000) as pool:
        pool.set_config("validation_games", validation_games)
        for group, count in (("val", validation_games),
                             ("train", train_games)):
            for game_index in range(count):
                game = pool.begin_game(f"{group}-{game_index}",
                                       kind="selfplay", model_id="fixture")
                for sample_index in range(samples_per_game):
                    stones = 6 + rng.randrange(8)
                    cells, side = make_board(rng, stones)
                    if sample_index == 0:
                        # Force centre control in the first sample of a game
                        # so both target classes appear.
                        force_centre(cells, 1 if game_index % 2 == 0 else 2)
                    pool.add_sample(game, record(cells, side))
                pool.commit_game(game, final_result=None)
    return path
