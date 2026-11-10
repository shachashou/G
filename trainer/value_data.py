#!/usr/bin/env python3
"""Training data for the value network: pool samples, contract, D4 (ticket 06).

Sources are committed pool samples (ticket 05). Each sample becomes:

- 450 absolute-colour features (``features.py``) - no perspective flip, so
  Renju's black/white asymmetry is preserved;
- a **black-absolute** target. ``teacher-v1`` values are side-to-move values,
  so the conversion is ``target = value * (+1 if black to move else -1)``.
  ``selfplay_result`` samples store a black-absolute result already.

Online augmentation randomly applies one of the 8 D4 spatial transforms (no
colour swap). Validation is never augmented and is isolated from training by
game identity and by D4-canonical position keys; a leaked position raises
instead of silently training on validation data.
"""
from __future__ import annotations

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

import features

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from sample_pool import SamplePool  # noqa: E402

SIZE = 15
D4_TABLES = features.symmetry_table()


@dataclass(frozen=True)
class Sample:
    cells: tuple[int, ...]
    side: str
    target: float  # black-absolute value in [-1, 1]
    source_id: str
    label_source: str
    label_version: str

    def sparse(self) -> list[int]:
        black = [i for i, c in enumerate(self.cells) if c == 1]
        white = [i for i, c in enumerate(self.cells) if c == 2]
        return features.sparse_indices(black, white)


@dataclass(frozen=True)
class DataBundle:
    train: list[Sample]
    validation: list[Sample]
    identity: dict

    def summary(self) -> dict:
        return {"train": len(self.train), "validation": len(self.validation),
                **self.identity}


def record_target(record: dict) -> float:
    """Teacher (side-to-move) or result (already black-absolute) -> target."""
    side = record["side_to_move"]
    source = record["label_source"]
    if source == "teacher":
        value = record.get("value")
        if value is None:
            raise ValueError("teacher sample without value")
        base = float(value)
        target = base if side == "black" else -base
    elif source == "selfplay_result":
        value = record.get("result_value")
        if value is None:
            raise ValueError("result sample without result_value")
        target = float(value)
    else:
        raise ValueError(f"sample {record.get('id')} has no supervision")
    if not -1.0 <= target <= 1.0:
        raise ValueError(f"target {target} outside [-1, 1]")
    return target


def _expected_side(cells) -> str | None:
    black = sum(1 for c in cells if c == 1)
    white = sum(1 for c in cells if c == 2)
    if black == white:
        return "black"
    if black == white + 1:
        return "white"
    return None


def canonical_key(cells, side: str) -> str:
    """D4-canonical identity of a position (colours and side preserved)."""
    occupied = [i for i, c in enumerate(cells) if c]
    best = None
    for transform in range(features.NUM_SYMMETRIES):
        key = tuple(sorted((features.transform_cell(cell, transform), cells[cell])
                           for cell in occupied))
        if best is None or key < best:
            best = key
    digest = hashlib.sha256(repr((best, side)).encode())
    return digest.hexdigest()


def _sample_from_record(record: dict) -> Sample:
    cells = tuple(int(c) for c in record["board"])
    side = record["side_to_move"]
    if len(cells) != SIZE * SIZE:
        raise ValueError("board must have 225 cells")
    expected = _expected_side(cells)
    if expected is None or expected != side:
        raise ValueError(
            f"sample {record.get('id')}: side {side!r} does not match the "
            f"stone count")
    return Sample(cells=cells, side=side, target=record_target(record),
                  source_id=str(record["source_id"]),
                  label_source=str(record["label_source"]),
                  label_version=str(record["label_version"]))


def check_d4_isolation(train: list[Sample], validation: list[Sample]) -> None:
    """Raises if a training position is D4-equivalent to a validation one."""
    validation_keys = {canonical_key(s.cells, s.side) for s in validation}
    leaked = [s for s in train
              if canonical_key(s.cells, s.side) in validation_keys]
    if leaked:
        raise ValueError(
            f"{len(leaked)} training position(s) are D4-equivalent to the "
            f"validation split; refusing to train")


def load_pool(pool_path: str | Path) -> DataBundle:
    """Reads the frozen train/validation split from a committed sample pool."""
    path = Path(pool_path)
    if not path.is_file():
        raise ValueError(f"sample pool not found: {path}")
    with SamplePool(path, create=False) as pool:
        problems = pool.validate()
        if problems:
            raise ValueError("pool failed validation: " + "; ".join(problems))
        train_rows = pool.read_batch(split="train")
        val_rows = pool.read_batch(split="validation")
        stats = pool.stats()
    if not train_rows:
        raise ValueError("pool has no committed training samples")
    if not val_rows:
        raise ValueError("pool has no committed validation samples")
    train = [_sample_from_record(row) for row in train_rows]
    validation = [_sample_from_record(row) for row in val_rows]
    check_d4_isolation(train, validation)
    digest = hashlib.sha256()
    for row in sorted(train_rows + val_rows,
                      key=lambda r: (r["source_id"], r["id"])):
        digest.update(str((row["source_id"], row["id"])).encode())
    identity = {
        "pool": str(path),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_games": len({s.source_id for s in train}),
        "validation_games": len({s.source_id for s in validation}),
        "label_versions": sorted({s.label_version for s in train + validation}),
        "samples_sha256": digest.hexdigest(),
        "pool_capacity": stats.get("capacity"),
    }
    return DataBundle(train=train, validation=validation, identity=identity)


def dense(indices: list[int]) -> list[float]:
    return features.dense_vector(indices)


def tensors(samples: list[Sample], *, augment: bool = False,
            rng: random.Random | None = None) -> tuple[Tensor, Tensor]:
    """Dense feature and target tensors; optional online D4 augmentation."""
    rows = []
    for sample in samples:
        vector = dense(sample.sparse())
        if augment and rng is not None:
            table = D4_TABLES[rng.randrange(features.NUM_SYMMETRIES)]
            vector = [vector[table[i]] for i in range(features.NUM_FEATURES)]
        rows.append(vector)
    x = torch.tensor(rows, dtype=torch.float32)
    y = torch.tensor([s.target for s in samples], dtype=torch.float32)
    return x, y


def batches(samples: list[Sample], batch_size: int, *,
            augment: bool = False, rng: random.Random | None = None,
            device: str = "cpu"):
    order = list(samples)
    if augment and rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        x, y = tensors(chunk, augment=augment, rng=rng)
        yield x.to(device), y.to(device)
