"""binpack-backed PyTorch dataset (ticket 08).

* Random access through the existing mmap reader in `trainer/binpack.py`.
* Train/validation split is by WHOLE GAME, never by position, so D4 mirrors of
  a position can never leak into validation.
* D4 eight-fold augmentation happens on the fly in `__getitem__`; only feature
  indices are permuted, score/WDL labels are unchanged (Renju is D4-invariant).

The pure-Python split logic is importable without torch/numpy so it can be
tested on a bare environment; only the Dataset class needs torch.
"""
from __future__ import annotations

import random
from array import array
from pathlib import Path
from typing import Iterator

import binpack
import features

try:
    import torch
    _DatasetBase = torch.utils.data.Dataset
except ImportError:  # pragma: no cover - bare Python environment
    torch = None
    _DatasetBase = object


def choose_validation_games(n_games: int, val_fraction: float,
                            seed: int) -> set[int]:
    """Deterministic set of game indices held out for validation."""
    if n_games <= 0:
        return set()
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    rng = random.Random(seed)
    order = list(range(n_games))
    rng.shuffle(order)
    count = int(round(n_games * val_fraction))
    if n_games > 1:
        count = max(1, min(n_games - 1, count)) if val_fraction > 0 else 0
    return set(order[:count])


def split_record_indices(game_index_map: array, val_games: set[int],
                         split: str, max_records: int | None = None) -> array:
    """Record indices for one split; game membership is the only criterion."""
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")
    want_val = split == "val"
    selected = array("I")
    for index, game in enumerate(game_index_map):
        if (game in val_games) == want_val:
            selected.append(index)
            if max_records is not None and len(selected) >= max_records:
                break
    return selected


class BinpackDataset(_DatasetBase):
    def __init__(self, path: str | Path, split: str = "train",
                 val_fraction: float = 0.1, seed: int = 0,
                 augment: bool = True, max_records: int | None = None):
        self.path = Path(path)
        self.split = split
        self.augment = augment and split == "train"
        self.reader = binpack.BinpackReader(self.path)
        self.game_index_map = self.reader.game_index_map()
        n_games = self.game_index_map[-1] + 1 if len(self.game_index_map) else 0
        self.validation_games = choose_validation_games(n_games, val_fraction, seed)
        self.indices = split_record_indices(
            self.game_index_map, self.validation_games, split, max_records)
        self.game_ids = {self.game_index_map[i] for i in self.indices}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        record = self.reader[self.indices[item]]
        indices = record.feature_indices()
        if self.augment:
            indices = features.transform_sparse(
                indices, random.randrange(features.NUM_SYMMETRIES))
        return (features.dense_vector(indices), record.stm,
                float(record.score), record.wdl)

    def close(self) -> None:
        self.reader.close()

    def iter_samples(self) -> Iterator[tuple[list[int], int, float, int]]:
        """Sparse-index variant used by export/parity checks (no augmentation)."""
        for i in range(len(self.indices)):
            record = self.reader[self.indices[i]]
            yield (record.feature_indices(), record.stm,
                   float(record.score), record.wdl)


def assert_disjoint(train: BinpackDataset, val: BinpackDataset) -> None:
    """Fails loudly if any game appears in both splits."""
    overlap = train.game_ids & val.game_ids
    if overlap:
        raise ValueError(f"game-level leakage: {len(overlap)} game(s) in both splits")
