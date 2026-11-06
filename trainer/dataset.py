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


def clip_score(value: float, clip: float) -> float:
    """Clamp a saturated search score into the trainable ``tanh`` range.

    Datagen encodes mate / out-of-range search scores as +/-32767. With the
    ticket-08 loss ``MSE(tanh(eval/400), tanh(score/400))`` those labels map to
    targets near +/-1. This compresses label magnitudes but does not by itself
    remove the prediction's gradient. Clipping is
    *optional data cleaning* (``clip <= 0`` keeps the original behaviour), the
    loss itself is unchanged, and WDL labels already carry the win/loss fact.
    """
    if clip <= 0:
        return value
    return max(-clip, min(clip, value))


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
                         split: str, max_records: int | None = None,
                         exclude_games: set[int] | None = None,
                         include_games: set[int] | None = None) -> array:
    """Record indices for one split; game membership is the only criterion.

    ``include_games``/``exclude_games`` (ticket 13) let a frozen validation
    split and a train manifest restrict the selected games explicitly; both
    are game-level filters applied before the ``max_records`` cap.
    """
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")
    want_val = split == "val"
    exclude = exclude_games or frozenset()
    include = include_games
    selected = array("I")
    for index, game in enumerate(game_index_map):
        if include is not None and game not in include:
            continue
        if game in exclude:
            continue
        if (game in val_games) == want_val:
            selected.append(index)
            if max_records is not None and len(selected) >= max_records:
                break
    return selected


class BinpackDataset(_DatasetBase):
    def __init__(self, path: str | Path, split: str = "train",
                 val_fraction: float = 0.1, seed: int = 0,
                 augment: bool = True, max_records: int | None = None,
                 score_clip: float = 0.0,
                 validation_games: set[int] | None = None,
                 exclude_games: set[int] | None = None,
                 include_games: set[int] | None = None):
        self.path = Path(path)
        self.split = split
        self.augment = augment and split == "train"
        self.score_clip = score_clip
        self.reader = binpack.BinpackReader(self.path)
        self.game_index_map = self.reader.game_index_map()
        n_games = self.game_index_map[-1] + 1 if len(self.game_index_map) else 0
        if validation_games is None:
            validation_games = choose_validation_games(n_games, val_fraction,
                                                       seed)
        self.validation_games = set(validation_games)
        if include_games is not None and include_games & self.validation_games:
            raise ValueError("include_games must not contain validation games")
        self.indices = split_record_indices(
            self.game_index_map, self.validation_games, split, max_records,
            exclude_games=exclude_games, include_games=include_games)
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
                clip_score(float(record.score), self.score_clip), record.wdl)

    def close(self) -> None:
        self.reader.close()

    def iter_samples(self) -> Iterator[tuple[list[int], int, float, int]]:
        """Sparse-index variant used by export/parity checks (no augmentation)."""
        for i in range(len(self.indices)):
            record = self.reader[self.indices[i]]
            yield (record.feature_indices(), record.stm,
                   clip_score(float(record.score), self.score_clip), record.wdl)


def assert_disjoint(train: BinpackDataset, val: BinpackDataset) -> None:
    """Fails loudly if any game appears in both splits."""
    overlap = train.game_ids & val.game_ids
    if overlap:
        raise ValueError(f"game-level leakage: {len(overlap)} game(s) in both splits")


class MultiBinpackDataset(_DatasetBase):
    """Concatenation of BinpackDatasets (ticket 13 train manifests).

    Validation always comes from the frozen single file; this wrapper only
    feeds the training side with the manifest's include-lists already applied
    per file, so extra data can never touch the frozen validation split.
    """

    def __init__(self, datasets: list[BinpackDataset]):
        if not datasets:
            raise ValueError("MultiBinpackDataset needs at least one file")
        self.datasets = datasets

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, item: int):
        for part in self.datasets:
            if item < len(part):
                return part[item]
            item -= len(part)
        raise IndexError(item)

    def close(self) -> None:
        for part in self.datasets:
            part.close()

    def describe(self) -> dict:
        return {"files": [str(d.path) for d in self.datasets],
                "records": len(self)}
