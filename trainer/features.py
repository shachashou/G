"""Authoritative 450-dim NNUE feature-index and D4 symmetry specification.

This module is the single source of truth shared by the Python trainer and the
C++ engine (ticket 09). It is deliberately dependency-free so the C++ parity
tests can be written against exactly these definitions without importing torch.

Index convention (absolute-colour features, NO perspective flip)::

    cell       = y * 15 + x        x grows right, y grows down, both 0..14
    black cell -> feature index 0   .. 224   (= cell)
    white cell -> feature index 225 .. 449   (= 225 + cell)

The side to move is NOT a feature: it selects one of the two output heads
(0 = black to move, 1 = white to move). This keeps Renju's black/white
asymmetry explicit and makes incremental accumulator updates trivial.

D4 symmetry (8 transforms)::

    t   name                (x, y) -> (x', y')
    0   identity            (x, y)
    1   rot90  clockwise    (14 - y, x)
    2   rot180              (14 - x, 14 - y)
    3   rot270              (y, 14 - x)
    4   flip x              (14 - x, y)
    5   flip x + rot90      (14 - y, 14 - x)
    6   flip y              (x, 14 - y)
    7   diagonal transpose  (y, x)

Renju rules are invariant under all eight transforms (five-in-a-row,
overline, double-four, double-three all depend only on line geometry), so
score and WDL labels are unchanged by augmentation. Only the feature indices
are permuted, and the colour partition is preserved.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

BOARD_SIZE = 15
CELLS = BOARD_SIZE * BOARD_SIZE
NUM_FEATURES = 2 * CELLS
NUM_SYMMETRIES = 8

BLACK_PLANE = 0
WHITE_PLANE = 1

_LAST = BOARD_SIZE - 1

_TRANSFORMS: tuple[Callable[[int, int], tuple[int, int]], ...] = (
    lambda x, y: (x, y),
    lambda x, y: (_LAST - y, x),
    lambda x, y: (_LAST - x, _LAST - y),
    lambda x, y: (y, _LAST - x),
    lambda x, y: (_LAST - x, y),
    lambda x, y: (_LAST - y, _LAST - x),
    lambda x, y: (x, _LAST - y),
    lambda x, y: (y, x),
)  # type: ignore[assignment]


def _build_permutations() -> tuple[tuple[int, ...], ...]:
    table = []
    for transform in _TRANSFORMS:
        row = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                nx, ny = transform(x, y)
                row.append(ny * BOARD_SIZE + nx)
        table.append(tuple(row))
    return tuple(table)


# PERMUTATIONS[t][cell] is the transformed cell index.
PERMUTATIONS: tuple[tuple[int, ...], ...] = _build_permutations()


def transform_cell(cell: int, t: int) -> int:
    """Apply D4 transform t to a board cell index."""
    if not 0 <= cell < CELLS:
        raise ValueError(f"cell out of range: {cell}")
    if not 0 <= t < NUM_SYMMETRIES:
        raise ValueError(f"transform out of range: {t}")
    return PERMUTATIONS[t][cell]


def feature_index(cell: int, plane: int) -> int:
    """plane: 0 = black, 1 = white (or ``Color``-style 1/2, see below)."""
    if plane not in (BLACK_PLANE, WHITE_PLANE):
        raise ValueError(f"plane must be 0 (black) or 1 (white), got {plane}")
    return plane * CELLS + cell


def feature_index_of_color(cell: int, color: int) -> int:
    """colour uses the engine's Cell enum values: 1 = black, 2 = white."""
    if color == 1:
        return cell
    if color == 2:
        return CELLS + cell
    raise ValueError(f"colour must be 1 (black) or 2 (white), got {color}")


def split_feature(index: int) -> tuple[int, int]:
    """Inverse of feature_index(): (plane, cell)."""
    if not 0 <= index < NUM_FEATURES:
        raise ValueError(f"feature index out of range: {index}")
    return divmod(index, CELLS)


def sparse_indices(black_cells: Iterable[int],
                   white_cells: Iterable[int]) -> list[int]:
    """Sorted 450-dim sparse index set for a position."""
    result = {cell for cell in black_cells}
    result.update(CELLS + cell for cell in white_cells)
    for index in result:
        if not 0 <= index < NUM_FEATURES:
            raise ValueError(f"feature index out of range: {index}")
    return sorted(result)


def transform_feature(index: int, t: int) -> int:
    """D4 transform applied to a 450-dim feature index (colour preserved)."""
    plane, cell = split_feature(index)
    return plane * CELLS + transform_cell(cell, t)


def transform_sparse(indices: Sequence[int], t: int) -> list[int]:
    """D4 transform of a sparse index set; returns a sorted list."""
    return sorted({transform_feature(i, t) for i in indices})


def dense_vector(indices: Iterable[int]) -> list[float]:
    """450-dim 0/1 dense vector (used by the trainer's DataLoader)."""
    vector = [0.0] * NUM_FEATURES
    for index in indices:
        if not 0 <= index < NUM_FEATURES:
            raise ValueError(f"feature index out of range: {index}")
        vector[index] = 1.0
    return vector


def symmetry_table() -> tuple[tuple[int, ...], ...]:
    """450-dim gather permutation per transform: out[i] = in[table[t][i]]."""
    tables = []
    for perm in PERMUTATIONS:
        inverse = [0] * CELLS
        for cell, transformed in enumerate(perm):
            inverse[transformed] = cell
        table = []
        for plane in (BLACK_PLANE, WHITE_PLANE):
            table.extend(plane * CELLS + cell for cell in inverse)
        tables.append(tuple(table))
    return tuple(tables)
