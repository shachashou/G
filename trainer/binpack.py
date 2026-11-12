"""Python read end of the ticket-07 self-play binary format.

The C++ worker (``engine/src/binpack.h``) writes fixed 64-byte records; this
module is the authoritative, dependency-free reader used by ``tools/datagen.py``
for validation/merge and later by the ticket-08 DataLoader.

File layout (little-endian)::

    [0, 64)            file header (magic "GOMOKU07" + version + config)
    [64, 64 + 64 * N) N records

Record layout::

    [0, 29)    black occupancy, 225 bits LSB-first, bit i == cell (i % 15, i // 15)
    [29, 58)   white occupancy, same encoding
    [58, 60)   int16 search score from the side-to-move perspective, saturated
    [60]       flags: bits 0-1 WDL, bit 2 stm, bit 3 game-start, bit 4 game-end
    [61]       ply (stones on the board)
    [62, 64)   uint16 move actually played (0..224), 0xFFFF = none

WDL is relative to the side to move: 0 = loss, 1 = draw, 2 = win,
3 = unknown (only legal while a worker is still playing that game).
"""

import argparse
import json
import mmap
import os
import struct
import sys
from array import array
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Iterable
from typing import Iterator

MAGIC = b"GOMOKU07"
VERSION = 1
HEADER_SIZE = 64
RECORD_SIZE = 64
PLANE_BYTES = 29
BOARD_SIZE = 15
CELLS = BOARD_SIZE * BOARD_SIZE
MERGED_SHARD_ID = 0xFFFFFFFF
NO_MOVE = 0xFFFF
SCORE_MAX = 32767

WDL_LOSS, WDL_DRAW, WDL_WIN, WDL_UNKNOWN = 0, 1, 2, 3
STM_BLACK, STM_WHITE = 0, 1
WDL_MASK = 0x03
STM_BIT = 0x04
GAME_START_BIT = 0x08
GAME_END_BIT = 0x10
RESERVED_MASK = 0xE0

BLACK_PLANE_OFFSET = 0
WHITE_PLANE_OFFSET = PLANE_BYTES
SCORE_OFFSET = 58
FLAGS_OFFSET = 60
PLY_OFFSET = 61
MOVE_OFFSET = 62

HEADER_STRUCT = struct.Struct("<8s6I3QBBHHH")
RECORD_TAIL_STRUCT = struct.Struct("<hBBH")
assert HEADER_STRUCT.size == HEADER_SIZE
assert RECORD_TAIL_STRUCT.size == RECORD_SIZE - SCORE_OFFSET
assert PLANE_BYTES * 8 >= CELLS


class BinpackError(Exception):
    """Malformed or inconsistent binpack file."""


@dataclass(frozen=True)
class Header:
    version: int = VERSION
    header_size: int = HEADER_SIZE
    record_size: int = RECORD_SIZE
    board_size: int = BOARD_SIZE
    search_depth: int = 6
    shard_id: int = 0
    record_count: int = 0
    game_count: int = 0
    base_seed: int = 0
    rule_mode: int = 2
    use_vcf: int = 1
    tt_mb: int = 16
    vcf_node_limit: int = 5000
    opening_temp_x100: int = 100


@dataclass(frozen=True)
class Record:
    index: int
    black: bytes
    white: bytes
    score: int
    wdl: int
    stm: int
    game_start: bool
    game_end: bool
    ply: int
    move: int

    def black_cells(self) -> tuple[int, ...]:
        return tuple(i for i, bit in enumerate(unpack_plane(self.black)) if bit)

    def white_cells(self) -> tuple[int, ...]:
        return tuple(i for i, bit in enumerate(unpack_plane(self.white)) if bit)

    def feature_indices(self) -> list[int]:
        """450-dim absolute-colour NNUE input: black 0..224, white 225..449."""
        black = unpack_plane(self.black)
        white = unpack_plane(self.white)
        return [i for i, bit in enumerate(black) if bit] + [
            CELLS + i for i, bit in enumerate(white) if bit
        ]


def pack_header(header: Header) -> bytes:
    return HEADER_STRUCT.pack(
        MAGIC,
        header.version,
        header.header_size,
        header.record_size,
        header.board_size,
        header.search_depth,
        header.shard_id,
        header.record_count,
        header.game_count,
        header.base_seed,
        header.rule_mode,
        header.use_vcf,
        header.tt_mb,
        header.vcf_node_limit,
        header.opening_temp_x100,
    )


def unpack_header(raw: bytes) -> Header:
    if len(raw) != HEADER_SIZE:
        raise BinpackError("truncated binpack header")
    fields = HEADER_STRUCT.unpack(raw)
    magic = fields[0]
    if magic != MAGIC:
        raise BinpackError(f"bad binpack magic {magic!r}, expected {MAGIC!r}")
    h = Header(*fields[1:])
    if h.version != VERSION:
        raise BinpackError(f"unsupported binpack version {h.version}")
    if h.header_size != HEADER_SIZE:
        raise BinpackError(f"bad header size {h.header_size}")
    if h.record_size != RECORD_SIZE:
        raise BinpackError(f"bad record size {h.record_size}")
    if h.board_size != BOARD_SIZE:
        raise BinpackError(f"unsupported board size {h.board_size}")
    if not 1 <= h.search_depth <= 32:
        raise BinpackError(f"implausible search depth {h.search_depth}")
    if h.rule_mode not in (1, 2):
        raise BinpackError(f"bad rule mode {h.rule_mode}")
    if h.use_vcf not in (0, 1) or not 1 <= h.tt_mb <= 4096:
        raise BinpackError("bad search config in header")
    return h


def read_header(path: str | os.PathLike[str]) -> Header:
    with open(path, "rb") as handle:
        return unpack_header(handle.read(HEADER_SIZE))


def unpack_plane(plane: bytes) -> bytes:
    """Return 225 bytes of 0/1 values; raise on non-zero padding bits."""
    if len(plane) != PLANE_BYTES:
        raise BinpackError(f"occupancy plane must be {PLANE_BYTES} bytes")
    if plane[-1] & 0xFE:
        raise BinpackError("non-zero padding bits in occupancy plane")
    return bytes((plane[i >> 3] >> (i & 7)) & 1 for i in range(CELLS))


def _decode(buf: bytes | mmap.mmap, index: int) -> Record:
    offset = HEADER_SIZE + index * RECORD_SIZE
    black = bytes(buf[offset + BLACK_PLANE_OFFSET:offset + BLACK_PLANE_OFFSET + PLANE_BYTES])
    white = bytes(buf[offset + WHITE_PLANE_OFFSET:offset + WHITE_PLANE_OFFSET + PLANE_BYTES])
    score, flags, ply, move = RECORD_TAIL_STRUCT.unpack_from(buf, offset + SCORE_OFFSET)
    unpack_plane(black)
    unpack_plane(white)
    if flags & RESERVED_MASK:
        raise BinpackError("non-zero reserved flag bits")
    if score == -32768:
        raise BinpackError("reserved score -32768")
    if ply > CELLS:
        raise BinpackError("ply out of range")
    if move != NO_MOVE and move >= CELLS:
        raise BinpackError("move out of range")
    return Record(
        index=index,
        black=black,
        white=white,
        score=score,
        wdl=flags & WDL_MASK,
        stm=STM_WHITE if flags & STM_BIT else STM_BLACK,
        game_start=bool(flags & GAME_START_BIT),
        game_end=bool(flags & GAME_END_BIT),
        ply=ply,
        move=move,
    )


class BinpackReader:
    """mmap-backed random access to one binpack file."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._file = open(self.path, "rb")
        try:
            self.header = unpack_header(self._file.read(HEADER_SIZE))
            size = os.fstat(self._file.fileno()).st_size
            body = size - HEADER_SIZE
            if body < 0 or body % RECORD_SIZE:
                raise BinpackError("file size is not header + whole 64-byte records")
            self.available = body // RECORD_SIZE
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._file.close()
            raise

    def __enter__(self) -> "BinpackReader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "_mmap", None) is not None:
            self._mmap.close()
            self._mmap = None  # type: ignore[assignment]
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None  # type: ignore[assignment]

    def __len__(self) -> int:
        return self.available

    def __getitem__(self, index: int) -> Record:
        if index < 0:
            index += self.available
        if not 0 <= index < self.available:
            raise IndexError(index)
        return _decode(self._mmap, index)

    def iter_records(self, limit: int | None = None) -> Iterator[Record]:
        count = self.available if limit is None else min(limit, self.available)
        for i in range(count):
            yield self[i]

    def iter_games(self) -> Iterator[list[Record]]:
        game: list[Record] = []
        for record in self.iter_records():
            if record.game_start and game:
                yield game
                game = []
            game.append(record)
        if game:
            yield game

    def game_index_map(self) -> "array.array[int]":
        """For every record, the index of its game. Reads only the flags byte.

        Used by the trainer to split train/validation by WHOLE games without
        materialising every Record object (important for 10M-record files).
        """
        mapping = array("I")
        game = 0
        for index in range(self.available):
            offset = HEADER_SIZE + index * RECORD_SIZE + FLAGS_OFFSET
            flags = self._mmap[offset]
            if index > 0 and flags & GAME_START_BIT:
                game += 1
            mapping.append(game)
        return mapping

    def validate(self, max_records: int | None = None) -> dict:
        """Full structural + semantic validation; returns stats and error list."""
        errors: list[str] = []
        stats = {
            "path": str(self.path),
            "records": self.available,
            "header_records": self.header.record_count,
            "games": 0,
            "header_games": self.header.game_count,
            "partial": max_records is not None and max_records < self.available,
            "scanned": 0,
            "wdl_unknown": 0,
            "saturated_scores": 0,
            "min_ply": None,
            "max_ply": None,
            "decisive_games": 0,
            "draw_games": 0,
            "errors": errors,
        }
        if not stats["partial"] and self.available != self.header.record_count:
            errors.append(
                f"header says {self.header.record_count} records but file holds {self.available}"
            )

        game: list[Record] = []
        pending_end = False
        limit = self.available if max_records is None else min(max_records, self.available)
        for index in range(limit):
            try:
                record = self[index]
            except BinpackError as exc:
                errors.append(f"record {index}: {exc}")
                continue
            stats["scanned"] += 1
            black = unpack_plane(record.black)
            white = unpack_plane(record.white)
            stones = sum(black) + sum(white)
            overlap = sum(1 for b, w in zip(black, white) if b and w)
            if overlap:
                errors.append(f"record {index}: black/white planes overlap at {overlap} cells")
            if stones != record.ply:
                errors.append(f"record {index}: ply {record.ply} != {stones} stones")
            if record.stm != (record.ply & 1):
                errors.append(f"record {index}: side to move disagrees with ply")
            if record.move == NO_MOVE or black[record.move] or white[record.move]:
                errors.append(f"record {index}: invalid recorded move {record.move}")
            if record.wdl == WDL_UNKNOWN:
                stats["wdl_unknown"] += 1
                errors.append(f"record {index}: result was never backfilled")
            if record.score in (SCORE_MAX, -SCORE_MAX):
                stats["saturated_scores"] += 1
            stats["min_ply"] = record.ply if stats["min_ply"] is None else min(stats["min_ply"], record.ply)
            stats["max_ply"] = record.ply if stats["max_ply"] is None else max(stats["max_ply"], record.ply)

            if pending_end and not record.game_start:
                errors.append(f"record {index}: new game without game-start flag")
            if record.game_start:
                if game:
                    self._finish_game(game, stats, errors)
                game = []
            if not game and not record.game_start:
                errors.append(f"record {index}: first record of a game lacks game-start flag")
            if game and record.ply <= game[-1].ply:
                errors.append(f"record {index}: ply does not increase within a game")
            game.append(record)
            pending_end = record.game_end

        if game:
            if not stats["partial"]:
                self._finish_game(game, stats, errors)
            else:
                stats["games"] += 1
        if not stats["partial"] and stats["games"] != self.header.game_count:
            errors.append(
                f"header says {self.header.game_count} games but found {stats['games']}"
            )
        return stats

    @staticmethod
    def _finish_game(game: list[Record], stats: dict, errors: list[str]) -> None:
        stats["games"] += 1
        first = game[0]
        if not game[-1].game_end:
            errors.append(f"record {game[-1].index}: last record of a game lacks game-end flag")
        if first.wdl == WDL_DRAW:
            stats["draw_games"] += 1
            expected = {r: WDL_DRAW for r in range(len(game))}
        else:
            stats["decisive_games"] += 1
            winner = first.stm if first.wdl == WDL_WIN else (1 - first.stm)
            expected = {
                i: (WDL_WIN if record.stm == winner else WDL_LOSS)
                for i, record in enumerate(game)
            }
        for i, record in enumerate(game):
            if record.wdl != expected[i]:
                errors.append(
                    f"record {record.index}: WDL {record.wdl} inconsistent with game result "
                    f"(expected {expected[i]})"
                )
                break


def validate_file(path: str | os.PathLike[str], max_records: int | None = None) -> dict:
    with BinpackReader(path) as reader:
        return reader.validate(max_records=max_records)


def merge_files(inputs: Iterable[str | os.PathLike[str]], output: str | os.PathLike[str]) -> dict:
    """Concatenate shard record bodies into one file with a single header."""
    paths = [Path(p) for p in inputs]
    if not paths:
        raise BinpackError("no input shards")
    headers: list[Header] = []
    for path in paths:
        header = read_header(path)
        with BinpackReader(path) as reader:
            if reader.available != header.record_count:
                raise BinpackError(f"{path}: truncated or unfinished shard")
        headers.append(header)

    first = headers[0]
    comparable = (
        "version", "header_size", "record_size", "board_size", "search_depth",
        "rule_mode", "use_vcf", "tt_mb", "vcf_node_limit", "opening_temp_x100",
    )
    for path, header in zip(paths[1:], headers[1:]):
        for field in comparable:
            if getattr(header, field) != getattr(first, field):
                raise BinpackError(f"{path}: {field} differs from first shard")

    merged = replace(
        first,
        shard_id=MERGED_SHARD_ID,
        record_count=sum(h.record_count for h in headers),
        game_count=sum(h.game_count for h in headers),
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w+b") as dst:
        dst.write(pack_header(replace(merged, record_count=0, game_count=0)))
        for path in paths:
            with open(path, "rb") as src:
                src.seek(HEADER_SIZE)
                remaining = read_header(path).record_count * RECORD_SIZE
                while remaining > 0:
                    chunk = src.read(min(1 << 20, remaining))
                    if not chunk:
                        raise BinpackError(f"{path}: unexpected end of shard")
                    dst.write(chunk)
                    remaining -= len(chunk)
        dst.seek(0)
        dst.write(pack_header(merged))
    return {
        "output": str(out),
        "inputs": [str(p) for p in paths],
        "records": merged.record_count,
        "games": merged.game_count,
        "size_bytes": out.stat().st_size,
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="inspect/validate binpack files")
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("info")
    info.add_argument("path")
    check = sub.add_parser("validate")
    check.add_argument("path")
    check.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args(argv)

    if args.command == "info":
        print(json.dumps(read_header(args.path).__dict__, indent=2))
        return 0
    result = validate_file(args.path, max_records=args.max_records)
    print(json.dumps(result, indent=2))
    return 3 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(_main())
