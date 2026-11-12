#!/usr/bin/env python3
"""Rolling training sample pool with capacity, dedup and a frozen split (ticket 05).

The pool is one SQLite file. A game (or a standalone sample source) is
*staged* while it is being produced and becomes readable only when the writer
commits it in a single transaction:

- staged rows are invisible to readers and are deleted on ``abort_game``;
- a crash leaves the rows staged, never half-committed;
- readers take a snapshot (WAL), so evicting old games while a reader works
  does not pull rows out from under it.

Contents:

- games: provenance per source (game id, kind, model, result, commit time,
  split) and the frozen validation membership;
- samples: board, side, tactical status, supervision (teacher value or a later
  self-play result), teacher identity, dedup keys;
- counters and runs: cumulative generated/valid/deduped/evicted counts, filter
  reasons, and the measured effective rate of each collection run.

Dedup and leakage:

- an exact position (board + side + rule) is stored once;
- with ``dedup_d4`` (default on) a D4-equivalent board is also stored once, so
  a symmetry does not count as independent raw data;
- validation membership is frozen at the first commits; a training sample that
  is D4-equivalent to a validation sample is refused, and validation keys are
  kept in their own table.

Usage::

    from sample_pool import SamplePool

    with SamplePool("data/pool/pool.sqlite3", capacity=1000) as pool:
        game = pool.begin_game("game-1", kind="selfplay", model_id="handcrafted",
                               rule="renju")
        pool.add_sample(game, record)
        pool.commit_game(game, final_result="black_win")
        stats = pool.stats()

Command line::

    python tools/sample_pool.py stats data/pool/pool.sqlite3
    python tools/sample_pool.py validate data/pool/pool.sqlite3
    python tools/sample_pool.py evict data/pool/pool.sqlite3 --capacity 500
"""

import argparse
import hashlib
import json
import random
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

APPLICATION_ID = 0x47504F4C  # "GPOL"
SCHEMA_VERSION = 1
POOL_FORMAT = "gomoku-nnue-pool-v1"
SIZE = 15

DEFAULT_CONFIG = {
    "capacity": 1000,
    "validation_games": 1,
    "validation_seed": 0,
    "dedup_d4": True,
    "new_ratio": 0.5,
    "strategy_version": "collect-v1",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS counters (
    name TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL UNIQUE,
    source_kind TEXT NOT NULL CHECK (source_kind IN
        ('selfplay', 'teacher', 'standalone', 'import')),
    model_id TEXT,
    rule TEXT NOT NULL CHECK (rule IN ('renju')),
    created_at TEXT NOT NULL,
    committed_at TEXT,
    final_result TEXT CHECK (final_result IS NULL OR
                             final_result IN ('black_win', 'white_win', 'draw')),
    status TEXT NOT NULL CHECK (status IN ('staged', 'committed', 'aborted')),
    split TEXT CHECK (split IN ('train', 'validation')),
    sample_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    board BLOB NOT NULL CHECK (length(board) = 225),
    side_to_move TEXT NOT NULL CHECK (side_to_move IN ('black', 'white')),
    rule TEXT NOT NULL CHECK (rule IN ('renju')),
    tactical_status TEXT NOT NULL,
    label_source TEXT NOT NULL CHECK (label_source IN
        ('teacher', 'selfplay_result', 'none')),
    label_version TEXT NOT NULL,
    raw_eval TEXT,
    raw_score INTEGER,
    raw_perspective TEXT,
    normalization TEXT,
    normalization_param REAL,
    value REAL,
    result_value REAL,
    suggested_move TEXT,
    depth INTEGER,
    budget_ms INTEGER,
    threads INTEGER,
    teacher_release TEXT,
    teacher_version TEXT,
    teacher_exe_sha256 TEXT,
    teacher_weights_sha256 TEXT,
    teacher_config_sha256 TEXT,
    dedup_key TEXT NOT NULL UNIQUE,
    d4_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK (label_source != 'teacher' OR value IS NOT NULL),
    CHECK (label_source != 'selfplay_result' OR result_value IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS samples_game ON samples (game_id);
CREATE INDEX IF NOT EXISTS samples_d4 ON samples (d4_key);
CREATE TABLE IF NOT EXISTS validation_keys (
    d4_key TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    config TEXT NOT NULL,
    generated INTEGER NOT NULL DEFAULT 0,
    valid INTEGER NOT NULL DEFAULT 0,
    inserted INTEGER NOT NULL DEFAULT 0,
    filters TEXT,
    effective_per_second REAL,
    note TEXT
);
"""

REQUIRED_COLUMNS = {
    "id", "game_id", "board", "side_to_move", "rule", "tactical_status",
    "label_source", "label_version", "raw_eval", "raw_score",
    "raw_perspective", "normalization", "normalization_param", "value",
    "result_value", "suggested_move", "depth", "budget_ms", "threads",
    "teacher_release", "teacher_version", "teacher_exe_sha256",
    "teacher_weights_sha256", "teacher_config_sha256", "dedup_key", "d4_key",
    "created_at",
}

FILTER_COUNTER = "filter:"


class PoolError(RuntimeError):
    """The pool is missing, damaged or not compatible with this reader."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _expected_side(cells):
    black = sum(1 for c in cells if c == 1)
    white = sum(1 for c in cells if c == 2)
    if black == white:
        return "black"
    if black == white + 1:
        return "white"
    return None


def _transform(cells, transform: int) -> tuple:
    out = [0] * 225
    for index, value in enumerate(cells):
        if not value:
            continue
        x, y = index % SIZE, index // SIZE
        if transform >= 4:
            x = SIZE - 1 - x
        for _ in range(transform % 4):
            x, y = y, SIZE - 1 - x
        out[y * SIZE + x] = value
    return tuple(out)


def exact_key(cells, side: str) -> str:
    digest = hashlib.sha256(bytes(cells))
    digest.update(b"|" + side.encode())
    return digest.hexdigest()


def d4_key(cells, side: str) -> str:
    best = min(_transform(cells, t) for t in range(8))
    digest = hashlib.sha256(bytes(best))
    digest.update(b"|" + side.encode())
    return digest.hexdigest()


class SamplePool:
    def __init__(self, path: str | Path, *, capacity: int | None = None,
                 create: bool = True, config: dict | None = None) -> None:
        self.path = Path(path)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() and not create:
            raise PoolError(f"sample pool does not exist: {self.path}")
        try:
            self._conn = sqlite3.connect(str(self.path), timeout=15,
                                         check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error as exc:
            raise PoolError(f"cannot open sample pool {self.path}: {exc}") from exc
        try:
            self._prepare(create)
            if capacity is not None:
                self.set_capacity(int(capacity))
        except Exception:
            self._conn.close()
            raise

    # -- lifecycle -------------------------------------------------------

    def _prepare(self, create: bool) -> None:
        try:
            app_id = int(self._conn.execute("PRAGMA application_id").fetchone()[0])
            version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            tables = {row[0] for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}
        except sqlite3.DatabaseError as exc:
            raise PoolError(
                f"not a sample pool or the file is corrupt: {self.path} "
                f"({exc})") from exc
        if not tables:
            if not create:
                raise PoolError(f"empty sample pool: {self.path}")
            self._create_schema()
            return
        if app_id != APPLICATION_ID:
            raise PoolError(
                f"{self.path} is not a sample pool "
                f"(application_id={app_id:#x}, expected {APPLICATION_ID:#x})")
        if version > SCHEMA_VERSION:
            raise PoolError(
                f"pool schema {version} is newer than this reader "
                f"({SCHEMA_VERSION}); refusing to guess")
        if version < SCHEMA_VERSION:
            raise PoolError(
                f"pool schema {version} is older than this reader "
                f"({SCHEMA_VERSION}); migration is not implemented")
        meta = dict(self._conn.execute("SELECT key, value FROM meta"))
        if meta.get("pool_format") != POOL_FORMAT:
            raise PoolError(
                f"unexpected pool format {meta.get('pool_format')!r}; "
                f"expected {POOL_FORMAT!r}")
        columns = {row[1] for row in
                   self._conn.execute("PRAGMA table_info(samples)")}
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise PoolError(
                f"pool schema is missing columns: {', '.join(sorted(missing))}")

    def _create_schema(self) -> None:
        with self._conn:
            self._conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(SCHEMA)
            config = dict(DEFAULT_CONFIG)
            config.update({"created_at": _now()})
            self._conn.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                [("pool_format", POOL_FORMAT),
                 ("schema_version", str(SCHEMA_VERSION)),
                 ("config", json.dumps(config, ensure_ascii=False)),
                 ("validation", json.dumps({"source_ids": [], "frozen_at": None})),
                 ("eviction_policy",
                  "oldest committed training game first; validation games are "
                  "never evicted")])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SamplePool":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- config / counters ----------------------------------------------

    def config(self) -> dict:
        raw = dict(self._conn.execute("SELECT key, value FROM meta"))
        config = json.loads(raw.get("config", "{}"))
        config["validation"] = json.loads(raw.get("validation", "{}"))
        config["capacity"] = int(config.get("capacity", 0))
        return config

    def set_capacity(self, capacity: int) -> None:
        if capacity < 1:
            raise PoolError("capacity must be positive")
        with self._conn:
            current = json.loads(dict(self._conn.execute(
                "SELECT key, value FROM meta"))["config"])
            current["capacity"] = int(capacity)
            self._conn.execute("UPDATE meta SET value = ? WHERE key = 'config'",
                               (json.dumps(current, ensure_ascii=False),))

    def set_config(self, key: str, value) -> None:
        """Records a run-configuration value; capacity has its own setter."""
        if key == "capacity":
            self.set_capacity(int(value))
            return
        with self._conn:
            raw = json.loads(dict(self._conn.execute(
                "SELECT key, value FROM meta"))["config"])
            raw[key] = value
            self._conn.execute("UPDATE meta SET value = ? WHERE key = 'config'",
                               (json.dumps(raw, ensure_ascii=False),))

    def _bump(self, name: str, amount: int = 1) -> None:
        if amount == 0:
            return
        self._conn.execute(
            "INSERT INTO counters (name, value) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET value = value + excluded.value",
            (name, int(amount)))

    def record_filter(self, reason: str, amount: int = 1) -> None:
        """Counts a filtered candidate by reason (mate, vcf_proven, ...)."""
        with self._conn:
            self._bump(f"{FILTER_COUNTER}{reason}", amount)

    def record_generated(self, amount: int = 1) -> None:
        """Counts candidate positions the collector considered."""
        with self._conn:
            self._bump("generated", amount)

    def record_observations(self, generated: int, filters: dict) -> None:
        """Flush one game's observation counters in a single durable transaction."""
        with self._conn:
            self._bump("generated", generated)
            for reason, count in filters.items():
                self._bump(f"{FILTER_COUNTER}{reason}", count)

    def counters(self) -> dict:
        return {row[0]: int(row[1]) for row in
                self._conn.execute("SELECT name, value FROM counters")}

    @staticmethod
    def _label_clause(label_sources):
        if label_sources is None:
            return "", ()
        sources = tuple(sorted(set(label_sources)))
        if not sources or set(sources) - {"teacher", "selfplay_result", "none"}:
            raise PoolError("invalid supervision sources")
        return " AND s.label_source IN (" + ",".join("?" for _ in sources) + ")", sources

    def counts(self, label_sources=None) -> dict:
        clause, params = self._label_clause(label_sources)
        rows = self._conn.execute(
            "SELECT g.split AS split, COUNT(*) AS n, COUNT(DISTINCT g.id) AS games FROM samples s "
            "JOIN games g ON g.id = s.game_id "
            "WHERE g.status = 'committed'" + clause + " GROUP BY g.split", params)
        sizes = {"train": 0, "validation": 0}
        games = {"train": 0, "validation": 0}
        for row in rows:
            sizes[row["split"]] = int(row["n"])
            games[row["split"]] = int(row["games"])
        return {
            "pool_samples": sizes["train"] + sizes["validation"],
            "train_samples": sizes["train"],
            "validation_samples": sizes["validation"],
            "train_games": games["train"],
            "validation_game_count": games["validation"],
            "capacity": int(self.config()["capacity"]),
        }

    def stats(self, label_sources=None) -> dict:
        config = self.config()
        counters = self.counters()
        counts = self.counts(label_sources)
        total = self.counts()
        committed = (counters.get("committed_training_samples", 0) if label_sources is None
                     else sum(counters.get("committed_training_samples:" + s, 0) for s in set(label_sources)))
        validation = config["validation"].get("source_ids", [])
        runs = self._conn.execute(
            "SELECT COUNT(*) FROM runs").fetchone()[0]
        return {
            **counts,
            "total_pool_samples": total["pool_samples"],
            "total_validation_samples": total["validation_samples"],
            "label_sources": sorted(set(label_sources)) if label_sources is not None else None,
            "generated": counters.get("generated", 0),
            "valid": counters.get("valid", 0),
            "inserted": counters.get("inserted", 0),
            "committed_training_samples": committed,
            "commit_sequence": counters.get("commit_sequence", 0),
            "duplicate": counters.get("duplicate", 0),
            "d4_duplicate": counters.get("d4_duplicate", 0),
            "d4_leak": counters.get("d4_leak", 0),
            "evicted_games": counters.get("evicted_games", 0),
            "evicted_samples": counters.get("evicted_samples", 0),
            "committed_games": self._conn.execute(
                "SELECT COUNT(*) FROM games WHERE status = 'committed'"
            ).fetchone()[0],
            "staged_games": self._conn.execute(
                "SELECT COUNT(*) FROM games WHERE status = 'staged'"
            ).fetchone()[0],
            "filters": {name[len(FILTER_COUNTER):]: int(value)
                        for name, value in counters.items()
                        if name.startswith(FILTER_COUNTER)},
            "validation_games": list(validation),
            "validation_frozen_at": config["validation"].get("frozen_at"),
            "effective_per_second": self._last_rate(),
            "d4_augmentations": 0,
            "runs": int(runs),
            "utilization": (total["pool_samples"] / counts["capacity"]
                            if counts["capacity"] else 0.0),
        }

    def _last_rate(self) -> float:
        row = self._conn.execute(
            "SELECT effective_per_second FROM runs "
            "WHERE effective_per_second IS NOT NULL "
            "ORDER BY id DESC LIMIT 1").fetchone()
        return float(row[0]) if row else 0.0

    # -- staging / commit ------------------------------------------------

    def begin_game(self, source_id: str, *, kind: str = "selfplay",
                   model_id: str | None = None, rule: str = "renju") -> int:
        if not source_id:
            raise PoolError("source_id is required")
        if kind not in ("selfplay", "teacher", "standalone", "import"):
            raise PoolError(f"unknown source kind {kind!r}")
        if rule != "renju":
            raise PoolError("sample pool accepts Renju games only")
        try:
            with self._conn:
                cursor = self._conn.execute(
                    "INSERT INTO games (source_id, source_kind, model_id, rule,"
                    " created_at, status) VALUES (?, ?, ?, ?, ?, 'staged')",
                    (source_id, kind, model_id, rule, _now()))
                return int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise PoolError(f"source_id {source_id!r} already exists") from exc

    def abort_game(self, game_id: int) -> bool:
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM games WHERE id = ? AND status = 'staged'",
                (int(game_id),))
            return cursor.rowcount > 0

    def abort_staged_games(self) -> int:
        """Drops staged rows left by a crashed producer; committed data is kept."""
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM games WHERE status = 'staged'")
            return int(cursor.rowcount)

    @contextmanager
    def _sample_transaction(self):
        if self._conn.in_transaction:
            yield
        else:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                yield

    def add_samples(self, game_id: int, records) -> list[str]:
        """Stage a whole game's labels atomically; commit_game still publishes it."""
        with self._sample_transaction():
            return [self.add_sample(game_id, record) for record in records]

    def add_sample(self, game_id: int, record: dict) -> str:
        """Stages one sample; returns inserted|duplicate|d4_duplicate."""
        now = _now()
        board = record.get("board")
        if not isinstance(board, (bytes, bytearray)) or len(board) != 225:
            raise PoolError("record.board must be 225 bytes")
        if any(value not in (0, 1, 2) for value in board):
            raise PoolError("record.board values must be 0/1/2")
        cells = list(board)
        side = record.get("side_to_move")
        if side not in ("black", "white"):
            raise PoolError("record.side_to_move must be black or white")
        expected = _expected_side(cells)
        if expected is None or side != expected:
            raise PoolError("side to move does not match the stone count")
        if record.get("rule") != "renju":
            raise PoolError("sample pool accepts Renju samples only")
        label_source = record.get("label_source")
        if label_source not in ("teacher", "selfplay_result", "none"):
            raise PoolError("label_source must be teacher, selfplay_result or none")
        if label_source == "teacher" and record.get("value") is None:
            raise PoolError("teacher samples require a normalized value")
        if label_source == "selfplay_result" and record.get("result_value") is None:
            raise PoolError("result samples require a result value")
        dedup = exact_key(cells, side)
        canonical = d4_key(cells, side)
        with self._sample_transaction():
            # Dedup check and insert must share the same writer reservation:
            # different collectors can stage symmetric positions concurrently.
            game = self._conn.execute(
                "SELECT status FROM games WHERE id = ?", (int(game_id),)).fetchone()
            if game is None:
                raise PoolError(f"game {game_id} does not exist")
            if game["status"] != "staged":
                raise PoolError(f"game {game_id} is no longer staged")
            if self._conn.execute("SELECT 1 FROM samples WHERE dedup_key = ?",
                                  (dedup,)).fetchone():
                self._bump("duplicate")
                return "duplicate"
            if self.config().get("dedup_d4", True):
                if self._conn.execute("SELECT 1 FROM samples WHERE d4_key = ?",
                                      (canonical,)).fetchone():
                    self._bump("d4_duplicate")
                    return "d4_duplicate"
            self._bump("valid")
            try:
                self._conn.execute(
                    "INSERT INTO samples (game_id, board, side_to_move, rule,"
                    " tactical_status, label_source, label_version, raw_eval,"
                    " raw_score, raw_perspective, normalization,"
                    " normalization_param, value, result_value, suggested_move,"
                    " depth, budget_ms, threads, teacher_release,"
                    " teacher_version, teacher_exe_sha256,"
                    " teacher_weights_sha256, teacher_config_sha256, dedup_key,"
                    " d4_key, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
                    " ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (int(game_id), bytes(cells), side, "renju",
                     str(record.get("tactical_status", "unknown")), label_source,
                     str(record.get("label_version", "teacher-v1")),
                     record.get("raw_eval"), record.get("raw_score"),
                     record.get("raw_perspective"), record.get("normalization"),
                     record.get("normalization_param"), record.get("value"),
                     record.get("result_value"), record.get("suggested_move"),
                     record.get("depth"), record.get("budget_ms"),
                     record.get("threads"), record.get("teacher_release"),
                     record.get("teacher_version"),
                     record.get("teacher_exe_sha256"),
                     record.get("teacher_weights_sha256"),
                     record.get("teacher_config_sha256"), dedup, canonical, now))
            except sqlite3.IntegrityError:
                self._bump("duplicate")
                return "duplicate"
            self._bump("inserted")
        return "inserted"

    def commit_game(self, game_id: int, *,
                    final_result: str | None = None, validation_group: str | None = None) -> dict:
        """Makes a staged source readable and enforces capacity."""
        if final_result is not None and final_result not in (
                "black_win", "white_win", "draw"):
            raise PoolError("final_result must be NULL, black_win, white_win or draw")
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            game = self._conn.execute(
                "SELECT * FROM games WHERE id = ?", (int(game_id),)).fetchone()
            if game is None:
                raise PoolError(f"game {game_id} does not exist")
            if game["status"] != "staged":
                raise PoolError(f"game {game_id} is not staged")
            sample_count = self._conn.execute(
                "SELECT COUNT(*) FROM samples WHERE game_id = ?",
                (int(game_id),)).fetchone()[0]
            result_rows = self._conn.execute(
                "SELECT result_value FROM samples WHERE game_id = ? AND label_source = 'selfplay_result'",
                (int(game_id),)).fetchall()
            expected_result = {"black_win": 1.0, "white_win": -1.0, "draw": 0.0}.get(final_result)
            if result_rows and (expected_result is None or any(r[0] != expected_result for r in result_rows)):
                raise PoolError("result supervision requires a completed game and matching black-absolute outcome")
            config = self.config()
            validation = config["validation"]
            frozen = list(validation.get("source_ids", []))
            frozen_count = len(frozen)
            if validation_group is not None:
                self._label_clause([validation_group])
                frozen_count = self._conn.execute(
                    "SELECT COUNT(*) FROM games g WHERE g.status = 'committed' AND g.split = 'validation' "
                    "AND EXISTS (SELECT 1 FROM samples s WHERE s.game_id = g.id AND s.label_source = ?)",
                    (validation_group,)).fetchone()[0]
            if (sample_count > 0
                    and frozen_count < int(config.get("validation_games", 0))):
                split = "validation"
                frozen.append(game["source_id"])
                validation["source_ids"] = frozen
                if not validation.get("frozen_at"):
                    validation["frozen_at"] = _now()
                self._conn.execute(
                    "UPDATE meta SET value = ? WHERE key = 'validation'",
                    (json.dumps(validation, ensure_ascii=False),))
            else:
                split = "train"
            if split == "train":
                leaked = self._conn.execute(
                    "SELECT COUNT(*) FROM samples s JOIN validation_keys v "
                    "ON v.d4_key = s.d4_key WHERE s.game_id = ?",
                    (int(game_id),)).fetchone()[0]
                if leaked:
                    self._conn.execute(
                        "DELETE FROM samples WHERE game_id = ? AND d4_key IN "
                        "(SELECT d4_key FROM validation_keys)", (int(game_id),))
                    self._bump("d4_leak", leaked)
                    self._bump("evicted_samples", leaked)
                    sample_count -= leaked
            else:
                keys = [row[0] for row in self._conn.execute(
                    "SELECT d4_key FROM samples WHERE game_id = ?",
                    (int(game_id),))]
                for key in keys:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO validation_keys (d4_key) "
                        "VALUES (?)", (key,))
                leaked = self._conn.execute(
                    "SELECT COUNT(*) FROM samples s JOIN games g "
                    "ON g.id = s.game_id WHERE g.split = 'train' "
                    "AND g.status = 'committed' AND s.d4_key IN "
                    "(SELECT d4_key FROM validation_keys WHERE d4_key IN "
                    " (SELECT d4_key FROM samples WHERE game_id = ?))",
                    (int(game_id),)).fetchone()[0]
                if leaked:
                    # Global D4 dedup normally prevents this; if it was
                    # disabled, drop the training copies rather than leak.
                    self._conn.execute(
                        "DELETE FROM samples WHERE game_id IN "
                        "(SELECT id FROM games WHERE split = 'train' "
                        " AND status = 'committed') AND d4_key IN "
                        "(SELECT d4_key FROM samples WHERE game_id = ?)",
                        (int(game_id),))
                    self._bump("d4_leak", leaked)
                    self._bump("evicted_samples", leaked)
            self._conn.execute(
                "UPDATE games SET status = 'committed', split = ?,"
                " committed_at = ?, final_result = ?, sample_count = ?"
                " WHERE id = ?",
                (split, _now(), final_result, sample_count, int(game_id)))
            evicted = self._enforce_capacity()
            self._bump("commit_sequence")
            if split == "train":
                self._bump("committed_training_samples", sample_count)
                for source, count in self._conn.execute(
                        "SELECT label_source, COUNT(*) FROM samples WHERE game_id = ? GROUP BY label_source",
                        (int(game_id),)).fetchall():
                    self._bump("committed_training_samples:" + source, count)
            self._conn.execute("COMMIT")
            return {"game_id": int(game_id), "split": split,
                    "samples": sample_count, "evicted": evicted}
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def _enforce_capacity(self) -> dict:
        capacity = int(self.config()["capacity"])
        evicted_games = 0
        evicted_samples = 0
        while True:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM samples s JOIN games g "
                "ON g.id = s.game_id WHERE g.status = 'committed'"
            ).fetchone()[0]
            if total <= capacity:
                break
            victim = self._conn.execute(
                "SELECT g.id, g.sample_count FROM games g "
                "WHERE g.status = 'committed' AND g.split = 'train' "
                "ORDER BY g.committed_at, g.id LIMIT 1").fetchone()
            if victim is None:
                # Only validation games remain: keep them and record overflow.
                self._bump("overflow_samples", total - capacity)
                break
            self._conn.execute("DELETE FROM games WHERE id = ?", (victim[0],))
            evicted_games += 1
            evicted_samples += int(victim[1])
        if evicted_games:
            self._bump("evicted_games", evicted_games)
            self._bump("evicted_samples", evicted_samples)
        return {"games": evicted_games, "samples": evicted_samples}

    # -- read ------------------------------------------------------------

    def training_snapshot(self, *, train_limit=None, new_ratio=None, seed=0, label_sources=None) -> dict:
        """One coherent WAL snapshot, including split and commit counters."""
        self._conn.execute("BEGIN")
        try:
            problems = self.validate()
            if problems:
                raise PoolError("pool failed validation: " + "; ".join(problems))
            return {"train": self.read_batch(split="train", limit=train_limit,
                                             new_ratio=new_ratio, seed=seed, label_sources=label_sources),
                    "validation": self.read_batch(split="validation", label_sources=label_sources),
                    "stats": self.stats(label_sources)}
        finally:
            self._conn.execute("ROLLBACK")

    def read_batch(self, *, split: str = "train", limit: int | None = None,
                   seed: int = 0, new_ratio: float | None = None, label_sources=None) -> list[dict]:
        """Committed samples from one split; mixed new/old when limited."""
        if split not in ("train", "validation"):
            raise PoolError("split must be train or validation")
        rows = self._fetch(split, label_sources)
        if limit is not None and limit < len(rows):
            ratio = (self.config().get("new_ratio", 0.5)
                     if new_ratio is None else new_ratio)
            rows = self._mix(rows, limit, ratio, seed)
        return rows

    def _fetch(self, split: str, label_sources=None) -> list[dict]:
        clause, params = self._label_clause(label_sources)
        cursor = self._conn.execute(
            "SELECT s.*, g.source_id, g.source_kind, g.final_result,"
            " g.model_id FROM samples s JOIN games g ON g.id = s.game_id "
            "WHERE g.status = 'committed' AND g.split = ?" + clause + " ORDER BY s.id",
            (split, *params))
        out = []
        for row in cursor:
            record = dict(row)
            record["board"] = list(record["board"])
            out.append(record)
        return out

    @staticmethod
    def _mix(rows: list[dict], limit: int, ratio: float, seed: int) -> list[dict]:
        rng = random.Random(seed)
        # AUTOINCREMENT sample ids track insertion order; source names (and
        # decimal seeds within them) do not encode chronological recency.
        ordered = sorted(rows, key=lambda r: r["id"])
        newest = ordered[-max(1, len(ordered) // 5):]
        older = [r for r in ordered if r not in newest]
        want_new = min(len(newest), max(0, round(limit * ratio)))
        new = rng.sample(newest, want_new) if want_new else []
        rest = min(limit - len(new), len(older))
        old = rng.sample(older, rest) if rest else []
        picked = new + old
        if len(picked) < limit:
            pool = [r for r in ordered if r not in picked]
            picked += rng.sample(pool, min(limit - len(picked), len(pool)))
        return picked

    @contextmanager
    def read_snapshot(self):
        """A read transaction; evictions elsewhere do not change its rows."""
        cursor = self._conn.cursor()
        cursor.execute("BEGIN")
        try:
            def snapshot() -> list[dict]:
                out = []
                for row in cursor.execute(
                        "SELECT s.* FROM samples s JOIN games g "
                        "ON g.id = s.game_id WHERE g.status = 'committed'"):
                    record = dict(row)
                    record["board"] = list(record["board"])
                    out.append(record)
                return out
            yield snapshot
        finally:
            cursor.execute("COMMIT")

    # -- runs / stats ----------------------------------------------------

    def start_run(self, config: dict) -> int:
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO runs (started_at, config) VALUES (?, ?)",
                (_now(), json.dumps(config, ensure_ascii=False)))
            return int(cursor.lastrowid)

    def finish_run(self, run_id: int, *, generated: int, valid: int,
                   inserted: int, filters: dict, elapsed_seconds: float,
                   note: str = "") -> dict:
        rate = inserted / elapsed_seconds if elapsed_seconds > 0 else 0.0
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET finished_at = ?, generated = ?, valid = ?,"
                " inserted = ?, filters = ?, effective_per_second = ?, note = ?"
                " WHERE id = ?",
                (_now(), int(generated), int(valid), int(inserted),
                 json.dumps(filters, ensure_ascii=False), rate, note,
                 int(run_id)))
        return {"inserted": int(inserted), "elapsed_seconds": elapsed_seconds,
                "effective_per_second": rate}

    # -- integrity -------------------------------------------------------

    def validate(self) -> list[str]:
        problems: list[str] = []
        try:
            for row in self._conn.execute("PRAGMA integrity_check"):
                if row[0] != "ok":
                    problems.append(f"integrity_check: {row[0]}")
            columns = {row[1] for row in
                       self._conn.execute("PRAGMA table_info(samples)")}
            missing = REQUIRED_COLUMNS - columns
            if missing:
                problems.append(f"missing columns: {', '.join(sorted(missing))}")
            games = self._conn.execute("SELECT * FROM games").fetchall()
            game_by_id = {g["id"]: g for g in games}
            for game in games:
                if game["status"] == "committed" and not game["split"]:
                    problems.append(f"game {game['id']}: committed without split")
                if game["status"] == "committed" and not game["committed_at"]:
                    problems.append(f"game {game['id']}: committed without time")
            for sample in self._conn.execute(
                    "SELECT s.id, s.game_id, s.board, s.side_to_move, s.d4_key, s.label_source,"
                    " s.value, s.result_value, g.status, g.split "
                    "FROM samples s JOIN games g ON g.id = s.game_id"):
                cells = list(sample["board"])
                if len(cells) != 225:
                    problems.append(f"sample {sample['id']}: bad board length")
                    continue
                expected = _expected_side(cells)
                if expected is None or expected != sample["side_to_move"]:
                    problems.append(
                        f"sample {sample['id']}: side to move mismatch")
                if sample["label_source"] == "teacher" and sample["value"] is None:
                    problems.append(f"sample {sample['id']}: teacher without value")
                if sample["label_source"] == "selfplay_result" \
                        and sample["result_value"] is None:
                    problems.append(f"sample {sample['id']}: result without value")
                if sample["label_source"] == "selfplay_result":
                    game = game_by_id[sample["game_id"]]
                    if game["status"] == "committed":
                        expected = {"black_win": 1.0, "white_win": -1.0, "draw": 0.0}.get(game["final_result"])
                        if expected is None or sample["result_value"] != expected:
                            problems.append(f"sample {sample['id']}: missing or inconsistent completed outcome")
            leaked = self._conn.execute(
                "SELECT COUNT(*) FROM samples s JOIN games g ON g.id = s.game_id "
                "JOIN validation_keys v ON v.d4_key = s.d4_key "
                "WHERE g.status = 'committed' AND g.split = 'train'").fetchone()[0]
            if leaked:
                problems.append(
                    f"{leaked} training sample(s) are D4-equivalent to validation")
            frozen = json.loads(dict(self._conn.execute(
                "SELECT key, value FROM meta"))["validation"]).get(
                    "source_ids", [])
            actual = [row[0] for row in self._conn.execute(
                "SELECT source_id FROM games WHERE status = 'committed' "
                "AND split = 'validation' ORDER BY committed_at, id")]
            if sorted(frozen) != sorted(actual):
                problems.append(
                    "frozen validation membership does not match committed games")
        except sqlite3.DatabaseError as exc:
            problems.append(f"semantic check failed: {exc}")
        return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("stats", "print pool counters and capacity"),
                            ("validate", "run integrity and semantic checks"),
                            ("evict", "reduce the pool to a new capacity")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("pool")
        if name == "evict":
            command.add_argument("--capacity", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        with SamplePool(args.pool, create=False) as pool:
            if args.command == "stats":
                print(json.dumps(pool.stats(), ensure_ascii=False, indent=2))
                return 0
            if args.command == "validate":
                problems = pool.validate()
                if problems:
                    for problem in problems:
                        print(problem, file=sys.stderr)
                    return 1
                print(f"sample pool OK: {args.pool}")
                return 0
            pool.set_capacity(args.capacity)
            with pool._conn:
                evicted = pool._enforce_capacity()
            print(json.dumps(evicted, ensure_ascii=False))
            return 0
    except PoolError as exc:
        print(f"sample_pool: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
