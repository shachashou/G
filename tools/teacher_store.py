#!/usr/bin/env python3
"""Versioned teacher-sample store (ticket 04).

One SQLite file holds every teacher request: accepted samples and the ones
that were filtered out, with the reason. Training reads ``iter_records()`` and
takes the accepted rows; the filtered rows stay for auditing counts.

Format rules:

- ``PRAGMA application_id`` identifies a teacher store, ``user_version`` the
  schema. Any other application id, a newer schema or a damaged file is
  rejected with a message instead of being read partially.
- Only Renju samples are stored by this store (the teacher is configured for
  Renju); a wrong-rule row is refused on write and reported by ``validate()``.
- The board is exactly 225 cells, and the side to move must agree with the
  stone count (black = white or black = white + 1).
- ``final_result`` is NULL when no real result exists (standalone position or
  an unfinished game); it is never filled with a draw.
- ``validate()`` runs SQLite's integrity check and the semantic checks above
  and returns every problem it finds.

Usage::

    from teacher_store import TeacherStore

    with TeacherStore("data/teacher/samples.sqlite3") as store:
        store.add(record)
        print(store.counts())
        for row in store.iter_records(accepted_only=True):
            ...

Command line::

    python tools/teacher_store.py stats data/teacher/samples.sqlite3
    python tools/teacher_store.py validate data/teacher/samples.sqlite3
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

APPLICATION_ID = 0x47544D31  # "GTM1"
SCHEMA_VERSION = 1
STORE_FORMAT = "gomoku-nnue-teacher-store-v1"

UNKNOWN_FILTERS = frozenset({"teacher_timeout", "teacher_error", "process_error"})
RULE = "renju"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS teacher_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    board BLOB NOT NULL CHECK (length(board) = 225),
    rule TEXT NOT NULL CHECK (rule IN ('renju')),
    side_to_move TEXT NOT NULL CHECK (side_to_move IN ('black', 'white')),
    ply INTEGER,
    last_move INTEGER,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('game', 'standalone')),
    source_id TEXT NOT NULL,
    model_id TEXT,
    tactical_status TEXT NOT NULL,
    teacher_status TEXT NOT NULL,
    filter_reason TEXT,
    accepted INTEGER NOT NULL CHECK (accepted IN (0, 1)),
    final_result TEXT CHECK (final_result IS NULL OR
                             final_result IN ('black_win', 'white_win', 'draw')),
    raw_eval TEXT,
    raw_score INTEGER,
    mate_plies INTEGER,
    perspective TEXT NOT NULL,
    normalization TEXT NOT NULL,
    normalization_param REAL NOT NULL,
    value REAL,
    suggested_move TEXT,
    pv TEXT,
    depth INTEGER,
    budget_ms INTEGER NOT NULL,
    threads INTEGER NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    teacher_release TEXT NOT NULL,
    teacher_version TEXT NOT NULL,
    teacher_exe_sha256 TEXT NOT NULL,
    teacher_weights_sha256 TEXT NOT NULL,
    teacher_config_sha256 TEXT NOT NULL,
    label_version TEXT NOT NULL,
    note TEXT,
    CHECK (accepted = 0 OR (teacher_status = 'ok' AND value IS NOT NULL)),
    CHECK (accepted = 1 OR filter_reason IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS teacher_samples_accepted
    ON teacher_samples (accepted);
CREATE INDEX IF NOT EXISTS teacher_samples_source
    ON teacher_samples (source_id);
"""

REQUIRED_COLUMNS = {
    "id", "created_at", "board", "rule", "side_to_move", "ply", "last_move",
    "source_kind", "source_id", "model_id", "tactical_status",
    "teacher_status", "filter_reason", "accepted", "final_result", "raw_eval",
    "raw_score", "mate_plies", "perspective", "normalization",
    "normalization_param", "value", "suggested_move", "pv", "depth",
    "budget_ms", "threads", "elapsed_ms", "teacher_release", "teacher_version",
    "teacher_exe_sha256", "teacher_weights_sha256", "teacher_config_sha256",
    "label_version", "note",
}


class StoreError(RuntimeError):
    """The store is missing, damaged or not compatible with this reader."""


def _expected_side(cells: list[int]) -> str | None:
    black = sum(1 for c in cells if c == 1)
    white = sum(1 for c in cells if c == 2)
    if black == white:
        return "black"
    if black == white + 1:
        return "white"
    return None


class TeacherStore:
    def __init__(self, path: str | Path, *, create: bool = True) -> None:
        self.path = Path(path)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() and not create:
            raise StoreError(f"teacher store does not exist: {self.path}")
        try:
            self._conn = sqlite3.connect(str(self.path), timeout=10)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error as exc:
            raise StoreError(f"cannot open teacher store {self.path}: {exc}") from exc
        try:
            self._prepare(create)
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
            raise StoreError(
                f"not a teacher store or the file is corrupt: {self.path} "
                f"({exc})") from exc

        if not tables:
            if not create:
                raise StoreError(f"empty teacher store: {self.path}")
            self._create_schema()
            return

        if app_id != APPLICATION_ID:
            raise StoreError(
                f"{self.path} is not a teacher store "
                f"(application_id={app_id:#x}, expected {APPLICATION_ID:#x})")
        if version > SCHEMA_VERSION:
            raise StoreError(
                f"teacher store schema {version} is newer than this reader "
                f"({SCHEMA_VERSION}); refusing to guess")
        if version < SCHEMA_VERSION:
            raise StoreError(
                f"teacher store schema {version} is older than this reader "
                f"({SCHEMA_VERSION}); migration is not implemented")
        meta = dict(self._conn.execute("SELECT key, value FROM meta"))
        if meta.get("store_format") != STORE_FORMAT:
            raise StoreError(
                f"unexpected store format {meta.get('store_format')!r}; "
                f"expected {STORE_FORMAT!r}")
        self._check_columns()

    def _create_schema(self) -> None:
        with self._conn:
            self._conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.executescript(SCHEMA)
            self._conn.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                [("store_format", STORE_FORMAT),
                 ("schema_version", str(SCHEMA_VERSION))])

    def _check_columns(self) -> None:
        columns = {row[1] for row in
                   self._conn.execute("PRAGMA table_info(teacher_samples)")}
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise StoreError(
                f"teacher store schema is missing columns: "
                f"{', '.join(sorted(missing))}")

    def close(self) -> None:
        try:
            self._conn.close()
        finally:
            pass

    def __enter__(self) -> "TeacherStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- write -----------------------------------------------------------

    def add(self, record: dict) -> int:
        """Validates and appends one request (accepted or filtered)."""
        self._validate_record(record)
        columns = sorted(record)
        placeholders = ", ".join("?" for _ in columns)
        sql = (f"INSERT INTO teacher_samples ({', '.join(columns)}) "
               f"VALUES ({placeholders})")
        try:
            with self._conn:
                cursor = self._conn.execute(sql, [record[c] for c in columns])
                return int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise StoreError(f"teacher store rejected the record: {exc}") from exc
        except sqlite3.DatabaseError as exc:
            raise StoreError(f"cannot write to teacher store: {exc}") from exc

    def _validate_record(self, record: dict) -> None:
        board = record.get("board")
        if not isinstance(board, (bytes, bytearray)) or len(board) != 225:
            raise StoreError("record.board must be 225 bytes")
        if any(value not in (0, 1, 2) for value in board):
            raise StoreError("record.board values must be 0/1/2")
        cells = list(board)
        if record.get("rule") != RULE:
            raise StoreError(
                f"teacher store accepts Renju samples only, got "
                f"{record.get('rule')!r}")
        side = record.get("side_to_move")
        if side not in ("black", "white"):
            raise StoreError("record.side_to_move must be black or white")
        expected = _expected_side(cells)
        if expected is None:
            raise StoreError(
                "board stone counts are not a legal turn (black must equal "
                "white, or exceed it by one)")
        if side != expected:
            raise StoreError(
                f"side_to_move {side!r} does not match the stone count "
                f"(expected {expected!r})")
        accepted = record.get("accepted")
        if accepted not in (0, 1):
            raise StoreError("record.accepted must be 0 or 1")
        if accepted == 1 and record.get("teacher_status") != "ok":
            raise StoreError("accepted records require teacher_status = ok")
        if accepted == 1 and record.get("value") is None:
            raise StoreError("accepted records require a normalized value")
        if accepted == 0 and not record.get("filter_reason"):
            raise StoreError("filtered records require a filter_reason")
        result = record.get("final_result")
        if result is not None and result not in ("black_win", "white_win",
                                                 "draw"):
            raise StoreError("final_result must be NULL, black_win, white_win or draw")
        if record.get("source_kind") not in ("game", "standalone"):
            raise StoreError("source_kind must be game or standalone")
        if not record.get("source_id"):
            raise StoreError("source_id is required")
        for name in ("created_at", "tactical_status", "teacher_status",
                     "perspective", "normalization", "teacher_release",
                     "teacher_version", "teacher_exe_sha256",
                     "teacher_weights_sha256", "teacher_config_sha256",
                     "label_version"):
            if record.get(name) in (None, ""):
                raise StoreError(f"record.{name} is required")

    # -- read ------------------------------------------------------------

    def counts(self) -> dict:
        rows = self._conn.execute(
            "SELECT accepted, filter_reason, teacher_status, tactical_status, "
            "COUNT(*) AS n FROM teacher_samples "
            "GROUP BY accepted, filter_reason, teacher_status, tactical_status")
        counts = {"total": 0, "accepted": 0, "filtered": 0, "unknown": 0,
                  "vcf_unknown": 0, "by_reason": {}, "by_status": {},
                  "by_tactical": {}}
        for row in rows:
            n = int(row["n"])
            counts["total"] += n
            if row["accepted"]:
                counts["accepted"] += n
            else:
                counts["filtered"] += n
                reason = row["filter_reason"]
                counts["by_reason"][reason] = counts["by_reason"].get(reason, 0) + n
                if reason in UNKNOWN_FILTERS:
                    counts["unknown"] += n
            status = row["teacher_status"]
            counts["by_status"][status] = counts["by_status"].get(status, 0) + n
            tactical = row["tactical_status"]
            counts["by_tactical"][tactical] = (
                counts["by_tactical"].get(tactical, 0) + n)
            if tactical == "vcf_unknown":
                counts["vcf_unknown"] += n
        return counts

    def iter_records(self, *, accepted_only: bool = False):
        sql = "SELECT * FROM teacher_samples"
        if accepted_only:
            sql += " WHERE accepted = 1"
        sql += " ORDER BY id"
        for row in self._conn.execute(sql):
            record = dict(row)
            record["board"] = list(record["board"])
            yield record

    def validate(self) -> list[str]:
        problems: list[str] = []
        try:
            rows = self._conn.execute("PRAGMA integrity_check").fetchall()
        except sqlite3.DatabaseError as exc:
            return [f"integrity_check failed: {exc}"]
        for row in rows:
            if row[0] != "ok":
                problems.append(f"integrity_check: {row[0]}")
        try:
            self._check_columns()
        except StoreError as exc:
            problems.append(str(exc))
        try:
            bad = self._conn.execute(
                "SELECT id FROM teacher_samples WHERE length(board) != 225 "
                "OR accepted NOT IN (0, 1)").fetchall()
            for row in bad:
                problems.append(f"sample {row[0]}: malformed board or accepted flag")
            bad = self._conn.execute(
                "SELECT id FROM teacher_samples WHERE rule != 'renju'").fetchall()
            for row in bad:
                problems.append(f"sample {row[0]}: rule is not renju")
            for record in self.iter_records():
                expected = _expected_side(record["board"])
                if expected is None or expected != record["side_to_move"]:
                    problems.append(
                        f"sample {record['id']}: side to move does not match "
                        f"the stone count")
                if record["accepted"] and record["value"] is None:
                    problems.append(f"sample {record['id']}: accepted without value")
                if record["accepted"] and record["final_result"] == "draw" \
                        and record["source_kind"] == "standalone":
                    problems.append(
                        f"sample {record['id']}: standalone sample claims a draw")
        except sqlite3.DatabaseError as exc:
            problems.append(f"semantic check failed: {exc}")
        return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    stats = sub.add_parser("stats", help="print request/sample counts")
    stats.add_argument("store")
    valid = sub.add_parser("validate", help="run integrity and semantic checks")
    valid.add_argument("store")
    args = parser.parse_args(argv)
    try:
        with TeacherStore(args.store, create=False) as store:
            if args.command == "stats":
                print(json.dumps(store.counts(), ensure_ascii=False, indent=2))
                return 0
            problems = store.validate()
            if problems:
                for problem in problems:
                    print(problem, file=sys.stderr)
                return 1
            print(f"teacher store OK: {store.path} ({store.counts()['total']} rows)")
            return 0
    except StoreError as exc:
        print(f"teacher_store: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
