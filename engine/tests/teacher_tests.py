#!/usr/bin/env python3
"""Teacher labelling and sample-store tests (ticket 04).

Runs the real local Rapfi teacher and the real executor, writes samples to a
temporary store, and checks the contract end to end:

- install identity (release, hashes, banner) is verified before use;
- a quiet position becomes an accepted sample with raw score, side-to-move
  perspective, normalization and value, and reads back identically;
- mate scores, decided positions and proven VCF positions are filtered and
  counted separately, never accepted;
- teacher timeouts are counted as unknown, not as no-label-success;
- the store rejects wrong rules, damaged files and newer schemas, and
  ``validate()`` catches semantic damage;
- standalone samples and unfinished games keep a NULL result (no forged draw).

Usage: python teacher_tests.py <gomoku_nnue.exe> <tools dir>
Exit code 77 means "teacher not installed" (registered as a ctest skip).
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

EXE = sys.argv[1] if len(sys.argv) > 1 else "build/engine/gomoku_nnue.exe"
TOOLS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tools")
sys.path.insert(0, str(TOOLS.resolve()))

from executor import ExecutorClient  # noqa: E402
from rapfi_teacher import (RapfiTeacher, TeacherReport, check_install,  # noqa: E402
                           load_lock)
from teacher_label import label_position  # noqa: E402
from teacher_store import StoreError, TeacherStore  # noqa: E402

SIZE = 15
failures = 0


def check(cond, msg):
    global failures
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures += 1


def cells_from(blacks, whites):
    cells = [0] * (SIZE * SIZE)
    for x, y in blacks:
        cells[y * SIZE + x] = 1
    for x, y in whites:
        cells[y * SIZE + x] = 2
    return cells


def coords(text):
    out = []
    for token in text.split(","):
        token = token.strip()
        if len(token) >= 2:
            out.append((ord(token[0].upper()) - ord("A"), int(token[1:]) - 1))
    return out


QUIET_MOVES = "H8,H9,I8,I9"
QUIET_BLACK = coords("H8,I8")
QUIET_WHITE = coords("H9,I9")
MATE_MOVES = "H8,H7,I9,G7,J8,J9,G9,F10"


class StubTeacher:
    """Used only to exercise the unknown-count path deterministically."""

    def __init__(self, identity, status):
        self.identity = identity
        self.status = status

    def label(self, cells, *, side, budget_ms=None):
        return TeacherReport(status=self.status, budget_ms=budget_ms or 0,
                             threads=1, identity=self.identity,
                             message="stub teacher")


def main():
    identity = load_lock()
    install = check_install(identity)
    print("[teacher install]")
    if not install.ok:
        print(f"  SKIP teacher not installed: {install.summary()}")
        return 77
    check(True, "teacher install matches the lock (engine, renju weights, config)")

    tmpdir = tempfile.mkdtemp(prefix="teacher-tests-")
    store_path = os.path.join(tmpdir, "samples.sqlite3")
    try:
        return run(identity, store_path, tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run(identity, store_path, tmpdir):
    with ExecutorClient(EXE) as executor, \
            RapfiTeacher(identity=identity) as teacher, \
            TeacherStore(store_path) as store:
        print("[identity / banner]")
        banner = teacher.about()
        check(banner["name"] == "Rapfi", "engine banner names Rapfi")
        check(banner["version"] == identity.engine_version,
              "banner version matches the lock")
        check(identity.release == "250615" and identity.label_version == "teacher-v1",
              "release and label version are pinned")
        check(all(item["sha256"] for item in identity.weights.values()),
              "Renju weight hashes are recorded")

        print("[accepted sample and store round-trip]")
        report = label_position(
            executor=executor, teacher=teacher, store=store,
            cells=cells_from(QUIET_BLACK, QUIET_WHITE), side="black",
            source_kind="standalone", source_id="test-quiet", ply=8)
        check(report["accepted"] is True and report["stored"] is True,
              "quiet position is accepted and stored")
        check(report["teacher_status"] == "ok", "teacher status is ok")
        check(report["tactical_status"] == "vcf_clear",
              "VCF filter ran first and found no proven win")
        check(isinstance(report["raw_score"], int),
              "raw search score is recorded as an integer")
        expected_value = __import__("math").tanh(
            report["raw_score"] / identity.normalization_param)
        check(abs(report["value"] - expected_value) < 1e-12,
              "normalized value equals tanh(score / 600)")
        check(report["label_version"] == "teacher-v1",
              "label version is on the report")
        check(report["suggested_move"] is not None,
              "teacher's suggested move is recorded separately from the score")
        check(-1.0 <= report["value"] <= 1.0, "value is inside [-1, 1]")
        record = next(store.iter_records(accepted_only=True))
        check(record["board"] == cells_from(QUIET_BLACK, QUIET_WHITE),
              "stored board reads back identically")
        check(record["side_to_move"] == "black", "stored side to move matches")
        check(record["raw_score"] == report["raw_score"]
              and abs(record["value"] - report["value"]) < 1e-12,
              "stored score and value read back identically")
        check(record["final_result"] is None,
              "standalone sample has no fabricated result")
        check(record["teacher_exe_sha256"] == identity.engine_sha256
              and record["teacher_weights_sha256"] == identity.weights_sha256,
              "sample carries the teacher identity")
        check(record["source_id"] == "test-quiet"
              and record["source_kind"] == "standalone",
              "sample carries its source identity")
        check(store.counts()["accepted"] == 1, "valid sample count is 1")

        print("[mate score is filtered]")
        # An 8-ply sequence Rapfi mates, while the VCF filter stays clear.
        executor.new_game("renju")
        for token in MATE_MOVES.split(","):
            x, y = coords(token)[0]
            assert executor.play(x, y).get("ok"), token
        state = executor.state()
        mate = label_position(
            executor=executor, teacher=teacher, store=store,
            cells=state["cells"], side=state["side_to_move"],
            source_kind="standalone", source_id="test-mate", ply=8)
        check(mate["accepted"] is False, "mate score is not accepted")
        check(mate["filter_reason"] == "mate", "mate score is filtered as mate")
        check(mate["raw_eval"].startswith("+") and mate["raw_score"] is None,
              "mate is kept as raw evidence, not as a numeric score")
        check(mate["value"] is None, "mate has no normalized value")

        print("[side-to-move perspective]")
        # White to move and winning: a positive mate proves the score is from
        # the side to move, not always Black's. VCF is limited to one plies so
        # the teacher, not the tactical filter, reports the mate.
        white_puzzle_blacks = coords(
            "K13,M12,M11,J10,G9,I9,J9,M9,G8,H8")
        white_puzzle_whites = coords(
            "L14,J12,L12,I11,H10,I10,L10,H9,F8")
        perspective = label_position(
            executor=executor, teacher=teacher, store=store,
            cells=cells_from(white_puzzle_blacks, white_puzzle_whites),
            side="white", source_kind="standalone",
            source_id="test-perspective", vcf_plies=1)
        check(perspective["teacher_status"] == "mate"
              and perspective["raw_eval"].startswith("+"),
              "White-to-move winning position reports a positive score")
        check(perspective["mate_plies"] is not None
              and perspective["mate_plies"] > 0,
              "positive mate has a positive distance")

        print("[proven VCF is filtered before the teacher]")
        counter_blacks = coords("H8,J8,K8,I7,L7,I6")
        counter_whites = coords("H9,L8,J7,K6,I5,L5")
        vcf = label_position(
            executor=executor, teacher=teacher, store=store,
            cells=cells_from(counter_blacks, counter_whites), side="black",
            source_kind="standalone", source_id="test-vcf")
        check(vcf["accepted"] is False and vcf["filter_reason"] == "vcf_proven",
              "proven VCF is excluded from static-value samples")
        check(vcf["teacher_status"] == "skipped",
              "teacher is not called for a proven VCF")
        check(vcf["tactical_status"] == "vcf_proven",
              "tactical status records the VCF hit")

        print("[decided position is filtered]")
        executor.new_game("renju")
        for token in "H8,H9,I8,I9,J8,J9,K8,K9,L8".split(","):
            x, y = coords(token)[0]
            response = executor.play(x, y)
            assert response.get("ok"), token
        final = executor.state()
        check(final["result"] == "black_win", "test position really is decided")
        over = label_position(
            executor=executor, teacher=teacher, store=store,
            cells=final["cells"], side=final["side_to_move"],
            source_kind="game", source_id="test-game",
            final_result="black_win", ply=9)
        check(over["accepted"] is False and over["filter_reason"] == "game_over",
              "decided position is filtered as game_over")
        rows = [record for record in store.iter_records()
                if record["source_id"] == "test-game"]
        check(rows and rows[-1]["final_result"] == "black_win",
              "a real final result is stored with the game source")
        check(rows[-1]["accepted"] == 0, "decided position is not accepted")

        print("[teacher timeout is unknown]")
        unknown_store = TeacherStore(os.path.join(tmpdir, "unknown.sqlite3"))
        try:
            timeout = label_position(
                executor=executor, teacher=StubTeacher(identity, "timeout"),
                store=unknown_store, cells=cells_from(QUIET_BLACK, QUIET_WHITE),
                side="black", source_kind="standalone",
                source_id="test-timeout")
            counts = unknown_store.counts()
            check(timeout["accepted"] is False
                  and timeout["filter_reason"] == "teacher_timeout",
                  "timeout is filtered as teacher_timeout")
            check(counts["unknown"] == 1,
                  "timeout is counted as an unknown case")
            check(counts["accepted"] == 0, "unknown case never enters the queue")
            check(unknown_store.validate() == [],
                  "store with only filtered rows still validates")
        finally:
            unknown_store.close()

        print("[store counts]")
        counts = store.counts()
        check(counts["total"] == 5, "every request is recorded")
        check(counts["accepted"] == 1, "exactly one accepted sample")
        check(counts["by_reason"].get("mate") == 2, "mate count is recorded")
        check(counts["by_reason"].get("vcf_proven") == 1,
              "VCF filter count is recorded")
        check(counts["by_reason"].get("game_over") == 1,
              "game-over count is recorded")
        check(counts["unknown"] == 0, "no unknown in this store")
        problems = store.validate()
        check(problems == [], "store validates cleanly")

    print("[store integrity and compatibility]")
    copy_path = os.path.join(tmpdir, "copy.sqlite3")
    shutil.copyfile(store_path, copy_path)
    with open(copy_path, "rb") as handle:
        data = handle.read()
    with open(copy_path, "wb") as handle:
        handle.write(data[:100])
    try:
        TeacherStore(copy_path, create=False)
        check(False, "truncated store is rejected")
    except StoreError as exc:
        check("corrupt" in str(exc) or "not a teacher store" in str(exc),
              "truncated store is rejected with a clear message")

    newer_path = os.path.join(tmpdir, "newer.sqlite3")
    shutil.copyfile(store_path, newer_path)
    conn = sqlite3.connect(newer_path)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    try:
        TeacherStore(newer_path, create=False)
        check(False, "newer schema is rejected")
    except StoreError as exc:
        check("newer" in str(exc), "newer schema is rejected explicitly")

    wrong_rule_path = os.path.join(tmpdir, "wrong-rule.sqlite3")
    with TeacherStore(wrong_rule_path) as other:
        record = {
            "created_at": "2026-01-01T00:00:00+00:00",
            "board": bytes(cells_from([(0, 0)], [])),
            "rule": "freestyle",
            "side_to_move": "white",
            "ply": None, "last_move": None, "source_kind": "standalone",
            "source_id": "x", "model_id": None,
            "tactical_status": "vcf_unsupported", "teacher_status": "ok",
            "filter_reason": None, "accepted": 0, "final_result": None,
            "raw_eval": None, "raw_score": None, "mate_plies": None,
            "perspective": "side_to_move", "normalization": "tanh",
            "normalization_param": 600.0, "value": None,
            "suggested_move": None, "pv": None, "depth": None,
            "budget_ms": 800, "threads": 1, "elapsed_ms": 0,
            "teacher_release": identity.release,
            "teacher_version": identity.engine_version,
            "teacher_exe_sha256": identity.engine_sha256,
            "teacher_weights_sha256": identity.weights_sha256,
            "teacher_config_sha256": identity.config_sha256,
            "label_version": identity.label_version, "note": None,
        }
        try:
            other.add(record)
            check(False, "wrong rule is rejected")
        except StoreError as exc:
            check("Renju" in str(exc) or "renju" in str(exc),
                  "wrong rule is rejected with a clear message")
        # Semantic damage: side to move that contradicts the stone count.
        other._conn.execute(
            "INSERT INTO teacher_samples (created_at, board, rule, side_to_move,"
            " source_kind, source_id, tactical_status, teacher_status,"
            " filter_reason, accepted, perspective, normalization,"
            " normalization_param, budget_ms, threads, elapsed_ms,"
            " teacher_release, teacher_version, teacher_exe_sha256,"
            " teacher_weights_sha256, teacher_config_sha256, label_version)"
            " VALUES (?, ?, 'renju', 'white', 'standalone', 'bad', 'invalid',"
            " 'skipped', 'invalid_input', 0, 'side_to_move', 'tanh', 600.0,"
            " 800, 1, 0, ?, ?, ?, ?, ?, ?)",
            ("2026-01-01T00:00:00+00:00",
             bytes(cells_from([(0, 0)], [(2, 2)])),
             identity.release, identity.engine_version, identity.engine_sha256,
             identity.weights_sha256, identity.config_sha256,
             identity.label_version))
        other._conn.commit()
        problems = other.validate()
        check(any("side to move" in problem for problem in problems),
              "validate() catches a side-to-move mismatch")

    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
