#!/usr/bin/env python3
"""Teacher labelling pipeline: executor -> VCF filter -> Rapfi -> store (ticket 04).

One call answers: "may this position become a teacher-labelled training
sample, and if not, why?" The order matters:

1. The authoritative executor rebuilds and validates the position.
2. Positions that are decided are not labelled (``game_over``).
3. The shared VCF adapter is asked first; a proven VCF is not a static-value
   sample (``vcf_proven``). A budget/interrupt keeps the position but records
   ``vcf_unknown`` - "not found" is never read as "no other forced win".
4. Only then does Rapfi run. Its mate scores, "no search score" replies,
   timeouts and engine errors are filtered and counted separately.
5. Accepted samples carry the raw score, its perspective, the normalization
   and parameter, the normalized value, the suggested move, the teacher
   identity/budget and a nullable final result.

Standalone positions and unfinished games keep ``final_result = NULL``; the
pipeline never invents a draw.

Usage::

    from executor import ExecutorClient
    from rapfi_teacher import RapfiTeacher
    from teacher_store import TeacherStore
    from teacher_label import label_position

    with ExecutorClient("build/engine/gomoku_nnue.exe") as executor, \\
            RapfiTeacher() as teacher, \\
            TeacherStore("data/teacher/samples.sqlite3") as store:
        report = label_position(executor=executor, teacher=teacher, store=store,
                                cells=cells, side="black",
                                source_kind="standalone", source_id="demo-1")

Command line::

    python tools/teacher_label.py --engine build/engine/gomoku_nnue.exe \\
        --store data/teacher/samples.sqlite3 label --moves H8,H7,I9,G7
    python tools/teacher_label.py --store data/teacher/samples.sqlite3 stats
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from executor import ExecutorClient, ExecutorError  # noqa: E402
import vcf_tactics  # noqa: E402
from rapfi_teacher import (RapfiTeacher, TeacherError, load_lock,  # noqa: E402
                           check_install, parse_moves)
from teacher_store import StoreError, TeacherStore  # noqa: E402

SIZE = 15

TEACHER_FILTERS = {
    "mate": "mate",
    "no_score": "no_score",
    "timeout": "teacher_timeout",
    "error": "teacher_error",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _expected_side(cells: list[int]) -> str | None:
    black = sum(1 for c in cells if c == 1)
    white = sum(1 for c in cells if c == 2)
    if black == white:
        return "black"
    if black == white + 1:
        return "white"
    return None


def _report(*, accepted: bool, stored: bool, record_id: int | None,
            filter_reason: str | None, tactical_status: str,
            teacher_status: str, message: str = "", **extra) -> dict:
    report = {
        "accepted": accepted,
        "stored": stored,
        "record_id": record_id,
        "filter_reason": filter_reason,
        "tactical_status": tactical_status,
        "teacher_status": teacher_status,
        "message": message,
    }
    report.update(extra)
    return report


def label_position(*, executor: ExecutorClient, teacher: RapfiTeacher,
                   store: TeacherStore | None, cells: list[int], side: str,
                   rule: str = "renju", source_kind: str,
                   source_id: str, ply: int | None = None,
                   last_move: int | None = None, model_id: str | None = None,
                   final_result: str | None = None,
                   budget_ms: int | None = None, vcf_nodes: int = 50000,
                   vcf_plies: int = 10, return_record: bool = False) -> dict:
    """Labels one position; returns a report and appends to the store."""
    if len(cells) != SIZE * SIZE or any(c not in (0, 1, 2) for c in cells):
        raise ValueError("cells must be 225 values of 0/1/2")
    if side not in ("black", "white"):
        raise ValueError("side must be black or white")
    if source_kind not in ("game", "standalone"):
        raise ValueError("source_kind must be game or standalone")
    if not source_id:
        raise ValueError("source_id is required")
    identity = teacher.identity

    # 1. The authoritative executor validates the input position.
    rebuild = executor.rebuild(cells, side, rule)
    if not rebuild.get("ok"):
        return _report(accepted=False, stored=False, record_id=None,
                       filter_reason="invalid_input", tactical_status="invalid",
                       teacher_status="skipped",
                       message=str(rebuild.get("error", "rebuild failed")))
    if not rebuild.get("side_to_move_consistent", True):
        return _report(accepted=False, stored=False, record_id=None,
                       filter_reason="inconsistent_turn",
                       tactical_status="invalid", teacher_status="skipped",
                       message="side to move does not match the stone count")

    # This store keeps Renju samples only; the teacher is configured for Renju.
    if rule != "renju":
        return _report(accepted=False, stored=False, record_id=None,
                       filter_reason="unsupported_rule",
                       tactical_status="unsupported",
                       teacher_status="skipped",
                       message="teacher store accepts Renju samples only")

    # 2. Decided positions are not value samples.
    state = executor.state()
    if state.get("result") != "ongoing":
        return _finish(
            executor=executor, teacher=teacher, store=store, cells=cells,
            side=side, rule=rule, source_kind=source_kind, source_id=source_id,
            ply=ply, last_move=last_move, model_id=model_id,
            final_result=final_result, budget_ms=budget_ms,
            tactical_status="game_over", teacher_status="skipped",
            filter_reason="game_over", accepted=False,
            message=f"position already decided: {state.get('result')}")

    # 3. Shared VCF filter.
    vcf = vcf_tactics.probe(executor, max_nodes=vcf_nodes,
                            max_attack_plies=vcf_plies)
    vcf_status = str(vcf.get("status"))
    if vcf_tactics.is_proven(vcf):
        tactical = "vcf_proven"
        return _finish(
            executor=executor, teacher=teacher, store=store, cells=cells,
            side=side, rule=rule, source_kind=source_kind, source_id=source_id,
            ply=ply, last_move=last_move, model_id=model_id,
            final_result=final_result, budget_ms=budget_ms,
            tactical_status=tactical, teacher_status="skipped",
            filter_reason="vcf_proven", accepted=False,
            include_record=return_record,
            message="a forced VCF exists; not a static-value sample")
    if vcf_status == "disproven":
        tactical = "vcf_clear"
    elif vcf_status in ("budget_exhausted", "interrupted"):
        tactical = "vcf_unknown"
    else:
        tactical = "vcf_unsupported"

    # 4. The teacher's search score.
    try:
        report = teacher.label(cells, side=side, budget_ms=budget_ms)
    except TeacherError as exc:
        return _finish(
            executor=executor, teacher=teacher, store=store, cells=cells,
            side=side, rule=rule, source_kind=source_kind, source_id=source_id,
            ply=ply, last_move=last_move, model_id=model_id,
            final_result=final_result, budget_ms=budget_ms,
            tactical_status=tactical, teacher_status="error",
            filter_reason="teacher_error", accepted=False,
            include_record=return_record, message=str(exc))

    if report.status != "ok":
        return _finish(
            executor=executor, teacher=teacher, store=store, cells=cells,
            side=side, rule=rule, source_kind=source_kind, source_id=source_id,
            ply=ply, last_move=last_move, model_id=model_id,
            final_result=final_result, budget_ms=budget_ms,
            tactical_status=tactical, teacher_status=report.status,
            filter_reason=TEACHER_FILTERS.get(report.status, "teacher_error"),
            accepted=False, teacher_report=report,
            include_record=return_record,
            message=report.message or f"teacher status {report.status}")

    value = report.normalized()
    if value is None:
        return _finish(
            executor=executor, teacher=teacher, store=store, cells=cells,
            side=side, rule=rule, source_kind=source_kind, source_id=source_id,
            ply=ply, last_move=last_move, model_id=model_id,
            final_result=final_result, budget_ms=budget_ms,
            tactical_status=tactical, teacher_status="ok",
            filter_reason="no_value", accepted=False, teacher_report=report,
            include_record=return_record,
            message="teacher returned no normalized value")
    return _finish(
        executor=executor, teacher=teacher, store=store, cells=cells,
        side=side, rule=rule, source_kind=source_kind, source_id=source_id,
        ply=ply, last_move=last_move, model_id=model_id,
        final_result=final_result, budget_ms=budget_ms,
        tactical_status=tactical, teacher_status="ok", filter_reason=None,
        accepted=True, teacher_report=report, value=value,
        include_record=return_record)


def _finish(*, executor: ExecutorClient, teacher: RapfiTeacher,
            store: TeacherStore | None, cells: list[int], side: str,
            rule: str, source_kind: str, source_id: str, ply: int | None,
            last_move: int | None, model_id: str | None,
            final_result: str | None, budget_ms: int | None,
            tactical_status: str, teacher_status: str,
            filter_reason: str | None, accepted: bool,
            teacher_report=None, value: float | None = None,
            message: str = "", include_record: bool = False) -> dict:
    identity = teacher.identity
    record = {
        "created_at": _now(),
        "board": bytes(cells),
        "rule": rule,
        "side_to_move": side,
        "ply": ply,
        "last_move": last_move,
        "source_kind": source_kind,
        "source_id": source_id,
        "model_id": model_id,
        "tactical_status": tactical_status,
        "teacher_status": teacher_status,
        "filter_reason": filter_reason,
        "accepted": 1 if accepted else 0,
        "final_result": final_result,
        "raw_eval": teacher_report.raw_eval if teacher_report else None,
        "raw_score": teacher_report.raw_score if teacher_report else None,
        "mate_plies": teacher_report.mate_plies if teacher_report else None,
        "perspective": "side_to_move",
        "normalization": identity.normalization,
        "normalization_param": identity.normalization_param,
        "value": value,
        "suggested_move": (f"{teacher_report.suggested_move[0]},"
                           f"{teacher_report.suggested_move[1]}"
                           if teacher_report and teacher_report.suggested_move
                           else None),
        "pv": teacher_report.pv if teacher_report else None,
        "depth": teacher_report.depth if teacher_report else None,
        "budget_ms": (teacher_report.budget_ms if teacher_report
                      else int(budget_ms or identity.default_budget_ms)),
        "threads": (teacher_report.threads if teacher_report
                    else identity.default_threads),
        "elapsed_ms": teacher_report.elapsed_ms if teacher_report else 0,
        "teacher_release": identity.release,
        "teacher_version": identity.engine_version,
        "teacher_exe_sha256": identity.engine_sha256,
        "teacher_weights_sha256": identity.weights_sha256,
        "teacher_config_sha256": identity.config_sha256,
        "label_version": identity.label_version,
        "note": message or None,
    }
    record_id = store.add(record) if store is not None else None
    counts = store.counts() if store is not None else None
    report = _report(
        accepted=accepted, stored=store is not None, record_id=record_id,
        filter_reason=filter_reason, tactical_status=tactical_status,
        teacher_status=teacher_status, message=message,
        value=value,
        raw_eval=record["raw_eval"], raw_score=record["raw_score"],
        mate_plies=record["mate_plies"],
        suggested_move=record["suggested_move"], depth=record["depth"],
        budget_ms=record["budget_ms"], elapsed_ms=record["elapsed_ms"],
        label_version=record["label_version"],
        teacher_identity=identity.describe(),
        counts=counts)
    if include_record:
        report["record"] = record
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", default="build/engine/gomoku_nnue.exe")
    parser.add_argument("--store", default="data/teacher/samples.sqlite3")
    parser.add_argument("--budget-ms", type=int, default=None)
    parser.add_argument("--vcf-nodes", type=int, default=50000)
    parser.add_argument("--vcf-plies", type=int, default=10)
    sub = parser.add_subparsers(dest="command", required=True)
    label = sub.add_parser("label", help="label one position given as moves")
    label.add_argument("moves", help="H8,H7,I9,... (black first)")
    label.add_argument("--source-id", default="cli-standalone")
    label.add_argument("--source-kind", default="standalone",
                       choices=("game", "standalone"))
    label.add_argument("--model-id", default=None)
    label.add_argument("--final-result", default=None,
                       choices=("black_win", "white_win", "draw"))
    stats = sub.add_parser("stats", help="print request/sample counts")
    validate = sub.add_parser("validate", help="validate the store")
    args = parser.parse_args(argv)

    if args.command in ("stats", "validate"):
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
                print(f"teacher store OK: {store.path}")
                return 0
        except StoreError as exc:
            print(f"teacher_label: {exc}", file=sys.stderr)
            return 2

    cells, side = parse_moves(args.moves)
    identity = load_lock()
    check = check_install(identity)
    if not check.ok:
        print(f"teacher_label: {check.summary()}", file=sys.stderr)
        return 2
    try:
        with ExecutorClient(args.engine) as executor, \
                RapfiTeacher(identity=identity) as teacher, \
                TeacherStore(args.store) as store:
            report = label_position(
                executor=executor, teacher=teacher, store=store, cells=cells,
                side=side, source_kind=args.source_kind,
                source_id=args.source_id, ply=len(
                    [t for t in args.moves.replace(" ", ",").split(",") if t]),
                model_id=args.model_id, final_result=args.final_result,
                budget_ms=args.budget_ms, vcf_nodes=args.vcf_nodes,
                vcf_plies=args.vcf_plies)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report.get("stored") else 2
    except (TeacherError, ExecutorError, StoreError) as exc:
        print(f"teacher_label: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
