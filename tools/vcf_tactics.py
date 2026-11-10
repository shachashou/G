#!/usr/bin/env python3
"""Reusable VCF (victory by continuous fours) tactical boundary (ticket 03).

The search itself is the vendored upstream Renju solver (quintet, MIT) reached
through the engine's C++ adapter. This module is the headless Python side of
that boundary: data collection, labelling and test tooling use it directly, so
nothing here depends on the GUI, on engine-internal state or on a teacher
label string.

Only a ``proven`` query returns a variation, and that variation has already
been replayed through the authoritative executor before it is reported.
``disproven`` means "no VCF within these bounds"; ``budget_exhausted`` and
``interrupted`` mean unknown and must not be read as "no mate".

Usage::

    from executor import ExecutorClient
    from vcf_tactics import probe, describe, variation

    with ExecutorClient("build/engine/gomoku_nnue.exe") as client:
        client.new_game("renju")
        for x, y in [(7, 7), (7, 6), (8, 7)]:
            client.play(x, y)
        report = probe(client, max_nodes=50000, max_attack_plies=10)
        print(describe(report), variation(report))

Command line::

    .venv\\Scripts\\python.exe tools/vcf_tactics.py --engine build/engine/gomoku_nnue.exe check H8,H7,I8
    .venv\\Scripts\\python.exe tools/vcf_tactics.py --engine build/engine/gomoku_nnue.exe filter positions.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from executor import ExecutorClient, ExecutorError  # noqa: E402

__all__ = [
    "STATUS_TEXT",
    "probe",
    "describe",
    "variation",
    "first_move",
    "is_proven",
    "is_unknown",
    "filter_positions",
]

STATUS_TEXT = {
    "proven": "已证明：存在连续冲四杀",
    "disproven": "未找到：给定预算内没有 VCF（不代表没有其他强制杀法）",
    "budget_exhausted": "未知：节点预算耗尽，搜索未完成",
    "interrupted": "未知：达到时间上限被中断",
    "unsupported_rule": "不支持：开源算杀只实现有禁手（Renju）规则",
    "invalid_input": "输入错误：局面或参数非法，或上游变化未通过权威执行器复查",
}

UNKNOWN_STATUSES = frozenset({"budget_exhausted", "interrupted"})


def _report(response: dict) -> dict:
    report = response.get("vcf")
    if not isinstance(report, dict):
        raise ExecutorError(f"executor vcf response has no vcf object: {response!r}")
    return report


def is_proven(report: dict) -> bool:
    return report.get("status") == "proven"


def is_unknown(report: dict) -> bool:
    return report.get("status") in UNKNOWN_STATUSES


def probe(client: ExecutorClient, *, max_nodes: int | None = None,
          max_attack_plies: int | None = None,
          time_ms: int | None = None) -> dict:
    """Queries the current executor position; returns the ``vcf`` object."""
    return _report(client.vcf(max_nodes=max_nodes,
                              max_attack_plies=max_attack_plies,
                              time_ms=time_ms))


def variation(report: dict) -> list[tuple[int, int]]:
    """The proven line as (x, y), attacker first; empty when not proven."""
    if not is_proven(report):
        return []
    return [(int(p[0]), int(p[1])) for p in report.get("variation", [])]


def first_move(report: dict) -> tuple[int, int] | None:
    point = report.get("first_move")
    if not point:
        return None
    return int(point[0]), int(point[1])


def describe(report: dict) -> str:
    """One human-readable line: status, budget, cost and evidence."""
    status = str(report.get("status", "invalid_input"))
    text = STATUS_TEXT.get(status, status)
    budget = report.get("budget", {}) or {}
    parts = [
        text,
        f"预算 {budget.get('max_nodes', '?')} 节点 / {budget.get('max_attack_plies', '?')} 攻击手"
        f" / 时间上限 {budget.get('time_ms', 0)} ms",
        f"实际 {report.get('nodes', 0)} 节点，{report.get('elapsed_ms', 0)} ms",
    ]
    end = report.get("end") or {}
    if is_proven(report):
        kind = end.get("kind")
        if kind == "fours":
            parts.append("证据：双四/直四，两个杀点")
        elif kind == "forbidden":
            parts.append("证据：唯一点被黑方禁手封死")
        start = first_move(report)
        if start is not None:
            parts.append(f"首着 ({start[0]},{start[1]})")
        line = variation(report)
        if line:
            parts.append("变化 " + " ".join(f"({x},{y})" for x, y in line))
        else:
            parts.append("变化：无（该局面攻击方已有四，上游不展开）")
    message = report.get("message")
    if message:
        parts.append(str(message))
    return "；".join(parts)


def _parse_point(token: str) -> tuple[int, int]:
    token = token.strip()
    if not token:
        raise ExecutorError("empty move token")
    if "," in token:
        x_text, y_text = token.split(",", 1)
        return int(x_text), int(y_text)
    if token[0].isalpha():
        return ord(token[0].upper()) - ord("A"), int(token[1:]) - 1
    raise ExecutorError(f"unrecognised move token {token!r}")


def _apply_moves(client: ExecutorClient, moves: Iterable[str], rule: str) -> None:
    client.new_game(rule)
    for token in moves:
        token = token.strip()
        if not token:
            continue
        if token.lower() in ("pass", "--"):
            raise ExecutorError("pass moves are not supported by the executor")
        x, y = _parse_point(token)
        response = client.play(x, y)
        if not response.get("ok"):
            raise ExecutorError(f"position rejected at {token}: {response}")


def filter_positions(positions: Iterable[dict], client: ExecutorClient, *,
                     max_nodes: int | None = None,
                     max_attack_plies: int | None = None,
                     rule: str = "renju") -> Iterator[dict]:
    """Labels positions for the data pipeline, one report per input.

    Each item is ``{"cells": [...225 ints...], "side_to_move": "black"|"white"}``
    (the executor's rebuild format). The yield adds ``vcf`` and ``describe``;
    the caller decides whether to drop proven/unknown positions. This is the
    reusable path ticket 05 will use to filter forced-tactics positions.
    """
    for index, item in enumerate(positions):
        response = client.rebuild(item["cells"],
                                  item.get("side_to_move", "black"), rule)
        if not response.get("ok"):
            yield {"index": index, "error": response.get("error", "rebuild failed")}
            continue
        report = probe(client, max_nodes=max_nodes,
                       max_attack_plies=max_attack_plies)
        yield {"index": index, "vcf": report, "describe": describe(report)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", required=True, help="gomoku_nnue executable")
    parser.add_argument("--max-nodes", type=int, default=50000)
    parser.add_argument("--max-attack-plies", type=int, default=10)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="query one position given as move list")
    check.add_argument("moves", nargs="*", default=[],
                       help="moves like H8,H7,I8 (alternating from black)")

    filt = sub.add_parser("filter", help="label a JSONL file of positions")
    filt.add_argument("input", help="JSONL with {cells, side_to_move} objects")
    filt.add_argument("--output", default="-", help="output JSONL or - for stdout")

    args = parser.parse_args(argv)
    try:
        with ExecutorClient(args.engine) as client:
            if args.command == "check":
                _apply_moves(client, args.moves, "renju")
                report = probe(client, max_nodes=args.max_nodes,
                               max_attack_plies=args.max_attack_plies)
                print(describe(report))
                return 0
            output = sys.stdout if args.output == "-" else open(
                args.output, "w", encoding="utf-8")
            try:
                with open(args.input, encoding="utf-8") as handle:
                    items = (json.loads(line) for line in handle if line.strip())
                    for labelled in filter_positions(
                            items, client, max_nodes=args.max_nodes,
                            max_attack_plies=args.max_attack_plies):
                        output.write(json.dumps(labelled, ensure_ascii=False) + "\n")
            finally:
                if output is not sys.stdout:
                    output.close()
            return 0
    except ExecutorError as exc:
        print(f"vcf_tactics: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
