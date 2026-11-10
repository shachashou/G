#!/usr/bin/env python3
"""Headless tests for the reusable VCF tactical boundary (ticket 03).

Checks that tools/vcf_tactics.py can label positions for data collection
without the GUI, and that it keeps the proven / unknown distinction intact.

Usage: python vcf_tactics_tests.py <gomoku_nnue.exe> [tools dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

EXE = sys.argv[1] if len(sys.argv) > 1 else "build/engine/gomoku_nnue.exe"
TOOLS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tools")
sys.path.insert(0, str(TOOLS.resolve()))

from executor import ExecutorClient  # noqa: E402
import vcf_tactics  # noqa: E402

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


COUNTER_BLACK = [(7, 7), (9, 7), (10, 7), (8, 6), (11, 6), (8, 5)]
COUNTER_WHITE = [(7, 8), (11, 7), (9, 6), (10, 5), (8, 4), (11, 4)]


def main():
    with ExecutorClient(EXE) as client:
        client.rebuild(cells_from(COUNTER_BLACK, COUNTER_WHITE), "black",
                       "renju")
        report = vcf_tactics.probe(client, max_nodes=50000,
                                   max_attack_plies=10)
        check(vcf_tactics.is_proven(report), "probe reports a proven VCF")
        check(vcf_tactics.variation(report)
              == [(8, 7), (6, 7), (8, 9), (8, 8), (9, 8)],
              "variation is exposed as (x, y) points")
        check(vcf_tactics.first_move(report) == (8, 7),
              "first move is exposed")
        text = vcf_tactics.describe(report)
        check("已证明" in text and "首着" in text, "describe is human readable")
        check("预算" in text and "节点" in text, "describe records the budget")

        tight = vcf_tactics.probe(client, max_nodes=1, max_attack_plies=10)
        check(vcf_tactics.is_unknown(tight),
              "budget exhaustion stays unknown for the data pipeline")
        check(not vcf_tactics.is_proven(tight), "unknown is never proven")
        check(vcf_tactics.variation(tight) == [],
              "unknown carries no variation")
        check("未知" in vcf_tactics.describe(tight),
              "describe says unknown, not 'no mate'")

        positions = [
            {"cells": cells_from(COUNTER_BLACK, COUNTER_WHITE),
             "side_to_move": "black"},
            {"cells": [0] * (SIZE * SIZE), "side_to_move": "black"},
        ]
        labels = list(vcf_tactics.filter_positions(positions, client,
                                                   max_nodes=50000,
                                                   max_attack_plies=10))
        check(len(labels) == 2, "filter_positions labels every input")
        check(labels[0]["vcf"]["status"] == "proven",
              "first labelled position is proven")
        check(labels[1]["vcf"]["status"] == "disproven",
              "second labelled position is disproven")
        check(all("describe" in item for item in labels),
              "labels carry a readable description")

    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
