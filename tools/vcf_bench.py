#!/usr/bin/env python3
"""Measure the VCF call sites and budgets the search can use (ticket 03).

The search can ask the vendored upstream VCF solver:

- ``off``               - never (A/B baseline),
- ``root``              - one query at the root,
- ``root_and_children`` - root query plus an opponent-VCF check on the top root
  moves.

This script answers "where should it be called, and with what budget" with
numbers instead of inheriting the old root-only solver's settings: for each
position it reports whether the proven line was used, the VCF node cost, the
number of child checks/losses and the wall-clock time of the whole think call.

Usage::

    .venv\\Scripts\\python.exe tools/vcf_bench.py --engine build/engine/gomoku_nnue.exe
    .venv\\Scripts\\python.exe tools/vcf_bench.py --engine ... --json report.json --repeats 7
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from executor import ExecutorClient, ExecutorError  # noqa: E402

SIZE = 15

# Deterministic positions. Coordinates are "H8" style: column letter, row 1-15.
POSITIONS = [
    {
        "name": "counter-four (proven VCF, 3 attacks)",
        "blacks": "H8,J8,K8,I7,L7,I6",
        "whites": "H9,L8,J7,K6,I5,L5",
        "side": "black",
    },
    {
        "name": "puzzle 3040 (proven VCF, 12 attacks)",
        "blacks": "L14,E13,H13,H12,I11,G10,H10,K10,J9,H8,I8",
        "whites": "I14,G13,K12,H11,L11,I10,F9,H9,I9,L9,J8",
        "side": "black",
    },
    {
        "name": "quiet midgame (no VCF expected)",
        "moves": "H8,H7,I9,G7,J8,J9,G9,F10",
        "side": "black",
    },
    {
        "name": "defender to move (child checks)",
        "blacks": "H8,J8,K8,I7,L7,I6",
        "whites": "H9,L8,J7,K6,I5,L5",
        "side": "white",
    },
]

POLICIES = ("off", "root", "root_and_children")


def cells_from(blacks: str, whites: str) -> list[int]:
    cells = [0] * (SIZE * SIZE)
    for x, y in parse_points(blacks):
        cells[y * SIZE + x] = 1
    for x, y in parse_points(whites):
        cells[y * SIZE + x] = 2
    return cells


def parse_points(text: str) -> list[tuple[int, int]]:
    out = []
    for token in text.split(","):
        token = token.strip()
        if len(token) >= 2:
            out.append((ord(token[0].upper()) - ord("A"), int(token[1:]) - 1))
    return out


def load_position(client: ExecutorClient, position: dict) -> None:
    if "moves" in position:
        client.new_game("renju")
        for token in position["moves"].split(","):
            x, y = parse_points(token)[0]
            response = client.play(x, y)
            if not response.get("ok"):
                raise ExecutorError(f"move {token} rejected: {response}")
        return
    response = client.rebuild(cells_from(position["blacks"], position["whites"]),
                              position["side"], "renju")
    if not response.get("ok"):
        raise ExecutorError(f"rebuild rejected: {response}")


def run_once(client: ExecutorClient, depth: int, policy: str,
             vcf_nodes: int, vcf_plies: int = 10) -> dict:
    started = time.perf_counter()
    response = client.think(depth=depth, vcf_policy=policy,
                            vcf_nodes=vcf_nodes, vcf_plies=vcf_plies)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not response.get("ok"):
        raise ExecutorError(f"think failed: {response}")
    vcf = response.get("vcf") or {}
    return {
        "move": response.get("move"),
        "from_vcf": bool(response.get("from_vcf")),
        "vcf_status": vcf.get("status"),
        "vcf_nodes": int(vcf.get("nodes", 0)),
        "vcf_elapsed_ms": int(vcf.get("elapsed_ms", 0)),
        "child_checks": int(response.get("vcf_child_checks", 0)),
        "child_losses": int(response.get("vcf_child_losses", 0)),
        "think_ms": round(elapsed_ms, 2),
        "nodes": int(response.get("nodes", 0)),
    }


def summarise(rows: list[dict]) -> dict:
    return {
        "from_vcf": rows[0]["from_vcf"],
        "vcf_status": rows[0]["vcf_status"],
        "child_checks": rows[0]["child_checks"],
        "child_losses": rows[0]["child_losses"],
        "move": rows[0]["move"],
        "median_think_ms": round(statistics.median(r["think_ms"] for r in rows), 2),
        "max_think_ms": round(max(r["think_ms"] for r in rows), 2),
        "median_vcf_nodes": int(statistics.median(r["vcf_nodes"] for r in rows)),
        "median_search_nodes": int(statistics.median(r["nodes"] for r in rows)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--vcf-nodes", type=int, default=50000)
    parser.add_argument("--vcf-plies", type=int, default=10,
                        help="attacker moves the root query may use")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--budgets", default="1000,5000,50000,200000",
                        help="root budgets probed on the proven puzzle")
    args = parser.parse_args(argv)

    report = {"engine": args.engine, "depth": args.depth,
              "vcf_nodes": args.vcf_nodes, "vcf_plies": args.vcf_plies,
              "positions": {}}
    try:
        with ExecutorClient(args.engine) as client:
            for position in POSITIONS:
                load_position(client, position)
                per_policy = {}
                for policy in POLICIES:
                    rows = [run_once(client, args.depth, policy,
                                     args.vcf_nodes, args.vcf_plies)
                            for _ in range(args.repeats)]
                    per_policy[policy] = summarise(rows)
                report["positions"][position["name"]] = per_policy

            # Budget sweep on the proven puzzle: how cheap can a proof be?
            puzzle = POSITIONS[1]
            load_position(client, puzzle)
            sweep = {}
            for budget in [int(b) for b in args.budgets.split(",") if b.strip()]:
                report_vcf = client.vcf(max_nodes=budget,
                                        max_attack_plies=args.vcf_plies)["vcf"]
                sweep[str(budget)] = {
                    "status": report_vcf.get("status"),
                    "nodes": report_vcf.get("nodes"),
                    "elapsed_ms": report_vcf.get("elapsed_ms"),
                }
            report["budget_sweep"] = sweep
    except ExecutorError as exc:
        print(f"vcf_bench: {exc}", file=sys.stderr)
        return 2

    for name, per_policy in report["positions"].items():
        print(f"\n{name}")
        for policy, row in per_policy.items():
            print(f"  {policy:18s} move={row['move']} from_vcf={row['from_vcf']}"
                  f" vcf={row['vcf_status']} vcf_nodes={row['median_vcf_nodes']}"
                  f" child={row['child_checks']}/{row['child_losses']}"
                  f" think={row['median_think_ms']}ms"
                  f" (max {row['max_think_ms']}ms)"
                  f" search_nodes={row['median_search_nodes']}")
    print("\nroot budget sweep (proven puzzle):")
    for budget, row in report["budget_sweep"].items():
        print(f"  {budget:>7s} nodes -> {row['status']}"
              f" ({row['nodes']} used, {row['elapsed_ms']} ms)")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\nreport: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
