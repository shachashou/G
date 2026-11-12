#!/usr/bin/env python3
"""Ticket 07: multi-process self-play datagen orchestrator.

The heavy work happens in the C++ ``datagen`` executable (one process, one
shard, one RNG stream, one Search object). This script launches those workers,
collects their statistics, merges shards, validates the binary format and asks
the C++ worker to replay sampled records and re-check their search scores.

Commands (all run with the project venv)::

  .venv\\Scripts\\python.exe tools\\datagen.py run --workers 16 ^
      --games-per-worker 100 --depth 6 --out-dir data\\raw --tag round1 --merge
  .venv\\Scripts\\python.exe tools\\datagen.py merge --inputs "data/raw/round1/shard_*.binpack" --output data/raw/round1/train.binpack
  .venv\\Scripts\\python.exe tools\\datagen.py validate --input data/raw/round1/train.binpack
  .venv\\Scripts\\python.exe tools\\datagen.py verify --input data/raw/round1/train.binpack --samples 64
  .venv\\Scripts\\python.exe tools\\datagen.py calibrate --depths 6,5,4
  .venv\\Scripts\\python.exe tools\\datagen.py report --manifest data/raw/round1/manifest.json
"""

import argparse
import glob
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer"))
import binpack  # noqa: E402  (trainer/ is on sys.path by design)

from run_support import file_identity  # noqa: E402  (tools/ is on sys.path)

def _default_datagen_exe() -> Path:
    # Ninja (the cross-platform generator) puts binaries directly in
    # build/engine; keep the VS-generator and POSIX locations as fallbacks.
    candidates = [
        ROOT / "build" / "engine" / "datagen.exe",
        ROOT / "build" / "engine" / "Release" / "datagen.exe",
        ROOT / "build" / "engine" / "datagen",
    ]
    return next((c for c in candidates if c.is_file()), candidates[0])


DEFAULT_EXE = _default_datagen_exe()
DEFAULT_TARGET = 100_000
DEFAULT_BUDGET_HOURS = 1.0
STATS_RE = re.compile(r"(\w+)=([^\s]+)")


def common_options(args: argparse.Namespace, depth: int | None = None) -> list[str]:
    options = [
        "--depth", str(depth if depth is not None else args.depth),
        "--tt-mb", str(args.tt_mb),
        "--rule", str(args.rule),
        "--opening-plies", str(args.opening_plies),
        "--opening-region", str(args.opening_region),
        "--opening-temp", str(args.opening_temp),
        "--max-plies", str(args.max_plies),
        "--vcf-nodes", str(args.vcf_nodes),
    ]
    if args.no_vcf:
        options.append("--no-vcf")
    if getattr(args, "nnue", None) is not None:
        options += ["--nnue", str(args.nnue)]
    return options


def parse_stats(log_text: str) -> dict:
    lines = [line for line in log_text.splitlines() if line.startswith("STATS")]
    if not lines:
        return {}
    values = dict(STATS_RE.findall(lines[-1]))
    parsed = {}
    for key, value in values.items():
        try:
            parsed[key] = float(value) if "." in value else int(value)
        except ValueError:
            parsed[key] = value
    return parsed


def run_shards(
    *,
    exe: Path,
    workers: int,
    games_per_worker: int,
    out_dir: Path,
    seed: int,
    options: list[str],
) -> dict:
    """Launch `workers` independent C++ processes and wait for all of them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    started: list[tuple[int, Path, Path, object]] = []
    log_handles = []
    t0 = time.perf_counter()
    try:
        for shard_id in range(workers):
            shard = out_dir / f"shard_{shard_id:04d}.binpack"
            log_path = log_dir / f"worker_{shard_id:04d}.log"
            argv = [
                str(exe), "--out", str(shard),
                "--games", str(games_per_worker),
                "--seed", str(seed),
                "--shard-id", str(shard_id),
                *options,
            ]
            handle = open(log_path, "w", encoding="utf-8")
            log_handles.append(handle)
            proc = subprocess.Popen(
                argv, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT
            )
            started.append((shard_id, shard, log_path, proc, argv))
    finally:
        for handle in log_handles:
            handle.close()

    shards = []
    for shard_id, shard, log_path, proc, argv in started:
        returncode = proc.wait()
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        stats = parse_stats(log_text)
        shards.append({
            "shard_id": shard_id,
            "path": str(shard),
            "log": str(log_path),
            "command": argv,
            "returncode": returncode,
            "stats": stats,
            "records": int(stats.get("records", 0)),
            "games": int(stats.get("games", 0)),
        })
    # Wall-clock from spawning the first worker until the last one exits.
    elapsed = time.perf_counter() - t0
    return {"elapsed_seconds": elapsed, "shards": shards}


def build_manifest(
    *,
    args: argparse.Namespace,
    result: dict,
    depth: int,
    target: int,
    budget_hours: float,
) -> dict:
    shards = result["shards"]
    records = sum(s["records"] for s in shards)
    games = sum(s["games"] for s in shards)
    elapsed = result["elapsed_seconds"]
    rate = records / elapsed if elapsed > 0 else 0.0
    required = target / (budget_hours * 3600.0) if budget_hours > 0 else 0.0
    eta_hours = (target / rate / 3600.0) if rate > 0 else None
    meets = rate >= required
    cpu = os.cpu_count() or 1
    if meets:
        recommendation = (
            f"depth {depth} x {args.workers} workers 达到 {target:,} 局面的 "
            f"{budget_hours:g} 小时预算，可按此配置跑首轮。"
        )
    else:
        recommendation = (
            f"吞吐不足（实测 {rate:.1f} < 需要 {required:.1f} 局面/秒）。按预案："
            f"先试 `calibrate` 比较 depth {depth - 1}，或把 --workers 从 "
            f"{args.workers} 提到 {cpu}（本机 {cpu} 逻辑核），再考虑放宽时长。"
        )
    return {
        "config": {
            "exe": str(args.exe),
            "workers": args.workers,
            "games_per_worker": args.games_per_worker,
            "depth": depth,
            "tt_mb": args.tt_mb,
            "use_vcf": not args.no_vcf,
            "rule": args.rule,
            "opening_plies": args.opening_plies,
            "opening_region": args.opening_region,
            "opening_temp": args.opening_temp,
            "max_plies": args.max_plies,
            "vcf_nodes": args.vcf_nodes,
            "seed": args.seed,
            "out_dir": str(args.out_dir),
            "nnue": (file_identity(args.nnue)
                     if getattr(args, "nnue", None) is not None else None),
        },
        "host": {"cpu_count": cpu, "platform": platform.platform()},
        "totals": {
            "games": games,
            "records": records,
            "elapsed_seconds": round(elapsed, 3),
            "positions_per_second": round(rate, 2),
            "required_positions_per_second": round(required, 2),
            "target_positions": target,
            "budget_hours": budget_hours,
            "eta_hours": round(eta_hours, 3) if eta_hours is not None else None,
            "meets_budget": meets,
            "record_bytes": records * binpack.RECORD_SIZE,
        },
        "shards": shards,
        "recommendation": recommendation,
    }


def format_report(manifest: dict) -> str:
    totals = manifest["totals"]
    config = manifest["config"]
    lines = [
        "吞吐实测报告 / datagen throughput",
        f"  配置        : depth {config['depth']} x {config['workers']} workers, "
        f"VCF={'on' if config['use_vcf'] else 'off'}, rule={config['rule']}",
        f"  主机        : {manifest['host']['cpu_count']} 逻辑核, {manifest['host']['platform']}",
        f"  样本        : {totals['games']} 局 / {totals['records']:,} 局面 / "
        f"{totals['elapsed_seconds']:.3f} 秒",
        f"  吞吐        : {totals['positions_per_second']:.2f} 局面/秒 "
        f"(单 worker 均值 {totals['positions_per_second'] / max(1, config['workers']):.2f})",
        f"  目标        : {totals['target_positions']:,} 局面 "
        f"(约 {totals['target_positions'] * binpack.RECORD_SIZE / (1 << 20):.1f} MiB 记录体); "
        f"本次落盘 {totals['record_bytes'] / (1 << 20):.2f} MiB",
        f"  预算        : {totals['budget_hours']:g} 小时 -> 需要 "
        f"{totals['required_positions_per_second']:.2f} 局面/秒",
        f"  预计耗时    : {totals['eta_hours']:.2f} 小时" if totals["eta_hours"] is not None else "  预计耗时    : 无法估计",
        f"  是否达标    : {'是' if totals['meets_budget'] else '否'}",
        f"  建议        : {manifest['recommendation']}",
    ]
    return "\n".join(lines)


def cmd_run(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir
    if args.tag:
        out_dir = out_dir / args.tag
    result = run_shards(
        exe=args.exe,
        workers=args.workers,
        games_per_worker=args.games_per_worker,
        out_dir=out_dir,
        seed=args.seed,
        options=common_options(args),
    )
    manifest = build_manifest(
        args=args, result=result, depth=args.depth,
        target=args.target, budget_hours=args.budget_hours,
    )
    failed = [s for s in result["shards"] if s["returncode"] != 0]

    if args.merge:
        shard_paths = [Path(s["path"]) for s in result["shards"] if s["returncode"] == 0]
        if shard_paths:
            try:
                merged = binpack.merge_files(shard_paths, args.merge_output or (out_dir / "train.binpack"))
                manifest["merged"] = merged
            except binpack.BinpackError as exc:
                print(f"merge failed: {exc}", file=sys.stderr)
                failed.append({"path": "merge", "returncode": 3})

    report_path: Path = args.report or (out_dir / "manifest.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_report(manifest))
    print(f"manifest: {report_path}")
    if manifest.get("merged"):
        print(f"merged:   {manifest['merged']['output']} "
              f"({manifest['merged']['records']:,} records)")

    if failed:
        for item in failed:
            print(f"worker failed: {item.get('path')} rc={item.get('returncode')}; "
                  f"see {out_dir / 'logs'}", file=sys.stderr)
        return 3
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    if not depths:
        print("calibrate: --depths needs at least one depth", file=sys.stderr)
        return 2
    results = []
    for depth in depths:
        print(f"calibrating depth {depth} ...", flush=True)
        result = run_shards(
            exe=args.exe,
            workers=args.workers,
            games_per_worker=args.games_per_worker,
            out_dir=args.out_dir / f"depth{depth}",
            seed=args.seed,
            options=common_options(args, depth=depth),
        )
        manifest = build_manifest(
            args=args, result=result, depth=depth,
            target=args.target, budget_hours=args.budget_hours,
        )
        results.append(manifest)
        totals = manifest["totals"]
        print(f"  depth {depth}: {totals['positions_per_second']:.2f} pos/s, "
              f"{totals['records']:,} records in {totals['elapsed_seconds']:.2f}s, "
              f"ETA {totals['eta_hours']:.2f}h" if totals["eta_hours"] is not None
              else f"  depth {depth}: no records")

    ok = [m for m in results if m["totals"]["meets_budget"]]
    best = ok[0] if ok else None
    summary = {
        "target_positions": args.target,
        "budget_hours": args.budget_hours,
        "workers": args.workers,
        "results": results,
        "recommended_depth": best["config"]["depth"] if best else None,
        "recommendation": (
            f"保留 depth {best['config']['depth']}（最高且满足预算）" if best else
            "所有候选 depth 都未达标：提高 --workers 或放宽预算/目标"
        ),
    }
    report_path: Path = args.report or (args.out_dir / "calibration.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary["recommendation"])
    print(f"calibration report: {report_path}")
    return 0


def expand_inputs(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    return paths


def cmd_merge(args: argparse.Namespace) -> int:
    try:
        info = binpack.merge_files(expand_inputs(args.inputs), args.output)
    except (binpack.BinpackError, OSError) as exc:
        print(f"merge failed: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        result = binpack.validate_file(args.input, max_records=args.max_records)
    except (binpack.BinpackError, OSError) as exc:
        print(f"validate failed: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["errors"]:
        print(f"validate: {len(result['errors'])} problem(s)", file=sys.stderr)
        return 3
    print(f"validate: OK ({result['scanned']:,} records, {result['games']:,} games)")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    argv = [str(args.exe), "--verify", str(args.input),
            "--verify-samples", str(args.samples)]
    if getattr(args, "nnue", None) is not None:
        argv += ["--nnue", str(args.nnue)]
    proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


def cmd_report(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    print(format_report(manifest))
    return 0


def add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE, help="datagen.exe path")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--games-per-worker", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--nnue", type=Path, default=None,
                        help="GMNNUE01 weights file; the datagen worker loads "
                             "it and self-plays with NNUE evaluation (a bad "
                             "file is a hard error, not a fallback)")
    parser.add_argument("--tt-mb", type=int, default=16)
    parser.add_argument("--no-vcf", action="store_true")
    parser.add_argument("--rule", type=int, choices=(1, 2), default=2)
    parser.add_argument("--opening-plies", type=int, default=3)
    parser.add_argument("--opening-region", type=int, default=7)
    parser.add_argument("--opening-temp", type=float, default=1.0)
    parser.add_argument("--max-plies", type=int, default=225)
    parser.add_argument("--vcf-nodes", type=int, default=5000)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help="throughput-calibration reference size; not a "
                             "training precondition")
    parser.add_argument("--budget-hours", type=float, default=DEFAULT_BUDGET_HOURS,
                        help="throughput-calibration reference budget")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="launch self-play workers and write a throughput manifest")
    add_search_args(run)
    run.add_argument("--tag", default="", help="sub-directory under --out-dir")
    run.add_argument("--merge", action="store_true", help="merge shards into one file")
    run.add_argument("--merge-output", type=Path, default=None)
    run.add_argument("--report", type=Path, default=None, help="manifest JSON path")
    run.set_defaults(func=cmd_run)

    calibrate = sub.add_parser("calibrate", help="benchmark several depths and recommend one")
    add_search_args(calibrate)
    calibrate.add_argument("--depths", default="6,5,4")
    calibrate.add_argument("--report", type=Path, default=None)
    calibrate.set_defaults(func=cmd_calibrate)

    merge = sub.add_parser("merge", help="concatenate shard bodies into one binpack")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(func=cmd_merge)

    validate = sub.add_parser("validate", help="check magic, layout, WDL backfill and game boundaries")
    validate.add_argument("--input", required=True)
    validate.add_argument("--max-records", type=int, default=None)
    validate.set_defaults(func=cmd_validate)

    verify = sub.add_parser("verify", help="replay sampled records and compare search scores")
    verify.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    verify.add_argument("--input", required=True)
    verify.add_argument("--samples", type=int, default=32)
    verify.add_argument("--nnue", type=Path, default=None,
                        help="GMNNUE01 weights used to generate the records; "
                             "required to reproduce NNUE self-play scores")
    verify.set_defaults(func=cmd_verify)

    report = sub.add_parser("report", help="print an existing throughput manifest")
    report.add_argument("--manifest", required=True)
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
