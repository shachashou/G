#!/usr/bin/env python3
"""Ticket 11: bootstrap iteration loop (datagen -> train -> export -> gate).

One command runs one round, or a whole loop until convergence, of the M6
strength flywheel::

  datagen (NNUE incumbent) -> train (old + new data) -> export -> gate (new vs
  incumbent) -> promote or keep incumbent -> freeze final model.

The strength gate reuses ``tools/match_runner.py``: engine A is the freshly
exported candidate, engine B is the incumbent. A round whose candidate wins
the gate becomes the new incumbent; otherwise the incumbent is kept and the
candidate is archived but discarded. Convergence is "at least ``--min-rounds``
rounds and ``--no-promote-streak`` consecutive rounds without promotion", or
``--max-rounds`` reached. On convergence the incumbent is frozen as
``weights/final.{pt,nnue}`` with a ``final_manifest.json``.

The pure decision helpers (``promote_from_gate``, ``should_stop``,
``build_final_manifest``) are importable and unit-tested; the loop body shells
out to the existing tools so the heavy steps stay inside the project budget
runner's process tree.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from run_support import ROOT, file_identity, write_json

STATE_SCHEMA = 1
FINAL_MANIFEST_SCHEMA = 1

# match_runner exit codes (see tools/match_runner.py).
GATE_PASS = 0
GATE_FAIL = 2
GATE_INVALID = 3


# --------------------------------------------------------------------------
# Pure decision helpers (unit-tested; no subprocess or filesystem I/O).
# --------------------------------------------------------------------------

def promote_from_gate(report: dict) -> bool:
    """True when a complete, valid match passed its strength gate.

    A candidate promotes only if the match itself is complete (no anomalies /
    invalid backend) AND ``gate.passed`` is true. A gate failure (exit 2) or an
    invalid match (exit 3) both keep the incumbent.
    """
    if not isinstance(report, dict):
        return False
    gate = report.get("gate")
    if not isinstance(gate, dict):
        return False
    return bool(report.get("complete")) and bool(gate.get("passed"))


def should_stop(rounds: list[dict], *, min_rounds: int = 3,
                max_rounds: int | None = None,
                no_promote_streak: int = 2) -> bool:
    """Convergence: >= max_rounds, or >= min_rounds with the last
    ``no_promote_streak`` rounds all failing to promote."""
    n = len(rounds)
    if max_rounds is not None and n >= max_rounds:
        return True
    if n < min_rounds:
        return False
    tail = rounds[-no_promote_streak:]
    return len(tail) == no_promote_streak and all(
        not r.get("promoted", False) for r in tail
    )


def build_final_manifest(*, incumbent: dict, rounds: list[dict],
                         final_pt: Path, final_nnue: Path,
                         final_history: Path, final_export_report: Path,
                         stop_reason: str) -> dict:
    """The provenance record that turns an incumbent into the final model."""
    manifest = {
        "schema": FINAL_MANIFEST_SCHEMA,
        "final": {
            "checkpoint": str(final_pt.resolve()),
            "weights": str(final_nnue.resolve()),
            "history": str(final_history.resolve()),
            "export_report": str(final_export_report.resolve()),
            "source_round": incumbent.get("source_round"),
            "identity": {
                "checkpoint": file_identity(final_pt),
                "weights": file_identity(final_nnue),
            },
        },
        "stop_reason": stop_reason,
        "rounds": rounds,
    }
    return manifest


# --------------------------------------------------------------------------
# Command construction (thin wrappers so the loop body stays readable).
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapConfig:
    work_dir: Path
    final_dir: Path
    incumbent_nnue: Path
    incumbent_pt: Path | None
    frozen_validation: Path | None
    old_data: Path
    engine: Path
    datagen_exe: Path
    max_rounds: int
    min_rounds: int
    min_win_rate: float
    strict_gate: bool
    games: int
    seed: int
    openings: Path | None
    # datagen
    datagen_workers: int
    datagen_games_per_worker: int
    datagen_depth: int
    datagen_seed: int
    # training / export
    train_extra: list[str]
    export_extra: list[str]


def run_cmd(argv: list[str], *, label: str) -> subprocess.CompletedProcess:
    """Run one child tool, streaming its output through to the terminal."""
    print(f"\n=== bootstrap: {label} ===", flush=True)
    print("  $ " + " ".join(shlex.quote(str(a)) for a in argv), flush=True)
    return subprocess.run([str(a) for a in argv], cwd=str(ROOT))


def datagen_command(cfg: BootstrapConfig, round_index: int,
                    incumbent_nnue: Path, out_dir: Path) -> list[str]:
    return [
        sys.executable, str(ROOT / "tools" / "datagen.py"), "run",
        "--exe", str(cfg.datagen_exe),
        "--nnue", str(incumbent_nnue),
        "--workers", str(cfg.datagen_workers),
        "--games-per-worker", str(cfg.datagen_games_per_worker),
        "--depth", str(cfg.datagen_depth),
        "--seed", str(cfg.datagen_seed + round_index),
        "--out-dir", str(out_dir.parent),
        "--tag", out_dir.name,
        "--merge",
    ]


def check_command(cfg: BootstrapConfig, train_data: list[Path],
                  manifest: Path) -> list[str]:
    argv = [
        sys.executable, str(ROOT / "trainer" / "validation_freeze.py"), "check",
        "--frozen", str(cfg.frozen_validation),
        "--data", str(cfg.old_data),
    ]
    for path in train_data:
        argv += ["--train-data", str(path)]
    argv += ["--manifest", str(manifest)]
    return argv


def train_command(cfg: BootstrapConfig, manifest: Path | None,
                  checkpoint: Path, history: Path) -> list[str]:
    argv = [
        sys.executable, str(ROOT / "trainer" / "train.py"),
        "--data", str(cfg.old_data),
        "--output", str(checkpoint),
        "--history", str(history),
    ]
    if cfg.frozen_validation is not None:
        argv += ["--frozen-validation", str(cfg.frozen_validation)]
    if manifest is not None:
        argv += ["--train-manifest", str(manifest)]
    argv += cfg.train_extra
    return argv


def export_command(cfg: BootstrapConfig, checkpoint: Path, weights: Path,
                   report: Path) -> list[str]:
    argv = [
        sys.executable, str(ROOT / "trainer" / "export.py"),
        "--checkpoint", str(checkpoint),
        "--data", str(cfg.old_data),
        "--output", str(weights),
        "--report", str(report),
    ]
    if cfg.frozen_validation is not None:
        argv += ["--frozen-validation", str(cfg.frozen_validation)]
    argv += cfg.export_extra
    return argv


def engine_command(exe: Path, weights: Path) -> str:
    """A match_runner engine string: fixed depth 2 + NNUE weights."""
    return f'"{exe.resolve()}" --depth 2 --nnue "{weights.resolve()}"'


def gate_command(cfg: BootstrapConfig, candidate: Path, incumbent: Path,
                 output: Path) -> list[str]:
    argv = [
        sys.executable, str(ROOT / "tools" / "match_runner.py"),
        "--games", str(cfg.games),
        "--paired-openings",
        "--seed", str(cfg.seed),
        "--min-win-rate", str(cfg.min_win_rate),
        "--verify-backend",
        "--engine-a", engine_command(cfg.engine, candidate),
        "--engine-b", engine_command(cfg.engine, incumbent),
        "--output", str(output),
    ]
    if cfg.strict_gate:
        argv.append("--strict-gate")
    if cfg.openings is not None:
        argv += ["--openings", str(cfg.openings)]
    return argv


# --------------------------------------------------------------------------
# State / archiving.
# --------------------------------------------------------------------------

def load_state(work_dir: Path) -> dict | None:
    state_path = work_dir / "state.json"
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema") != STATE_SCHEMA:
        raise SystemExit(f"bootstrap state schema {state.get('schema')!r} "
                         f"!= {STATE_SCHEMA}; remove {state_path} or migrate")
    return state


def initial_state(cfg: BootstrapConfig) -> dict:
    return {
        "schema": STATE_SCHEMA,
        "incumbent": {
            "nnue": str(cfg.incumbent_nnue.resolve()),
            "pt": str(cfg.incumbent_pt.resolve()) if cfg.incumbent_pt else None,
            "source_round": 0,
        },
        "data_files": [str(cfg.old_data.resolve())],
        "frozen_validation": (str(cfg.frozen_validation.resolve())
                              if cfg.frozen_validation else None),
        "rounds": [],
    }


def incumbent_nnue_path(state: dict) -> Path:
    return Path(state["incumbent"]["nnue"])


# --------------------------------------------------------------------------
# Loop body.
# --------------------------------------------------------------------------

def _record_failure(cfg: BootstrapConfig, state: dict, round_index: int,
                    error: str, exit_code: int, incumbent: Path) -> dict:
    """Record a failed round so it still counts toward the stop condition and
    is visible for review. The incumbent and data file list are left unchanged,
    so the round can be re-run with ``--round`` once the cause is fixed."""
    round_result = {
        "round": round_index,
        "incumbent": {
            "nnue": str(incumbent.resolve()),
            "source_round": state["incumbent"].get("source_round"),
        },
        "error": error,
        "exit_code": exit_code,
        "promoted": False,
    }
    state["rounds"].append(round_result)
    write_json(cfg.work_dir / "state.json", state)
    print(f"round {round_index}: FAILED ({error})", flush=True)
    return round_result


def run_round(cfg: BootstrapConfig, state: dict, round_index: int) -> dict:
    """Execute one iteration and update the state in place; return the round."""
    round_dir = cfg.work_dir / f"round{round_index:03d}"
    datagen_dir = round_dir / "datagen"
    datagen_dir.mkdir(parents=True, exist_ok=True)
    incumbent = incumbent_nnue_path(state)

    # 1. datagen, driven by the incumbent NNUE weights.
    result = run_cmd(datagen_command(cfg, round_index, incumbent, datagen_dir),
                     label=f"round {round_index} datagen")
    new_data = datagen_dir / "train.binpack"
    if result.returncode != 0 or not new_data.exists():
        return _record_failure(cfg, state, round_index, "datagen failed",
                               result.returncode, incumbent)

    # 2. Train-data mixing: rebuild a leak-free manifest over old + all new
    #    files so no validation position (or D4-equivalent) leaks into the
    #    training set, and old data is retained to curb catastrophic forgetting.
    manifest = None
    if cfg.frozen_validation is not None:
        manifest = round_dir / "train_manifest.json"
        # Old data + every prior round + this round's fresh data, all scanned
        # for D4 leaks against the frozen validation split. The frozen source
        # file (old data) must be included so its train games stay in the mix.
        train_data = [Path(p) for p in state["data_files"]] + [new_data]
        result = run_cmd(check_command(cfg, train_data, manifest),
                         label=f"round {round_index} leakage check")
        if result.returncode != 0:
            return _record_failure(cfg, state, round_index,
                                   "leakage check failed", result.returncode,
                                   incumbent)

    # 3. Train.
    checkpoint = round_dir / "candidate.pt"
    history = round_dir / "history.json"
    result = run_cmd(train_command(cfg, manifest, checkpoint, history),
                     label=f"round {round_index} train")
    if result.returncode != 0:
        return _record_failure(cfg, state, round_index, "train failed",
                               result.returncode, incumbent)

    # 4. Export.
    candidate = round_dir / "candidate.nnue"
    export_report = round_dir / "export_report.json"
    result = run_cmd(export_command(cfg, checkpoint, candidate, export_report),
                     label=f"round {round_index} export")
    if result.returncode != 0:
        return _record_failure(cfg, state, round_index, "export failed",
                               result.returncode, incumbent)

    # 5. Gate: candidate (A) vs incumbent (B).
    gate_output = round_dir / "gate.json"
    result = run_cmd(gate_command(cfg, candidate, incumbent, gate_output),
                     label=f"round {round_index} gate")
    if result.returncode not in (GATE_PASS, GATE_FAIL):
        return _record_failure(cfg, state, round_index, "gate invalid",
                               result.returncode, incumbent)

    report = (json.loads(gate_output.read_text(encoding="utf-8"))
              if gate_output.exists() else {})
    promoted = promote_from_gate(report)
    summary = report.get("summary", {})
    round_result = {
        "round": round_index,
        "incumbent": {
            "nnue": str(incumbent.resolve()),
            "source_round": state["incumbent"].get("source_round"),
        },
        "data": str(new_data.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "weights": str(candidate.resolve()),
        "gate": {
            "output": str(gate_output.resolve()),
            "games": summary.get("games"),
            "win_rate": summary.get("win_rate"),
            "a_wins": summary.get("a_wins"),
            "draws": summary.get("draws"),
            "a_losses": summary.get("a_losses"),
            "passed": bool(report.get("gate", {}).get("passed")),
            "complete": bool(report.get("complete")),
        },
        "promoted": promoted,
    }
    if promoted:
        state["incumbent"] = {
            "nnue": str(candidate.resolve()),
            "pt": str(checkpoint.resolve()),
            "source_round": round_index,
        }
    state["data_files"].append(str(new_data.resolve()))
    state["rounds"].append(round_result)
    write_json(cfg.work_dir / "state.json", state)
    print(f"round {round_index}: "
          f"{'PROMOTED' if promoted else 'kept incumbent'} "
          f"(win_rate {summary.get('win_rate')})", flush=True)
    return round_result


def freeze_final(cfg: BootstrapConfig, state: dict, stop_reason: str) -> dict:
    """Copy the incumbent's checkpoint/weights to the final dir and write the
    manifest. Returns the manifest."""
    cfg.final_dir.mkdir(parents=True, exist_ok=True)
    incumbent = state["incumbent"]
    final_pt = cfg.final_dir / "final.pt"
    final_nnue = cfg.final_dir / "final.nnue"
    final_history = cfg.final_dir / "final_history.json"
    final_export = cfg.final_dir / "final_export_report.json"

    if not incumbent.get("pt"):
        raise SystemExit("incumbent has no checkpoint (.pt); rerun with "
                         "--incumbent-pt so final.pt can be frozen")
    source_pt = Path(incumbent["pt"])
    source_nnue = Path(incumbent["nnue"])
    if not source_pt.exists():
        raise SystemExit(f"incumbent checkpoint missing: {source_pt}")
    if not source_nnue.exists():
        raise SystemExit(f"incumbent weights missing: {source_nnue}")

    final_pt.write_bytes(source_pt.read_bytes())
    final_nnue.write_bytes(source_nnue.read_bytes())

    # Best-effort: carry the incumbent round's history/export report over when
    # the incumbent was produced by this loop (source_round > 0).
    round_dir = cfg.work_dir / f"round{incumbent['source_round']:03d}"
    for src, dst in ((round_dir / "history.json", final_history),
                     (round_dir / "export_report.json", final_export)):
        if src.exists():
            dst.write_bytes(src.read_bytes())

    manifest = build_final_manifest(
        incumbent=incumbent, rounds=state["rounds"],
        final_pt=final_pt, final_nnue=final_nnue,
        final_history=final_history, final_export_report=final_export,
        stop_reason=stop_reason,
    )
    write_json(cfg.final_dir / "final_manifest.json", manifest)
    return manifest


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ticket 11 bootstrap loop (datagen -> train -> export -> gate).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--work-dir", type=Path,
                        default=ROOT / "weights" / "bootstrap",
                        help="state + per-round artifacts (default weights/bootstrap)")
    parser.add_argument("--final-dir", type=Path, default=ROOT / "weights",
                        help="where final.{pt,nnue} and final_manifest.json land "
                             "(default weights/)")
    parser.add_argument("--incumbent-nnue", type=Path, required=True,
                        help="initial incumbent GMNNUE01 weights (M5-accepted)")
    parser.add_argument("--incumbent-pt", type=Path, default=None,
                        help="checkpoint behind the initial incumbent (provenance)")
    parser.add_argument("--frozen-validation", type=Path, default=None,
                        help="frozen validation JSON; enables mixed old+new data")
    parser.add_argument("--old-data", type=Path, required=True,
                        help="baseline training data (the frozen source file)")
    parser.add_argument("--engine", type=Path,
                        default=ROOT / "build" / "engine" / "Release" / "gomoku_nnue.exe")
    parser.add_argument("--datagen-exe", type=Path,
                        default=ROOT / "build" / "engine" / "Release" / "datagen.exe")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--min-rounds", type=int, default=3)
    parser.add_argument("--min-win-rate", type=float, default=0.50,
                        help="promotion threshold on candidate score rate")
    parser.add_argument("--strict-gate", action=argparse.BooleanOptionalAction,
                        default=True, help="use > instead of >= (default on)")
    parser.add_argument("--games", type=int, default=200,
                        help="gate game count (paired, even)")
    parser.add_argument("--seed", type=int, default=20261110,
                        help="gate opening seed")
    parser.add_argument("--openings", type=Path, default=None,
                        help="frozen gate openings JSON (optional)")
    parser.add_argument("--round", type=int, default=None,
                        help="run only this round number and exit (manual mode)")
    # datagen
    parser.add_argument("--datagen-workers", type=int, default=16)
    parser.add_argument("--datagen-games-per-worker", type=int, default=100)
    parser.add_argument("--datagen-depth", type=int, default=2)
    parser.add_argument("--datagen-seed", type=int, default=20261110,
                        help="base seed; round index is added per round")
    # training / export passthrough
    parser.add_argument("--train-extra", type=str, default="",
                        help="extra train.py args, shlex-split")
    parser.add_argument("--export-extra", type=str, default="",
                        help="extra export.py args, shlex-split")
    args = parser.parse_args(argv)

    if args.max_rounds < args.min_rounds:
        parser.error("--max-rounds must be >= --min-rounds")
    if not 0 < args.min_win_rate <= 1:
        parser.error("--min-win-rate must be in (0, 1]")
    if args.games < 2 or args.games % 2:
        parser.error("--games must be an even number >= 2 (paired openings)")
    if args.frozen_validation is None:
        # Without a frozen split we cannot mix old + new data leak-free; this
        # is a hard requirement of the loop, not a silent fallback.
        parser.error("--frozen-validation is required for the bootstrap loop")
    return args


def build_config(args: argparse.Namespace) -> BootstrapConfig:
    return BootstrapConfig(
        work_dir=args.work_dir,
        final_dir=args.final_dir,
        incumbent_nnue=args.incumbent_nnue,
        incumbent_pt=args.incumbent_pt,
        frozen_validation=args.frozen_validation,
        old_data=args.old_data,
        engine=args.engine,
        datagen_exe=args.datagen_exe,
        max_rounds=args.max_rounds,
        min_rounds=args.min_rounds,
        min_win_rate=args.min_win_rate,
        strict_gate=args.strict_gate,
        games=args.games,
        seed=args.seed,
        openings=args.openings,
        datagen_workers=args.datagen_workers,
        datagen_games_per_worker=args.datagen_games_per_worker,
        datagen_depth=args.datagen_depth,
        datagen_seed=args.datagen_seed,
        train_extra=shlex.split(args.train_extra),
        export_extra=shlex.split(args.export_extra),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(cfg.work_dir)
    if state is None:
        state = initial_state(cfg)
        write_json(cfg.work_dir / "state.json", state)

    if args.round is not None:
        if args.round < 1:
            raise SystemExit("--round must be >= 1")
        run_round(cfg, state, args.round)
        return 0

    started = time.monotonic()
    for round_index in range(1, cfg.max_rounds + 1):
        run_round(cfg, state, round_index)
        if should_stop(state["rounds"], min_rounds=cfg.min_rounds,
                       max_rounds=cfg.max_rounds):
            break

    stop_reason = ("max_rounds reached"
                   if len(state["rounds"]) >= cfg.max_rounds
                   else "converged (no promotion streak)")
    freeze_final(cfg, state, stop_reason)
    print(f"\nbootstrap converged after {len(state['rounds'])} round(s); "
          f"final model frozen from incumbent round "
          f"{state['incumbent']['source_round']}")
    print(f"final manifest: {cfg.final_dir / 'final_manifest.json'}")
    print(f"elapsed: {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
