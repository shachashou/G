#!/usr/bin/env python3
"""Self-tests for the ticket 11 bootstrap loop (tools/bootstrap.py).

Covers the pure decision helpers (gate promotion, stop condition, final
manifest) and the child-command constructors, without running any training or
matches. The loop body itself is exercised by the user under the budget runner.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import bootstrap as b  # noqa: E402

failures = 0


def check(cond, msg):
    global failures
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures += 1


def test_promote_from_gate():
    print("[promote_from_gate]")
    check(b.promote_from_gate({"complete": True, "gate": {"passed": True}}),
          "complete match passing the gate promotes")
    check(not b.promote_from_gate({"complete": True, "gate": {"passed": False}}),
          "gate failure keeps the incumbent")
    check(not b.promote_from_gate({"complete": False, "gate": {"passed": True}}),
          "an incomplete match never promotes")
    check(not b.promote_from_gate({}), "empty report never promotes")
    check(not b.promote_from_gate({"complete": True}), "missing gate never promotes")
    check(not b.promote_from_gate(None), "non-dict never promotes")


def test_should_stop():
    print("[should_stop]")
    check(not b.should_stop([], min_rounds=3), "no rounds -> keep going")
    check(not b.should_stop([{"promoted": True}], min_rounds=3),
          "below min_rounds -> keep going")
    check(not b.should_stop([{"promoted": True}, {"promoted": False}],
                            min_rounds=3),
          "below min_rounds -> keep going")
    check(b.should_stop([{"promoted": True}, {"promoted": False},
                         {"promoted": False}], min_rounds=3),
          "last two rounds failed -> stop")
    check(not b.should_stop([{"promoted": False}, {"promoted": True},
                             {"promoted": False}], min_rounds=3),
          "one of the last two promoted -> keep going")
    check(b.should_stop([{"promoted": False}] * 3, min_rounds=3),
          "three failures -> stop")
    check(b.should_stop([{"promoted": True}], min_rounds=3, max_rounds=1),
          "max_rounds reached short-circuits the streak check")


def test_build_final_manifest():
    print("[build_final_manifest]")
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        pt = tmp / "final.pt"
        nnue = tmp / "final.nnue"
        history = tmp / "final_history.json"
        export = tmp / "final_export_report.json"
        pt.write_bytes(b"checkpoint-bytes")
        nnue.write_bytes(b"weights-bytes")
        manifest = b.build_final_manifest(
            incumbent={"pt": str(pt), "nnue": str(nnue), "source_round": 2},
            rounds=[{"round": 1, "promoted": True},
                    {"round": 2, "promoted": True}],
            final_pt=pt, final_nnue=nnue,
            final_history=history, final_export_report=export,
            stop_reason="converged (no promotion streak)",
        )
        check(manifest["schema"] == b.FINAL_MANIFEST_SCHEMA, "schema recorded")
        check(manifest["final"]["source_round"] == 2, "source round recorded")
        check(manifest["final"]["identity"]["weights"]["sha256"] ==
              b.file_identity(nnue)["sha256"],
              "weights identity carries a sha256")
        check(manifest["final"]["identity"]["checkpoint"]["sha256"] ==
              b.file_identity(pt)["sha256"],
              "checkpoint identity carries a sha256")
        check(manifest["rounds"] == [{"round": 1, "promoted": True},
                                     {"round": 2, "promoted": True}],
              "round history is embedded")
        check("converged" in manifest["stop_reason"], "stop reason recorded")


def _cfg(tmp: Path) -> b.BootstrapConfig:
    return b.BootstrapConfig(
        work_dir=tmp / "work",
        final_dir=tmp / "final",
        incumbent_nnue=tmp / "incumbent.nnue",
        incumbent_pt=tmp / "incumbent.pt",
        frozen_validation=tmp / "frozen.json",
        old_data=tmp / "old.binpack",
        engine=tmp / "gomoku_nnue.exe",
        datagen_exe=tmp / "datagen.exe",
        max_rounds=5, min_rounds=3, min_win_rate=0.55, strict_gate=True,
        games=200, seed=1234, openings=None,
        datagen_workers=8, datagen_games_per_worker=10, datagen_depth=2,
        datagen_seed=777,
        train_extra=["--epochs", "2"],
        export_extra=[],
    )


def test_command_constructors():
    print("[command constructors]")
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        cfg = _cfg(tmp)
        out_dir = tmp / "work" / "round001" / "datagen"

        argv = b.datagen_command(cfg, 1, tmp / "incumbent.nnue", out_dir)
        joined = " ".join(str(a) for a in argv)
        check("--nnue" in argv and str(tmp / "incumbent.nnue") in joined,
              "datagen is driven by the incumbent weights")
        check("--merge" in argv, "datagen merges shards into one file")
        check(str(cfg.datagen_seed + 1) in argv, "per-round datagen seed varies")

        argv = b.train_command(cfg, tmp / "manifest.json",
                               tmp / "candidate.pt", tmp / "history.json")
        joined = " ".join(str(a) for a in argv)
        check("--frozen-validation" in argv, "train uses the frozen validation")
        check("--train-manifest" in argv, "train mixes old + new data")
        check("--epochs" in argv and "2" in argv, "train passthrough args flow")

        argv = b.gate_command(cfg, tmp / "candidate.nnue",
                              tmp / "incumbent.nnue", tmp / "gate.json")
        joined = " ".join(str(a) for a in argv)
        check(argv[argv.index("--min-win-rate") + 1] == "0.55",
              "promotion threshold is configurable")
        check("--strict-gate" in argv, "strict gate is on by default")
        check("--verify-backend" in argv, "gate verifies the NNUE backend")
        check("--paired-openings" in argv, "gate uses paired openings")
        # engine A is the candidate, engine B the incumbent
        a = argv[argv.index("--engine-a") + 1]
        b_cmd = argv[argv.index("--engine-b") + 1]
        check("candidate.nnue" in a and "incumbent.nnue" in b_cmd,
              "candidate is engine A, incumbent is engine B")


def main():
    test_promote_from_gate()
    test_should_stop()
    test_build_final_manifest()
    test_command_constructors()
    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
