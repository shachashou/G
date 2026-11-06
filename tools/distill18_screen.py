"""Explicit, single paired screening run AFTER ticket-18 Top-1 eligibility."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from distill18_summary import summarize
import match_runner as mr
from run_support import file_identity, write_json


def reject_opening_overlap(proposed, prior_manifests):
    signatures = {json.dumps(p, separators=(",", ":")) for p in proposed["positions"]}
    if len(signatures) != 10:
        raise ValueError("screening requires ten distinct paired openings")
    for path, manifest in prior_manifests:
        if signatures & {json.dumps(p, separators=(",", ":")) for p in manifest["positions"]}:
            raise ValueError(f"screening openings overlap {path}; seed must not be silently changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    directory = args.dir.resolve()
    status = summarize(directory, directory/"summary.json")
    if not status["eligible_for_screening"]:
        raise ValueError("no audited Top-1 candidate; screening blocked")
    result_path = directory/"screening.json"
    opening_path = directory/"screening_openings.json"
    if result_path.exists() or opening_path.exists():
        raise ValueError("screening already frozen/attempted; refusing a second run")
    engine = Path("build/engine/Release/gomoku_nnue.exe").resolve()
    weight = Path(status["arms"][status["selected_arm"]]["candidate"]["path"])
    exe_id, weight_id = file_identity(engine), file_identity(weight)
    command = ["--games", "20", "--paired-openings", "--seed", "20261118", "--strict-gate",
               "--min-win-rate", ".65", "--verify-backend", "--engine-a", f'"{engine}" --depth 2 --nnue "{weight}"',
               "--engine-b", f'"{engine}" --depth 2', "--output", str(result_path)]
    manifest = mr.opening_manifest(mr.parse_args(command))
    prior = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in Path("data/m5").rglob("*openings*.json")
             if path.resolve() != opening_path]
    reject_opening_overlap(manifest, prior)
    write_json(opening_path, manifest)
    write_json(directory/"screening_identity.json", {"selected_arm": status["selected_arm"],
              "candidate": weight_id, "engine": exe_id, "openings": file_identity(opening_path),
              "top1_summary": file_identity(directory/"summary.json")})
    code = mr.main([*command, "--openings", str(opening_path)])
    if file_identity(engine) != exe_id or file_identity(weight) != weight_id:
        raise ValueError("engine or candidate changed during screening")
    summarize(directory, directory/"summary.json")
    return code


if __name__ == "__main__":
    sys.exit(main())
