"""Audit ticket-18 artifacts and select at most one screening candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer"))
import distill18 as experiment
import move_correlation as mc
from teacher_labels import file_identity, verify_identity
from run_support import write_json


def rows_from(report):
    path = verify_identity(report["per_case"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("missing/duplicate per-case records")
    top1 = sum(r["top1_credit"] for r in rows)/len(rows)
    if top1 != report["results"]["top1_tie_adjusted"]:
        # statistics.mean uses compensated/exact arithmetic; compare credit
        # totals with a tight arithmetic tolerance, never rounded gate values.
        import math
        if not math.isclose(top1, report["results"]["top1_tie_adjusted"], rel_tol=1e-13, abs_tol=1e-15):
            raise ValueError("aggregate/per-case Top-1 mismatch")
    return rows


def gate(top1, historical, teacher, interval, valid):
    checks = {"minimum_top1": top1 >= .20,
              "historical_improvement": top1-historical >= .10,
              "teacher_fraction": top1 >= .50*teacher,
              "cluster_lower_bound": interval[0] > 0, "valid": bool(valid)}
    return {"passed": all(checks.values()), "checks": checks,
            "top1": top1, "best_historical": historical, "teacher_top1": teacher,
            "cluster_95_percent_interval": interval}


def summarize(directory: Path, output: Path):
    plan, metric = experiment.verify_plan(directory)
    plan_id = file_identity(directory / "experiment_plan.json")
    result = {"schema": 1, "plan": plan_id, "experiment_complete": False,
              "top1_gate_passed": False, "screening_passed": False,
              "eligible_for_screening": False, "eligible_for_ticket16": False, "arms": {},
              "formal_gate_run": False}
    try:
        stage = experiment.read(directory / "stage_a.json")
        if not stage.get("complete") or not stage.get("passed") or stage["plan"]["sha256"] != plan_id["sha256"]:
            raise ValueError("stage A incomplete/failed or belongs to another plan")
        baseline_path = verify_identity(stage["baselines"])
        verify_identity(stage["preflight"])
        baseline = experiment.read(baseline_path)
        references = []
        for name, identity in baseline["reports"].items():
            report = experiment.read(verify_identity(identity))
            if report["metric_contract"]["sha256"] != plan["inputs"]["metric_contract.json"]["sha256"] or not report.get("complete"):
                raise ValueError("historical metric identity mismatch")
            if name != "handcrafted":
                if report["weights"]["sha256"] != plan["references"][name]["sha256"]:
                    raise ValueError("historical weight identity mismatch")
                references.append(rows_from(report))
        if len(references) != 4:
            raise ValueError("four historical references required")
        for arm in ("d", "s"):
            history_path = directory / arm / "history.json"
            fidelity_path = directory / arm / "fidelity" / "report.json"
            history, fidelity = experiment.read(history_path), experiment.read(fidelity_path)
            correlation_path = verify_identity(fidelity["move_correlation"])
            correlation = experiment.read(correlation_path)
            verify_identity(fidelity["checkpoint"])
            verify_identity(fidelity["candidate"])
            if not history.get("complete") or not fidelity.get("complete") or not correlation.get("complete"):
                raise ValueError(f"{arm} incomplete experiment artifacts")
            for key, value in plan["hyperparameters"].items():
                if history["config"].get(key) != value:
                    raise ValueError(f"{arm}: hyperparameter {key} differs from sealed plan")
            if history["optimisation"]["optimizer_steps"] != 1600 or not history["optimisation"]["cap_reached"]:
                raise ValueError(f"{arm} did not complete 1600 updates")
            if file_identity(history["checkpoint"])["sha256"] != fidelity["checkpoint"]["sha256"]:
                raise ValueError(f"{arm} history/checkpoint identity mismatch")
            expected_target = plan["inputs"][f"target_{arm}.json"]["sha256"]
            if history["target_contract_identity"]["sha256"] != expected_target or fidelity["target_contract"]["sha256"] != expected_target:
                raise ValueError(f"{arm} target identity mismatch")
            if fidelity["frozen_validation"]["sha256"] != plan["inputs"]["frozen_validation.json"]["sha256"]:
                raise ValueError(f"{arm} frozen identity mismatch")
            if (correlation["weights"]["sha256"] != fidelity["candidate"]["sha256"]
                    or correlation["metric_contract"]["sha256"] != plan["inputs"]["metric_contract.json"]["sha256"]
                    or correlation["backend"] != "avx2" or correlation["mode"] != "nnue"
                    or correlation["probe"]["sha256"] != plan["probe"]["sha256"]):
                raise ValueError(f"{arm} final backend/weight/metric mismatch")
            candidate_rows = rows_from(correlation)
            interval = mc.cluster_interval(candidate_rows, references, plan["bootstrap_repeats"], plan["bootstrap_seed"])
            valid = (metric["coverage_passed"] and baseline["sensitivity_passed"]
                     and fidelity["cpp"]["exact"] and fidelity["cpp"]["all_primary_children"]
                     and fidelity["child_static_limit_hits"] == 0 and correlation["static_limit_hits"] == 0)
            decision = gate(correlation["results"]["top1_tie_adjusted"], baseline["best_historical_top1"], baseline["teacher_top1"], interval, valid)
            result["arms"][arm] = {"gate": decision, "history": file_identity(history_path),
                                   "fidelity": file_identity(fidelity_path), "candidate": fidelity["candidate"],
                                   "correlation": file_identity(correlation_path), "optimisation": history["optimisation"]}
        if result["arms"]["d"]["optimisation"]["samples_exposed"] != result["arms"]["s"]["optimisation"]["samples_exposed"]:
            raise ValueError("equal-step arms exposed different sample counts")
        result["experiment_complete"] = True
        eligible = [a for a in ("d", "s") if result["arms"][a]["gate"]["passed"]]
        result["top1_gate_passed"] = bool(eligible)
        result["eligible_for_screening"] = bool(eligible)
        result["selected_arm"] = max(eligible, key=lambda a: (result["arms"][a]["gate"]["top1"], a == "d")) if eligible else None
        result["reason"] = "one candidate eligible for paired screening" if eligible else "complete negative Top-1 result; no screening"
        if eligible and (directory/"screening.json").exists():
            import match_runner as mr
            screen = experiment.read(directory/"screening.json")
            identity = experiment.read(directory/"screening_identity.json")
            for name in ("candidate", "engine", "openings"):
                verify_identity(identity[name])
            selected = result["selected_arm"]
            config = screen["config"]
            if (identity["selected_arm"] != selected
                    or identity["candidate"]["sha256"] != result["arms"][selected]["candidate"]["sha256"]
                    or screen["provenance"]["A"]["weights"]["sha256"] != identity["candidate"]["sha256"]
                    or any(screen["provenance"][h]["exe"]["sha256"] != identity["engine"]["sha256"] for h in ("A", "B"))
                    or screen["openings"]["sha256"] != identity["openings"]["sha256"]):
                raise ValueError("screening identity mismatch")
            games = screen["games"]
            complete = (screen.get("complete") is True and len(games) == 20
                        and all(not g["anomalies"] for g in games)
                        and all(config[k] == v for k, v in {"games":20,"rule":2,"seed":20261118,
                            "paired_openings":True,"verify_backend":True,"strict_gate":True,
                            "min_win_rate":.65,"timeout_turn":0,"max_plies":225}.items()))
            passed = complete and mr.gate_result(sum(g["result"] == "A-win" for g in games),
                                                 sum(g["result"] == "draw" for g in games), 20, .65, True)["passed"]
            result.update(screening_passed=passed, eligible_for_ticket16=passed,
                          screening=file_identity(directory/"screening.json"),
                          reason="screening passed; candidate eligible for ticket 16" if passed else "screening incomplete/negative; ticket 16 remains blocked")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(experiment_complete=False, top1_gate_passed=False, eligible_for_screening=False,
                      selected_arm=None, reason=str(exc))
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or args.dir / "summary.json"
    result = summarize(args.dir.resolve(), output)
    print(result["reason"])
    return 0 if result["experiment_complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
