"""Compare completed independent stage-19-B fits without starting training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"trainer"))
import local_ranking_fit as fit
from teacher_labels import file_identity, verify_identity
from run_support import write_json


def summarise(manifest_path, float_path, qat_path, output):
    manifest = fit.load_manifest(manifest_path)
    result = {"stage": "19-B", "complete": False, "diagnostic_only": True, "fit_manifest": file_identity(manifest_path),
              "eligible_for_screening": False, "top1_gate_passed": False}
    try:
        reports = {"float": fit.read(float_path), "qat": fit.read(qat_path)}
        for arm, report in reports.items():
            if (not report.get("complete") or report.get("arm") != arm or not report.get("from_scratch")
                    or report["trained_quantized"] != (arm == "qat") or report["config"] != manifest["config"]
                    or report["fit_manifest"]["sha256"] != file_identity(manifest_path)["sha256"]
                    or report["optimizer_steps"] != manifest["config"]["steps"]
                    or report["samples_exposed"] != manifest["config"]["steps"]*manifest["config"]["batch_size"]
                    or report["initial_master_sha256"] != manifest["initial_master_sha256"]
                    or report["batch_schedule_sha256"] != manifest["batch_schedule_sha256"]):
                raise ValueError(f"incomplete or non-comparable {arm} fit")
            verify_identity(report["checkpoint"])
        passed = {arm: report["fit_gate"]["passed"] for arm, report in reports.items()}
        interpretation = ("both tiny-parent fits passed; inspect same-parent child ranking before a distribution control" if all(passed.values()) else
                          "independent float passed but QAT failed; quantisation-training control is the next priority" if passed["float"] else
                          "QAT passed but float failed; inspect optimisation evidence before a causal claim" if passed["qat"] else
                          "neither parent fit passed; no automatic child-training stage")
        eligible_c = []
        for arm, report in reports.items():
            local = report["evaluation"]["fit_parent_local"].get("trained_forward", {})
            excess, centred = local.get("teacher_best_excess"), local.get("pooled_centred_mse_ratio")
            if passed[arm] and excess is not None and centred is not None and (excess < .25 or centred > .75):
                eligible_c.append(arm)
        result.update(complete=True, smoke_test=manifest["config"]["smoke_test"],
                      official_stage_b_complete=not manifest["config"]["smoke_test"], fit_passed=passed,
                      eligible_for_stage_c=eligible_c if not manifest["config"]["smoke_test"] else [],
                      reason=interpretation, reports={"float": file_identity(float_path), "qat": file_identity(qat_path)},
                      metrics={arm: {"fit": r["evaluation"]["fit_parents"], "fit_local": r["evaluation"]["fit_parent_local"],
                                     "validation": r["evaluation"]["validation_local"]} for arm, r in reports.items()})
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(complete=False, reason=str(error), eligible_for_stage_c=[])
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=ROOT/"data/m5/distill19/fit_b")
    parser.add_argument("--float-report", type=Path)
    parser.add_argument("--qat-report", type=Path)
    args = parser.parse_args()
    directory = args.dir.resolve()
    result = summarise(directory/"fit_manifest.json", args.float_report or directory/"float_initial/report.json",
                        args.qat_report or directory/"qat_initial/report.json", directory/"summary.json")
    print(result["reason"])
    return 0 if result["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
