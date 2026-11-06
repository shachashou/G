"""Full held-out and legal-child fidelity for a ticket-18 exported candidate."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import torch

import export
import features
import move_correlation as mc
import targets
from teacher_labels import file_identity, verify_identity
import weights_format as wf


def model_predictions(model, samples, quantized=False):
    result = []
    with torch.no_grad():
        for offset in range(0, len(samples), 512):
            batch = samples[offset:offset+512]
            x = torch.tensor([features.dense_vector(s[0]) for s in batch], dtype=torch.float32)
            stm = torch.tensor([s[1] for s in batch], dtype=torch.long)
            result.extend(model(x, stm, quantized=quantized).tolist())
    return result


def cpp_scores(probe, weights, samples, backend):
    lines = "".join(f"{s[1]} " + " ".join(map(str, s[0])) + "\n" for s in samples)
    proc = subprocess.run([str(Path(probe).resolve()), "--weights", str(Path(weights).resolve()),
                           "--backend", backend, "infer"], input=lines, capture_output=True, text=True, timeout=120)
    if proc.returncode:
        raise ValueError(f"C++ {backend} inference failed: {proc.stderr}")
    values = [int(v) for v in proc.stdout.splitlines()]
    if len(values) != len(samples):
        raise ValueError("C++ fidelity response count mismatch")
    return values


def run(args, out):
    if not args.all_validation or args.weights is None or args.probe is None:
        raise ValueError("linear fidelity requires --all-validation --weights --probe")
    model, payload = export.load_model(args.checkpoint)
    train_set, val_set, frozen, _ = targets.checkpoint_datasets(payload, args.data, args.frozen_validation)
    try:
        samples = list(val_set.iter_samples())
        record_ids = list(val_set.indices)
    finally:
        train_set.close()
        val_set.close()
    contract = payload["target_contract"]
    metric_path = verify_identity(frozen["move_metric_contract"])
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    if file_identity(args.probe)["sha256"] != metric["probe"]["sha256"]:
        raise ValueError("fidelity probe differs from frozen probe")
    cases_path = verify_identity(metric["cases"])
    cases, _ = mc.load_cases(cases_path, metric_path)
    original_weights = model.quantized_weights()
    weights, meta = wf.read_weights(args.weights)
    if meta["version"] != 2 or meta["output_gain"] != model.config.output_gain:
        raise ValueError("candidate format/gain differs from checkpoint")
    export_report_path = args.export_report
    if export_report_path is None:
        raise ValueError("linear fidelity requires --export-report")
    export_report = json.loads(Path(export_report_path).read_text(encoding="utf-8"))
    if (export_report["checkpoint_identity"]["sha256"] != file_identity(args.checkpoint)["sha256"]
            or file_identity(export_report["weights_file"])["sha256"] != file_identity(args.weights)["sha256"]):
        raise ValueError("export report/checkpoint/candidate identity mismatch")
    if (export_report.get("complete") is not True
            or export_report["target_contract_identity"]["sha256"] != payload["target_contract_identity"]["sha256"]):
        raise ValueError("export report has incomplete/different target contract")
    for key in ("W0", "b0", "W1", "b1", "W2"):
        if list(original_weights[key]) != list(weights[key]):
            raise ValueError("export modified inference weights outside bias calibration")
    if not export_report["quantisation"]["calibrated"] and list(original_weights["b2"]) != list(weights["b2"]):
        raise ValueError("uncalibrated export changed output bias")
    variants = {"original_float": model_predictions(model, samples), "ste": model_predictions(model, samples, True),
                "raw_integer": export.int_evals(original_weights, samples),
                "final_integer": export.int_evals(weights, samples)}
    if variants["ste"] != variants["raw_integer"]:
        raise ValueError("held-out STE/integer mismatch")
    for backend in ("scalar", "avx2"):
        if cpp_scores(args.probe, args.weights, samples, backend) != variants["final_integer"]:
            raise ValueError(f"held-out {backend}/Python mismatch")
    children = []
    for line in mc.child_lines(cases):
        stm, board = line.split()
        sparse = [i for i, c in enumerate(board) if c == "x"] + [225+i for i, c in enumerate(board) if c == "o"]
        children.append((sparse, int(stm), 0, 0))
    # Reference, scalar and AVX2 cover EVERY frozen primary child, not a sample.
    child_final = export.int_evals(weights, children)
    child_raw = export.int_evals(original_weights, children)
    child_ste = model_predictions(model, children, True)
    if child_ste != child_raw:
        raise ValueError("child STE/Python integer mismatch")
    for backend in ("scalar", "avx2"):
        if cpp_scores(args.probe, args.weights, children, backend) != child_final:
            raise ValueError(f"child {backend}/Python mismatch")
    child_float = model_predictions(model, children)
    child_rankings = {}
    for name, values in (("original_float", child_float), ("ste", child_ste), ("final_integer", child_final)):
        ranking, _ = mc.aggregate_cases(cases, values)
        child_rankings[name] = ranking
    correlation = mc.assess_cases(Path(args.probe), Path(args.weights), "nnue", "avx2",
                                  cases_path, metric_path, out / "move_correlation.json")
    if correlation["results"] != child_rankings["final_integer"]:
        raise ValueError("board/sparse probe ranking differs")
    result = {"schema": 2, "complete": True, "checkpoint": file_identity(args.checkpoint),
              "candidate": file_identity(args.weights), "export_report": file_identity(export_report_path),
              "target_contract": payload["target_contract_identity"], "frozen_validation": file_identity(args.frozen_validation),
              "metric_contract": file_identity(metric_path), "move_cases": file_identity(cases_path),
              "manifest": {"assessment_record_ids": record_ids, "validation_membership": len(frozen["split_identity"]["validation_record_ids"]),
                           "assessment_records": len(samples), "validity_filtered": len(frozen["split_identity"]["validation_record_ids"])-len(samples),
                           "baseline_training_records": contract["constants"]["training_records"],
                           "d4_position_separation_verified": True},
              "cpp": {"exact": True, "probe": file_identity(args.probe), "assessment_records": len(samples),
                      "all_primary_children": True, "child_records": len(children)},
              "metrics": {name: targets.metrics(values, [s[2] for s in samples], contract["loss_normalizer"], contract["constants"])
                          for name, values in variants.items()},
              "comparisons": {"original_vs_final": export.error_stats(variants["original_float"], variants["final_integer"]),
                              "ste_vs_final": export.error_stats(variants["ste"], variants["final_integer"])},
              "child_rankings": child_rankings,
              "child_static_limit_hits": sum(abs(v) >= wf.STATIC_LIMIT for v in child_final),
              "move_correlation": file_identity(out / "move_correlation.json"),
              "limitation": "Repeated use of the existing validation games; no independent strength claim."}
    return result
