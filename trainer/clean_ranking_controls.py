"""Ticket 25: compare fixed-data hidden-LR arms and deployment gains."""
import argparse
import json
import math
from pathlib import Path

import torch

import clean_search_ranking as cr
import nnue
import weights_format as wf
from distill18_fidelity import cpp_scores
from run_support import file_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise ValueError("refusing to overwrite controls")
    args.out_dir.mkdir(parents=True)
    reports = [json.loads((p/"report.json").read_text()) for p in args.runs]
    first = reports[0]
    for report in reports:
        if not report["complete"] or not report["diagnostic_only"]:
            raise ValueError("incomplete or non-diagnostic input")
        if report["split"] != first["split"]:
            raise ValueError("different splits")
        for key in ("labels", "cases", "probe"):
            if report["sources"][key] != first["sources"][key]:
                raise ValueError("different source")
        for key in set(first["config"]) | set(report["config"]):
            if key not in ("hidden_lr", "augment_d4", "steps") and report["config"][key] != first["config"][key]:
                raise ValueError("uncontrolled configuration difference")
    source = first["sources"]
    for key in ("labels", "cases", "probe"):
        if file_identity(Path(source[key]["path"])) != source[key]:
            raise ValueError("source identity changed")
    fitted, held, manifest = cr.prepare(Path(source["labels"]["path"]), Path(source["cases"]["path"]))
    if manifest != first["split"]:
        raise ValueError("reconstructed split changed")
    sets = {"fit": fitted, "holdout": held}
    samples = {k: [s for r in rows for s in r["samples"]] for k, rows in sets.items()}
    result = []
    for path, report in zip(args.runs, reports):
        checkpoint = torch.load(path/"model.pt", map_location="cpu", weights_only=False)
        if not checkpoint["diagnostic_only"]:
            raise ValueError("refusing promotable checkpoint")
        model = nnue.NNUEModel(config=nnue.ModelConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model_state"])
        weights = {k: getattr(model, k).detach().flatten().tolist()
                   for k in ("W0", "b0", "W1", "b1", "W2", "b2")}
        arm = {"run": file_identity(path/"report.json"), "hidden_lr": report["config"]["hidden_lr"],
               "steps": report["config"]["steps"], "augment_d4": report["config"].get("augment_d4", False),
               "checks_original": report["checks"], "gains": {}}
        for gain in (1, 128):
            weights.update(format_version=3, output_gain=gain)
            target = args.out_dir/f"{path.name}_gain{gain}.nnue"
            target.write_bytes(wf.pack_weights(weights))
            reloaded, _ = wf.read_weights(target)
            arm["gains"][str(gain)] = {}
            for key, rows in sets.items():
                values = cpp_scores(Path(source["probe"]["path"]), target, samples[key], "scalar")
                reference = [wf.infer_float(reloaded, s[0], s[1]) for s in samples[key][:16]]
                rounded = [int(math.copysign(math.floor(abs(v)+.5), v)) for v in reference]
                if values[:16] != rounded:
                    raise ValueError("C++/Python exported inference mismatch")
                metrics = cr.summarize(rows, values)
                degradation = metrics["pair_disagreement"]-report["final_float"][key]["pair_disagreement"]
                metrics.update(static_limit_fraction=sum(abs(v) >= wf.STATIC_LIMIT for v in values)/len(values),
                               disagreement_degradation=degradation,
                               fidelity_passed=degradation <= cr.CONFIG["export_max_degradation"],
                               parity_checked=16)
                arm["gains"][str(gain)][key] = metrics
        result.append(arm)
    write_json(args.out_dir/"report.json", {"complete": True, "diagnostic_only": True,
               "arms": result, "controlled_split_verified": True,
               "limitation": "No calibrated search units or playing-strength evidence; historical validation fitted."})
    print(json.dumps({"complete": True, "arms": [{"hidden_lr": a["hidden_lr"],
          "gains": {g: {k: {m: v for m, v in r.items() if m != "per_parent"}
                          for k, r in rs.items()} for g, rs in a["gains"].items()}}
          for a in result]}))


if __name__ == "__main__":
    main()
