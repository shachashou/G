"""Ticket 19 follow-up: pairwise ranking supervision (RankNet logistic).

Reuses the frozen stage-C manifest (64 primary parents + children + handcrafted
H labels) and trains float/QAT arms with a PURE pairwise logistic loss over
candidate pairs within each parent. This directly tests whether relative move
ordering can be learned, the hypothesis stage C pointed to.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch

import diagnose
import local_ranking_child as child_mod
import local_ranking_diagnosis as ld
import nnue
import targets
import train
import weights_format as wf
from distill18_fidelity import model_predictions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import file_identity, write_json

CONFIG = {"steps": 1000, "batch_pairs": 256, "initial_seed": 42, "batch_seed": 42,
          "lr": .1, "hidden_lr": .001, "weight_decay": 1e-5, "tie_epsilon": 0,
          "pointwise_weight": 0.0, "quant_reg": 0.0, "excess_threshold": .75, "disagreement_threshold": .10}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_pairs(manifest):
    """All (i, j) child index pairs, flattened over parents, with move_score[i] > move_score[j]."""
    per_parent = manifest["children"]
    offsets, cursor = [], 0
    for entries in per_parent:
        offsets.append(cursor)
        cursor += len(entries)
    flat = [e for entries in per_parent for e in entries]
    move_scores = [-e["teacher"] for e in flat]  # parent-move perspective
    pairs = []
    for p, entries in enumerate(per_parent):
        base = offsets[p]
        n = len(entries)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if move_scores[base + i] > move_scores[base + j] + CONFIG["tie_epsilon"]:
                    pairs.append((base + i, base + j))
    return flat, move_scores, pairs


def pairwise_schedule(config, num_pairs):
    rng = random.Random(config["batch_seed"])
    batches = []
    for _ in range(config["steps"]):
        batches.append([rng.randrange(num_pairs) for _ in range(config["batch_pairs"])])
    return batches


def fit(manifest_path, arm, out, requested_device="auto"):
    if arm not in ("float", "qat") or out.exists():
        raise ValueError("unknown arm or refusing to overwrite ranking fit output")
    manifest = read(manifest_path)
    if manifest.get("stage") != "C" or not manifest.get("complete"):
        raise ValueError("invalid C manifest")
    gain = manifest["config"]["output_gain"]
    torch.set_num_threads(1)
    torch.manual_seed(CONFIG["initial_seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = train.resolve_device(requested_device)
    model = nnue.NNUEModel(seed=CONFIG["initial_seed"], config=nnue.ModelConfig(format_version=2, output_gain=gain))
    model.to(device)
    flat, move_scores, pairs = build_pairs(manifest)
    if len(pairs) < CONFIG["batch_pairs"]:
        raise ValueError("too few unequal teacher pairs for ranking fit")
    samples = [(e["sparse"], e["stm"], e["teacher"], 0) for e in flat]
    x, stm, labels, _ = diagnose.tensors(samples, device)
    pair_batches = pairwise_schedule(CONFIG, len(pairs))
    pair_t = torch.tensor(pair_batches, dtype=torch.long, device=device)
    pairs_a = torch.tensor([p[0] for p in pairs], device=device)
    pairs_b = torch.tensor([p[1] for p in pairs], device=device)
    optimiser = train.make_optimizer(model, CONFIG["lr"], CONFIG["hidden_lr"])
    quantized = arm == "qat"
    out.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema": 1, "complete": False, "stage": "19-R", "diagnostic_only": True, "arm": arm,
              "trained_quantized": quantized, "manifest": file_identity(manifest_path),
              "config": CONFIG, "from_scratch": True, "device": train.device_info(device),
              "tf32": False, "num_pairs": len(pairs)}
    write_json(out / "report.json", report)
    curve, projection_hits = [], 0
    try:
        for step in range(CONFIG["steps"]):
            idx = pair_t[step]
            a = pairs_a[idx]
            b = pairs_b[idx]
            s = -model(x, stm, quantized=quantized)  # parent-move score = -child eval
            logits = s[a] - s[b]
            loss = torch.nn.functional.softplus(-logits).mean()  # -log sigmoid(logits)
            if CONFIG["quant_reg"] > 0:
                loss = loss + CONFIG["quant_reg"] * model.quantization_regularizer()
            if not torch.isfinite(loss):
                raise ValueError("non-finite ranking loss")
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            projection_hits += child_mod.project_master(model)
            if step == 0 or (step + 1) % 100 == 0 or step == CONFIG["steps"] - 1:
                with torch.no_grad():
                    pred = model(x, stm, quantized=quantized).cpu().tolist()
                # ranking metrics on training parents
                local_rows = []
                cursor = 0
                for case, entries in zip(manifest["selected_parents"], manifest["children"]):
                    n = len(entries)
                    h = [e["teacher"] for e in entries]
                    pvals = pred[cursor:cursor + n]
                    local_rows.append(ld.local_scores(h, pvals, case["moves"].index(case["teacher_move"])))
                    cursor += n
                summary = ld.local_summary([{"local": {"trained_forward": r}} for r in local_rows])["trained_forward"]
                point_fit = targets.metrics(pred, [e["teacher"] for e in flat], gain, manifest["constants"])
                curve.append({"step": step + 1, "loss": float(loss.item()),
                              "teacher_best_excess": summary["teacher_best_excess"],
                              "pair_disagreement": summary["macro_pair_disagreement"],
                              "child_mse_ratio": point_fit["mse_ratio"]})
        evaluation = evaluate(model, quantized, manifest, gain)
        checks = {"teacher_best_excess": evaluation["local"]["teacher_best_excess"] >= CONFIG["excess_threshold"],
                  "pair_disagreement": evaluation["local"]["macro_pair_disagreement"] <= CONFIG["disagreement_threshold"]}
        torch.save({"diagnostic_only": True, "stage": "19-R", "arm": arm,
                    "config": vars(model.config), "model_state": model.state_dict(),
                    "manifest_identity": file_identity(manifest_path), "quant_forward": quantized}, out / "model.pt")
        report.update(complete=True, optimizer_steps=CONFIG["steps"],
                      samples_exposed=CONFIG["steps"] * CONFIG["batch_pairs"],
                      curve=curve, evaluation=evaluation,
                      master_projection_element_hits=projection_hits,
                      ranking_gate={"passed": all(checks.values()), "checks": checks},
                      elapsed_seconds=time.monotonic() - started)
        write_json(out / "report.json", report)
        print(json.dumps({"complete": True, "arm": arm, "ranking_gate": report["ranking_gate"],
                          "teacher_best_excess": evaluation["local"]["teacher_best_excess"],
                          "pair_disagreement": evaluation["local"]["macro_pair_disagreement"],
                          "child_mse_ratio": evaluation["point_fit"]["mse_ratio"],
                          "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))
        return report
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
        write_json(out / "report.json", report)
        raise


@torch.no_grad()
def evaluate(model, quantized, manifest, gain):
    per_parent = manifest["children"]
    flat = [e for entries in per_parent for e in entries]
    samples = [(e["sparse"], e["stm"], e["teacher"], 0) for e in flat]
    model = model.cpu()
    values = model_predictions(model, samples, quantized)
    point_fit = targets.metrics(values, [e["teacher"] for e in flat], gain, manifest["constants"])
    local_rows = []
    cursor = 0
    for case, entries in zip(manifest["selected_parents"], per_parent):
        n = len(entries)
        h = [e["teacher"] for e in entries]
        preds = values[cursor:cursor + n]
        local_rows.append(ld.local_scores(h, preds, case["moves"].index(case["teacher_move"])))
        cursor += n
    local = ld.local_summary([{"local": {"trained_forward": r}} for r in local_rows])["trained_forward"]
    return {"point_fit": point_fit, "local": local,
            "static_limit_hits": sum(abs(v) >= wf.STATIC_LIMIT for v in values)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=("float", "qat"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--quant-reg", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be >= 1")
    CONFIG["steps"] = args.steps
    CONFIG["quant_reg"] = args.quant_reg
    fit(args.manifest.resolve(), args.arm, args.out_dir.resolve(), args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
