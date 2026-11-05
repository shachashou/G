#!/usr/bin/env python3
"""Staged M5 evidence: fixed-set fit and original-model export fidelity.

Run long/uncertain jobs through tools/budget_run.py. Each output directory is
new and self-contained; these diagnostics never replace the reference model.
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

import binpack
import dataset
import export
import features
import nnue
import weights_format as wf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import file_identity, source_identity, write_json


def baseline_report(train_scores, train_wdl, eval_scores, eval_wdl, predictions,
                    score_scale=400) -> dict:
    """Training-fitted constant/priors, evaluated on exactly the model's set."""
    target = [math.tanh(v / score_scale) for v in eval_scores]
    constant = statistics.mean(math.tanh(v / score_scale) for v in train_scores)
    priors = [max(1e-12, train_wdl.count(k) / len(train_wdl)) for k in range(3)]
    total = sum(priors)
    priors = [p / total for p in priors]
    base = statistics.mean((constant - y) ** 2 for y in target)
    mse = statistics.mean((math.tanh(p / score_scale) - y) ** 2
                          for p, y in zip(predictions, target))
    return {"constant_target": constant, "class_priors": priors,
            "baseline_mse": base,
            "baseline_ce": statistics.mean(-math.log(priors[k]) for k in eval_wdl),
            "model_mse": mse, "mse_ratio": mse / base if base > 0 else None}


def select_records(reader, pool, count: int, seed: int) -> list[int]:
    """Fixed sparse sample, balanced across available side-to-move heads."""
    heads = [[], []]
    for index in pool:
        heads[reader[index].stm].append(index)
    if not all(heads):
        raise ValueError("diagnostic split must contain both side-to-move heads")
    rng = random.Random(seed)
    for head in heads:
        rng.shuffle(head)
    selected = []
    while len(selected) < count and any(heads):
        for head in heads:
            if head and len(selected) < count:
                selected.append(head.pop())
    return selected


def samples_at(reader, indices, clip):
    return [(reader[i].feature_indices(), reader[i].stm,
             dataset.clip_score(float(reader[i].score), clip), reader[i].wdl)
            for i in indices]


def tensors(samples, device="cpu"):
    return (torch.tensor([features.dense_vector(s[0]) for s in samples],
                         dtype=torch.float32, device=device),
            torch.tensor([s[1] for s in samples], dtype=torch.long, device=device),
            torch.tensor([s[2] for s in samples], dtype=torch.float32, device=device),
            torch.tensor([s[3] for s in samples], dtype=torch.long, device=device))


def metrics(predictions, samples, model):
    p = torch.tensor(predictions, dtype=torch.float32)
    score = torch.tensor([s[2] for s in samples], dtype=torch.float32)
    labels = torch.tensor([s[3] for s in samples], dtype=torch.long)
    c = model.config
    mse, ce = nnue.mixed_losses(p, score, labels, c.score_scale,
                                c.wdl_temperature, model.wdl_bias.detach().cpu())
    _, ce_no_bias = nnue.mixed_losses(p, score, labels, c.score_scale,
                                     c.wdl_temperature)
    return {"mse": float(mse), "ce_training_bias": float(ce),
            "ce_without_bias": float(ce_no_bias),
            "output": {"min": min(predictions), "max": max(predictions),
                       "mean": statistics.mean(predictions)},
            "per_head": {str(h): {"count": sum(s[1] == h for s in samples),
                                  "outputs": [p for p, s in zip(predictions, samples)
                                              if s[1] == h]} for h in (0, 1)}}


def prediction_comparison(a, b, scale):
    return {"score_units": export.error_stats(a, b),
            "target_space": export.error_stats(
                [math.tanh(v / scale) for v in a],
                [math.tanh(v / scale) for v in b])}


def fidelity(args, out: Path) -> dict:
    model, payload = export.load_model(args.checkpoint)
    if payload.get("diagnostic_only"):
        raise ValueError("same-set overfit checkpoint has no independent validation split")
    training = payload.get("train_args", {})
    if "seed" not in training or "val_fraction" not in training:
        raise ValueError("checkpoint lacks original seed/val_fraction; cannot reconstruct holdout")
    actual_identity = file_identity(args.data)
    recorded_identity = payload.get("data_identity")
    verified = bool(recorded_identity and recorded_identity.get("sha256") == actual_identity["sha256"])
    if recorded_identity and not verified:
        raise ValueError("dataset SHA-256 differs from checkpoint training dataset")
    if not verified and not args.allow_unverified_data:
        raise ValueError("legacy checkpoint has no dataset hash; --allow-unverified-data explicitly marks split independence unverified")
    if not verified and training.get("data"):
        original_path = Path(training["data"])
        if not original_path.is_absolute():
            original_path = ROOT / original_path
        if original_path.exists() and file_identity(original_path)["sha256"] != actual_identity["sha256"]:
            raise ValueError("supplied data differs from legacy checkpoint data path")
    seed = training["seed"]
    fraction = training["val_fraction"]
    clip = training.get("score_clip", 0.0)
    with binpack.BinpackReader(args.data) as reader:
        game_map = reader.game_index_map()
        held = dataset.choose_validation_games(reader.header.game_count, fraction, seed)
        train_ids = [i for i, g in enumerate(game_map) if g not in held]
        eval_ids = [i for i, g in enumerate(game_map) if g in held]
        limit = training.get("limit")
        if limit is not None:
            if limit < 1:
                raise ValueError("invalid checkpoint training limit")
            train_ids, eval_ids = train_ids[:limit], eval_ids[:limit]
        cal_ids = select_records(reader, train_ids, args.samples, args.seed)
        assessment_ids = select_records(reader, eval_ids, args.samples, args.seed + 1)
        calibration = samples_at(reader, cal_ids, clip)
        assessment = samples_at(reader, assessment_ids, clip)
        # Baseline uses the entire original training split, without validation labels.
        train_scores = [dataset.clip_score(reader[i].score, clip) for i in train_ids]
        train_wdl = [reader[i].wdl for i in train_ids]
        manifest = {"calibration_indices": cal_ids, "assessment_indices": assessment_ids,
                    "calibration_games": sorted({game_map[i] for i in cal_ids}),
                    "assessment_games": sorted({game_map[i] for i in assessment_ids}),
                    "validation_games": sorted(held), "split_seed": seed,
                    "val_fraction": fraction, "score_clip": clip,
                    "baseline_train_records": len(train_ids),
                    "training_limit": limit,
                    "data_identity": {"verified": verified,
                        "assessment_independence_verified": verified,
                        "limitation": None if verified else "legacy dataset identity not sealed; held-out membership assumed from split settings"},
                    "checkpoint_training_data": str(training.get("data", "unknown")),
                    "split_provenance": "reconstructed from checkpoint train_args; not a new test set"}
    weights = model.quantized_weights()
    def forwards(samples, quantized=False):
        x, stm, _, _ = tensors(samples)
        with torch.no_grad():
            return model(x, stm, quantized=quantized).tolist()
    original, ste = forwards(assessment), forwards(assessment, True)
    raw = export.int_evals(weights, assessment)
    calibrated = copy.deepcopy(weights)
    bias = export.calibrate_bias(calibrated, forwards(calibration),
                                 export.int_evals(weights, calibration),
                                 [s[1] for s in calibration])
    final = export.int_evals(calibrated, assessment)
    weights_path = out / "candidate.nnue"
    weights_path.write_bytes(wf.pack_weights(calibrated))
    variants = {"original_float": original, "ste": ste, "raw_integer": raw,
                "final_calibrated_integer": final}
    cpp = {"available": False, "reason": "--probe not supplied"}
    if args.probe:
        lines = "\n".join(f"{s[1]} " + " ".join(map(str, s[0])) for s in assessment) + "\n"
        values = {}
        for backend in ("scalar", "avx2"):
            result = subprocess.run([str(args.probe), "--weights", str(weights_path),
                                     "--backend", backend, "infer"], input=lines,
                                    text=True, capture_output=True, timeout=30, check=True)
            values[backend] = [int(v) for v in result.stdout.splitlines()]
            if values[backend] != final:
                raise ValueError(f"{backend} C++ does not match exported integer model")
        cpp = {"available": True, "probe": file_identity(args.probe),
               "scalar": values["scalar"], "avx2": values["avx2"], "exact": True}
    baselines = {name: baseline_report(train_scores, train_wdl,
                 [s[2] for s in assessment], [s[3] for s in assessment], predictions,
                 model.config.score_scale) for name, predictions in variants.items()}
    variant_metrics = {name: metrics(p, assessment, model) for name, p in variants.items()}
    for name, baseline in baselines.items():
        baseline["model_ce"] = variant_metrics[name]["ce_training_bias"]
        baseline["ce_ratio"] = baseline["model_ce"] / baseline["baseline_ce"] if baseline["baseline_ce"] > 0 else None
    return {"mode": "fidelity", "checkpoint": file_identity(args.checkpoint),
            "candidate": file_identity(weights_path), "manifest": manifest,
            "head_bias_calibration": bias, "cpp": cpp,
            "predictions": variants, "metrics": variant_metrics,
            "baseline": baselines,
            "comparisons": {"original_vs_raw": prediction_comparison(original, raw, model.config.score_scale),
                            "original_vs_final": prediction_comparison(original, final, model.config.score_scale),
                            "ste_vs_raw": prediction_comparison(ste, raw, model.config.score_scale)},
            "interpretation": "Arithmetic parity and export fidelity are separate. No capacity conclusion follows from this report alone."}


def activation_stats(model, x):
    with torch.no_grad():
        a = x @ model.W0 + model.b0
        b = a.clamp(0, model.config.qa) @ model.W1 / model.config.qa + model.b1
        return {name: {"low_fraction": float((v <= 0).float().mean()),
                       "high_fraction": float((v >= hi).float().mean())}
                for name, v, hi in (("accumulator", a, model.config.qa),
                                     ("l1", b, model.config.qb))}


def overfit(args, out: Path) -> dict:
    torch.manual_seed(args.seed)
    with binpack.BinpackReader(args.data) as reader:
        ids = select_records(reader, range(reader.available), args.samples, args.seed)
        samples = samples_at(reader, ids, args.score_clip)
        game_map = reader.game_index_map()
        manifest = {"record_indices": ids, "game_ids": sorted({game_map[i] for i in ids}),
                    "seed": args.seed, "augmentation": False,
                    "score_clip": args.score_clip, "evaluation": "same training samples; not generalization"}
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    x, stm, score, labels = tensors(samples, device)
    arms = {}
    for quantized in (False, True):
        name = "ste" if quantized else "float"
        model = nnue.NNUEModel(seed=args.seed).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        curve = []
        for step in range(args.steps + 1):
            prediction = model(x, stm, quantized=quantized)
            mse, ce = nnue.mixed_losses(prediction, score, labels,
                                        model.config.score_scale,
                                        model.config.wdl_temperature, model.wdl_bias)
            loss = mse + args.wdl_weight * ce + args.quant_reg * model.quantization_regularizer()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            gradients = {n: float(p.grad.norm()) for n, p in model.named_parameters()
                         if p.grad is not None}
            if step % args.log_every == 0 or step == args.steps:
                row = {"step": step, "loss": float(loss.detach()),
                       "mse": float(mse.detach()), "ce": float(ce.detach()),
                       "gradient_norms": gradients,
                       "activations_float": activation_stats(model, x),
                       "prediction": metrics(prediction.detach().cpu().tolist(), samples, model)}
                curve.append(row)
                write_json(out / f"{name}_curve.json", {"complete": False, "curve": curve})
                print(f"{name} step {step}/{args.steps}: mse={row['mse']:.6f}", flush=True)
            if step != args.steps:
                optimiser.step()
        # Keep the final same-set checkpoint; never select on independent validation.
        torch.save({"model_state": model.cpu().state_dict(), "config": vars(model.config),
                    "train_args": vars(args), "diagnostic_only": True}, out / f"{name}.pt")
        write_json(out / f"{name}_curve.json", {"complete": True, "curve": curve})
        arms[name] = {"curve": curve, "checkpoint": file_identity(out / f"{name}.pt")}
    return {"mode": "overfit", "manifest": manifest, "arms": arms,
            "interpretation": "Compare same-set float/STE fit, gradients and clipping before changing capacity. No held-out strength claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("fidelity", "overfit"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--allow-unverified-data", action="store_true",
                        help="legacy checkpoint only: report assumed split, not verified independence")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wdl-weight", type=float, default=0.2)
    parser.add_argument("--quant-reg", type=float, default=0.05)
    parser.add_argument("--score-clip", type=float, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.samples < 2 or args.steps < 1 or args.log_every < 1 or args.threads < 1:
        parser.error("samples >=2, steps/log-every/threads >=1 required")
    if args.mode == "fidelity" and not args.checkpoint:
        parser.error("fidelity requires --checkpoint")
    if args.out_dir.exists():
        parser.error("refusing to overwrite existing experiment directory")
    args.out_dir.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    report = {"complete": False, "mode": args.mode, "data": file_identity(args.data),
              "source": source_identity(), "config": {k: str(v) if isinstance(v, Path) else v
                                                        for k, v in vars(args).items()},
              "torch": torch.__version__}
    write_json(args.out_dir / "report.json", report)
    try:
        report.update(fidelity(args, args.out_dir) if args.mode == "fidelity"
                      else overfit(args, args.out_dir))
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        write_json(args.out_dir / "report.json", report)
    print(f"Report: {args.out_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
