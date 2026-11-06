"""Ticket 24: diagnostic-only RankNet fit on unsaturated S4 candidate labels.

This deliberately uses historical validation cases. Its checkpoint must never
be promoted to a playing candidate; the game holdout is diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

import diagnose
import features
import local_ranking_diagnosis as ld
import nnue
import search_score_diagnosis as sd
import train
import validation_freeze as vf
import weights_format as wf
from distill18_fidelity import cpp_scores, model_predictions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import file_identity, write_json

CONFIG = dict(seed=42, steps=5000, batch_pairs=128, lr=.1, hidden_lr=.001,
              holdout_fraction=.25, fit_max=.30, fit_reduction=.15,
              export_max_degradation=.01, holdout_max=.35, holdout_reduction=.10)


def prepare(labels_path, cases_path):
    sources = {c["record_id"]: c for c in sd.read_jsonl(cases_path)}
    labels = sd.read_jsonl(labels_path)
    rows = []
    for label in labels:
        if not all(abs(v) < sd.SCORE_MAX for v in label["S4"]):
            continue
        if len(set(label["S4"])) < 2:
            continue
        source = sources[label["record_id"]]
        for key in ("game_id", "moves", "stm", "ply"):
            if source[key] != label[key]:
                raise ValueError(f"source/label mismatch: {key}")
        if len(label["S4"]) != len(label["moves"]):
            raise ValueError("score/candidate length mismatch")
        if source["bucket"] != "primary" or any(source["child_results"]):
            raise ValueError("unexpected terminal or non-primary source")
        samples = [(ld.sparse(sd.child_board(source, m)), 1-source["stm"], 0, 0)
                   for m in source["moves"]]
        keys = {vf.canonical_position(s[0], s[1]) for s in samples}
        keys.add(vf.canonical_position(ld.sparse(source["board"]), source["stm"]))
        rows.append(dict(label=label, samples=samples, keys=keys))
    games = sorted({r["label"]["game_id"] for r in rows})
    random.Random(CONFIG["seed"]).shuffle(games)
    held_games = set(games[:max(1, round(len(games)*CONFIG["holdout_fraction"]))])
    held = [r for r in rows if r["label"]["game_id"] in held_games]
    held_keys = set().union(*(r["keys"] for r in held))
    leaking = {r["label"]["game_id"] for r in rows
               if r["label"]["game_id"] not in held_games and r["keys"] & held_keys}
    fitted = [r for r in rows if r["label"]["game_id"] not in held_games | leaking]
    if len(fitted) < 20 or len(held) < 10:
        raise ValueError("too few separated cases")
    assert not set().union(*(r["keys"] for r in fitted)) & held_keys
    manifest = {"eligible_parents": len(rows), "excluded_saturated_or_tied": len(labels)-len(rows),
                "excluded_overlap_games": sorted(leaking), "d4_separation_verified": True,
                "fit_ids": [r["label"]["record_id"] for r in fitted],
                "holdout_ids": [r["label"]["record_id"] for r in held],
                "fit_games": sorted({r["label"]["game_id"] for r in fitted}),
                "holdout_games": sorted(held_games)}
    return fitted, held, manifest


def summarize(rows, values):
    metrics, cursor = [], 0
    for row in rows:
        n = len(row["samples"])
        pred = [-v for v in values[cursor:cursor+n]]  # child -> parent perspective
        ref = row["label"]["S4"]
        metrics.append(dict(record_id=row["label"]["record_id"],
                            game_id=row["label"]["game_id"],
                            disagreement=sd.pair_disagreement(ref, pred),
                            top1=sd.top1_hit(ref, pred)))
        cursor += n
    if cursor != len(values):
        raise ValueError("prediction count mismatch")
    return {"parents": len(rows), "children": cursor,
            "pair_disagreement": statistics.mean(m["disagreement"] for m in metrics),
            "top1_tie_adjusted": statistics.mean(m["top1"] for m in metrics),
            "per_parent": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT/"data/m5/t23/full/cases.jsonl")
    parser.add_argument("--cases", type=Path, default=ROOT/"data/m5/t19/c_real/move_cases.jsonl")
    parser.add_argument("--probe", type=Path, default=ROOT/"build/engine/Release/nnue_probe.exe")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hidden-lr", type=float, default=CONFIG["hidden_lr"])
    parser.add_argument("--augment-d4", action="store_true")
    parser.add_argument("--steps", type=int, default=CONFIG["steps"])
    args = parser.parse_args()
    if not 0 < args.hidden_lr <= 1:
        parser.error("--hidden-lr must be in (0,1]")
    CONFIG["hidden_lr"] = args.hidden_lr
    if args.steps < 1:
        parser.error("--steps must be positive")
    CONFIG["steps"] = args.steps
    if args.augment_d4:
        CONFIG["augment_d4"] = True
    if args.out_dir.exists():
        raise ValueError("refusing to overwrite experiment")
    args.out_dir.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(CONFIG["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = train.resolve_device("auto")
    fitted, held, manifest = prepare(args.labels, args.cases)
    sources = {"labels": file_identity(args.labels), "cases": file_identity(args.cases),
               "probe": file_identity(args.probe), "script": file_identity(Path(__file__))}
    write_json(args.out_dir/"manifest.json", {**manifest, "sources": sources, "config": CONFIG})
    sets = {"fit": fitted, "holdout": held}
    samples = {k: [s for r in rows for s in r["samples"]] for k, rows in sets.items()}
    model = nnue.NNUEModel(seed=CONFIG["seed"]).to(device)
    initial = {k: summarize(rows, model_predictions(model.cpu(), samples[k])) for k, rows in sets.items()}
    model.to(device)
    baseline = {k: summarize(rows, [-v for r in rows for v in r["label"]["H"]])
                for k, rows in sets.items()}
    x, stm, _, _ = diagnose.tensors(samples["fit"], device)
    pairs, cursor = [], 0
    for row in fitted:
        ref = row["label"]["S4"]
        pairs.extend((cursor+i, cursor+j) for i in range(len(ref)) for j in range(len(ref))
                     if ref[i] > ref[j])
        cursor += len(ref)
    pair_tensor = torch.tensor(pairs, dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(CONFIG["seed"]+1)
    schedule = torch.randint(len(pairs), (CONFIG["steps"], CONFIG["batch_pairs"]),
                             generator=generator).to(device)
    transforms = torch.randint(8, (CONFIG["steps"], CONFIG["batch_pairs"]),
                               generator=torch.Generator().manual_seed(CONFIG["seed"]+2)).to(device)
    permutations = torch.tensor(features.symmetry_table(), device=device) if args.augment_d4 else None
    optimizer = train.make_optimizer(model, CONFIG["lr"], CONFIG["hidden_lr"])
    curve = []
    report = {"complete": False, "diagnostic_only": True, "config": CONFIG,
              "sources": sources, "split": manifest, "initial": initial, "static_H": baseline,
              "device": train.device_info(device), "pairs": len(pairs)}
    write_json(args.out_dir/"report.json", report)
    print(f"fit={len(fitted)} holdout={len(held)} pairs={len(pairs)}", flush=True)
    for step in range(CONFIG["steps"]):
        indices = pair_tensor[schedule[step]].flatten()
        inputs = x[indices]
        if args.augment_d4:
            # Use the same symmetry on both members of each ranked pair.
            gather = permutations[transforms[step]].repeat_interleave(2, dim=0)
            inputs = inputs.gather(1, gather)
        scores = -model(inputs, stm[indices]).reshape(-1, 2)
        loss = torch.nn.functional.softplus(scores[:, 1]-scores[:, 0]).mean()
        if not torch.isfinite(loss):
            raise ValueError("non-finite ranking loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step+1) % 1000 == 0:
            model.cpu()
            metrics = {k: summarize(rows, model_predictions(model, samples[k])) for k, rows in sets.items()}
            point = {"step": step+1, "loss": loss.item(),
                     **{k: {m: v for m, v in result.items() if m != "per_parent"}
                        for k, result in metrics.items()}}
            curve.append(point)
            print(json.dumps(point), flush=True)
            model.to(device)
    model.cpu()
    torch.save({"diagnostic_only": True, "stage": "24", "config": vars(model.config),
                "model_state": model.state_dict(), "sources": sources}, args.out_dir/"model.pt")
    final = {k: summarize(rows, model_predictions(model, samples[k])) for k, rows in sets.items()}
    weights = {k: getattr(model, k).detach().flatten().tolist() for k in ("W0", "b0", "W1", "b1", "W2", "b2")}
    weights.update(format_version=3, output_gain=model.config.output_gain)
    weight_path = args.out_dir/"diagnostic.nnue"
    weight_path.write_bytes(wf.pack_weights(weights))
    reloaded, meta = wf.read_weights(weight_path)
    for k in weights:
        if k in ("format_version", "output_gain"):
            continue
        p = getattr(model, k)
        with torch.no_grad():
            p.copy_(torch.tensor(reloaded[k]).reshape(p.shape))
    half = {k: summarize(rows, model_predictions(model, samples[k])) for k, rows in sets.items()}
    deployed = {k: summarize(rows, cpp_scores(args.probe, weight_path, samples[k], "scalar"))
                for k, rows in sets.items()}
    d = lambda metrics, k: metrics[k]["pair_disagreement"]
    checks = {"fit": d(final, "fit") <= CONFIG["fit_max"] and
              d(initial, "fit")-d(final, "fit") >= CONFIG["fit_reduction"],
              "float16_fidelity": all(d(half, k)-d(final, k) <= CONFIG["export_max_degradation"] for k in sets),
              "deployment_fidelity": all(d(deployed, k)-d(final, k) <= CONFIG["export_max_degradation"] for k in sets),
              "holdout_signal": d(deployed, "holdout") <= CONFIG["holdout_max"] and
              d(initial, "holdout")-d(deployed, "holdout") >= CONFIG["holdout_reduction"]}
    report.update(complete=True, curve=curve, final_float=final, float16=half, cpp=deployed,
                  checks=checks, weights_meta=meta, elapsed_seconds=time.monotonic()-started,
                  limitation="Historical validation data fitted; diagnostic weights cannot be promoted. No playing-strength claim.")
    write_json(args.out_dir/"report.json", report)
    print(json.dumps({"complete": True, "checks": checks, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
