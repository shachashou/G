"""Ticket 19 stage B: independent float/QAT fits on sealed clean parents.

This module never changes ticket-18 artifacts or starts later experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

import diagnose
import distill18
import features
import local_ranking_diagnosis as ld
import nnue
import targets
import teacher_labels as tl
import train
import weights_format as wf
from distill18_fidelity import model_predictions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from run_support import file_identity, source_identity, write_json

FIT_CONFIG = {"parents": 128, "per_head": 64, "steps": 1000, "batch_size": 64,
              "initial_seed": 42, "batch_seed": 42, "lr": .1, "hidden_lr": .001,
              "weight_decay": 1e-5, "output_gain": 128, "loss_normalizer": 128,
              "loss": "smooth-l1-beta1", "wdl_weight": 0, "quant_reg": 0,
              "d4": False, "checkpoint_policy": "last", "projection": "every-step-W64-b-int16",
              "mse_ratio_threshold": .25, "huber_ratio_threshold": .25, "smoke_test": False}
CODE_FILES = ("trainer/local_ranking_fit.py", "tools/local_ranking_summary.py", "tools/run_distill19_fit.ps1")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def master_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def schedule(config):
    generator = torch.Generator().manual_seed(config["batch_seed"])
    indices = torch.randint(config["parents"], (config["steps"], config["batch_size"]), generator=generator)
    return indices, hashlib.sha256(indices.numpy().tobytes()).hexdigest()


def select_parents(cases, config):
    selected, heads, games = [], {0: 0, 1: 0}, set()
    for case in cases:
        if case["split"] != "train_sample" or not case["eligible_for_followup_training"]:
            continue
        if case["eligible_for_followup_training"] != (not any(case["extended_overlap"].values())):
            raise ValueError("inconsistent extended D4 eligibility")
        head = case["stm"]
        if heads[head] >= config["per_head"]:
            continue
        if case["game_key"] in games:
            raise ValueError("fit parents must come from distinct complete games")
        selected.append(case)
        heads[head] += 1
        games.add(case["game_key"])
        if len(selected) == config["parents"]:
            break
    if heads != {0: config["per_head"], 1: config["per_head"]} or len(selected) != config["parents"]:
        raise ValueError("insufficient balanced clean parents; no automatic resampling")
    return selected


def prepare(analysis_dir: Path, fit_dir: Path, config=None):
    config = dict(FIT_CONFIG if config is None else config)
    if config != FIT_CONFIG and not (config.get("smoke_test") is True and config["steps"] <= 10 and config["parents"] <= 8):
        raise ValueError("stage B configuration differs from the predeclared contract")
    if fit_dir.exists():
        raise ValueError("refusing to overwrite fit directory")
    report = read(analysis_dir/"report.json")
    if not report.get("complete") or report.get("training_performed") is not False:
        raise ValueError("stage A is incomplete or is not a read-only diagnosis")
    tl.verify_identity(report["script"])
    for identity in report["inputs"].values():
        tl.verify_identity(identity)
    for identity in report["artifacts"].values():
        tl.verify_identity(identity)
    cases_path = tl.verify_identity(report["cases"])
    tl.verify_identity(report["sample_manifest"])
    base = Path(report["inputs"]["plan18"]["path"]).parent
    distill18.verify_plan(base)
    if any(not v["exact"] or v["static_limit_hits"] for v in report["parity"].values()):
        raise ValueError("stage A parity/static range precondition failed")
    cases = rows(cases_path)
    selected = select_parents(cases, config)
    for c in selected:
        if any(c["extended_overlap"].values()):
            raise ValueError("selected source game has an extended D4 overlap")
    if report["output_gain"] != config["output_gain"]:
        raise ValueError("stage A gain differs from stage B")
    labels = [c["teacher_parent"] for c in selected]
    constants = targets.fit_constants(labels, config["loss_normalizer"])
    baseline = targets.metrics([constants["mse_constant"]]*len(labels), labels, config["loss_normalizer"], constants)
    if baseline["baseline_mse"] <= 0 or baseline["baseline_huber"] <= 0:
        raise ValueError("degenerate selected labels; no automatic redraw")
    model = nnue.NNUEModel(seed=config["initial_seed"], config=nnue.ModelConfig(format_version=2, output_gain=config["output_gain"]))
    _, batch_hash = schedule(config)
    fit_dir.mkdir(parents=True)
    manifest = {"schema": 1, "complete": True, "stage": "B", "diagnostic_only": True,
                "source": source_identity(), "config": config, "analysis": file_identity(analysis_dir/"report.json"),
                "cases": report["cases"], "children": report["artifacts"]["children"],
                "plan18": report["inputs"]["plan18"], "selected_parents": selected,
                "constants": constants, "training_baseline": baseline,
                "initial_master_sha256": master_hash(model), "batch_schedule_sha256": batch_hash,
                "code_files": [file_identity(ROOT/name) for name in CODE_FILES]}
    write_json(fit_dir/"fit_manifest.json", manifest)
    print(f"stage B frozen: {len(selected)} clean parents, {config['per_head']}/head, two independent {config['steps']}-step fits")
    return manifest


def load_manifest(path):
    manifest = read(path)
    if manifest.get("schema") != 1 or not manifest.get("complete") or manifest.get("stage") != "B":
        raise ValueError("invalid stage B manifest")
    for identity in [manifest["analysis"], manifest["cases"], manifest["children"], manifest["plan18"], *manifest["code_files"]]:
        tl.verify_identity(identity)
    distill18.verify_plan(Path(manifest["plan18"]["path"]).parent)
    expected = select_parents(rows(manifest["cases"]["path"]), manifest["config"])
    if expected != manifest["selected_parents"]:
        raise ValueError("fit sample no longer matches the sealed deterministic selection")
    if manifest["config"] != FIT_CONFIG and not manifest["config"].get("smoke_test"):
        raise ValueError("unrecognised official stage B hyperparameters")
    return manifest


@torch.no_grad()
def project_master(model):
    clipped = torch.zeros((), device=model.W0.device, dtype=torch.int64)
    for name in ("W0", "b0", "W1", "b1", "W2", "b2"):
        value = getattr(model, name)
        lo, hi = (-wf.WEIGHT_CLAMP, wf.WEIGHT_CLAMP) if name.startswith("W") else (wf.INT16_MIN, wf.INT16_MAX)
        clipped += ((value < lo) | (value > hi)).sum()
        value.clamp_(lo, hi)
    return int(clipped.item())


def evaluate_fit(model, quantized, manifest):
    cases = rows(manifest["cases"]["path"])
    child_rows = rows(manifest["children"]["path"])
    selected_keys = {c["key"] for c in manifest["selected_parents"]}
    # Parent and child orders exactly follow stage A's frozen JSONLs.
    samples = [(ld.sparse(c["board"]), c["stm"], c["teacher_parent"], 0) for c in cases]
    teacher_children = defaultdict_list(child_rows)
    primary = [c for c in cases if c["bucket"] == "primary"]
    for case in primary:
        children = teacher_children[case["key"]]
        if len(children) != len(case["moves"]) or [r["key"] for r in children] != [f"{case['key']}:{m}" for m in case["moves"]]:
            raise ValueError("stage A child order/count mismatch")
        samples.extend((ld.sparse(b), 1-case["stm"], r["teacher"], 0) for b, r in zip(ld.child_boards(case), children))
    model = model.cpu()
    values = model_predictions(model, samples, quantized)
    parent_predictions = {c["key"]: v for c, v in zip(cases, values[:len(cases)])}
    local, cursor = [], len(cases)
    for case in primary:
        n = len(case["moves"])
        labels = [r["teacher"] for r in teacher_children[case["key"]]]
        stat = ld.local_scores(labels, values[cursor:cursor+n], case["moves"].index(case["teacher_move"]))
        local.append({"key": case["key"], "split": case["split"], "local": {"trained_forward": stat}})
        cursor += n
    if cursor != len(values):
        raise ValueError("evaluation coverage mismatch")
    normalizer = manifest["config"]["loss_normalizer"]
    constants = manifest["constants"]
    selected = manifest["selected_parents"]
    fit_values = [parent_predictions[c["key"]] for c in selected]
    fit_labels = [c["teacher_parent"] for c in selected]
    fit_metric = targets.metrics(fit_values, fit_labels, normalizer, constants)
    fit_heads = {str(h): targets.metrics([v for v, c in zip(fit_values, selected) if c["stm"] == h],
                                         [c["teacher_parent"] for c in selected if c["stm"] == h], normalizer, constants) for h in (0, 1)}
    val = [c for c in cases if c["split"] == "validation"]
    val_metric = targets.metrics([parent_predictions[c["key"]] for c in val], [c["teacher_parent"] for c in val], normalizer, constants)
    val_heads = {str(h): targets.metrics([parent_predictions[c["key"]] for c in val if c["stm"] == h],
                                         [c["teacher_parent"] for c in val if c["stm"] == h], normalizer, constants)
                 for h in {c["stm"] for c in val}}
    train_local = [r for r in local if r["key"] in selected_keys]
    val_local = [r for r in local if r["split"] == "validation"]
    return {"fit_parents": fit_metric, "fit_per_head": fit_heads, "validation_parents": val_metric, "validation_per_head": val_heads,
            "fit_parent_local": ld.local_summary(train_local), "validation_local": ld.local_summary(val_local),
            "eval_positions": len(samples), "fit_predictions": fit_values,
            "static_limit_hits": sum(abs(v) >= wf.STATIC_LIMIT for v in values)}


def defaultdict_list(child_rows):
    result = {}
    for row in child_rows:
        result.setdefault(row["parent_key"], []).append(row)
    return result


def fit(manifest_path: Path, arm: str, out: Path, requested_device="auto"):
    if arm not in ("float", "qat") or out.exists():
        raise ValueError("unknown arm or refusing to overwrite fit output")
    manifest = load_manifest(manifest_path)
    config = manifest["config"]
    torch.set_num_threads(1)
    torch.manual_seed(config["initial_seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = train.resolve_device(requested_device)
    model = nnue.NNUEModel(seed=config["initial_seed"], config=nnue.ModelConfig(format_version=2, output_gain=config["output_gain"]))
    initial = master_hash(model)
    batches, batch_hash = schedule(config)
    if initial != manifest["initial_master_sha256"] or batch_hash != manifest["batch_schedule_sha256"]:
        raise ValueError("initialisation or batch schedule differs from sealed plan")
    model.to(device)
    selected = manifest["selected_parents"]
    samples = [(ld.sparse(c["board"]), c["stm"], c["teacher_parent"], 0) for c in selected]
    x, stm, labels, _ = diagnose.tensors(samples, device)
    optimiser = train.make_optimizer(model, config["lr"], config["hidden_lr"])
    quantized = arm == "qat"
    contract = {"loss_normalizer": config["loss_normalizer"]}
    out.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema": 1, "complete": False, "stage": "B", "diagnostic_only": True, "arm": arm,
              "trained_quantized": quantized, "fit_manifest": file_identity(manifest_path), "config": config,
              "initial_master_sha256": initial, "batch_schedule_sha256": batch_hash,
              "from_scratch": True, "device": train.device_info(device), "tf32": False}
    write_json(out/"report.json", report)
    curve, projection_hits, static_hits = [], 0, 0
    try:
        for step, indices in enumerate(batches, 1):
            indices = indices.to(device)
            prediction = model(x[indices], stm[indices], quantized=quantized)
            loss, _ = targets.losses(prediction, labels[indices], contract)
            if not torch.isfinite(loss):
                raise ValueError("non-finite local fit loss")
            static_hits += int((prediction.detach().abs() >= wf.STATIC_LIMIT).sum().item())
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            projection_hits += project_master(model)
            if step == 1 or step % 100 == 0 or step == config["steps"]:
                with torch.no_grad():
                    p = model(x, stm, quantized=quantized).cpu().tolist()
                metric = targets.metrics(p, [s[2] for s in samples], config["loss_normalizer"], manifest["constants"])
                curve.append({"step": step, **metric})
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError("non-finite fit gradient")
        gradients = {n: {"l2": float(p.grad.norm()), "max": float(p.grad.abs().max())}
                     for n, p in model.named_parameters() if p.grad is not None}
        activations = diagnose.activation_stats(model, x, quantized)
        bounds = {name: {"min": float(p.detach().min()), "max": float(p.detach().max()),
                        "at_lower_fraction": float((p <= (-64 if name.startswith("W") else wf.INT16_MIN)).float().mean()),
                        "at_upper_fraction": float((p >= (64 if name.startswith("W") else wf.INT16_MAX)).float().mean())}
                  for name, p in model.named_parameters() if name != "wdl_bias"}
        evaluation = evaluate_fit(model, quantized, manifest)
        if quantized and (evaluation["fit_parents"]["normalised_mse"] != curve[-1]["normalised_mse"]):
            raise ValueError("QAT fitting results differ between training device and CPU evaluation")
        checks = {"mse_ratio": evaluation["fit_parents"]["mse_ratio"] <= config["mse_ratio_threshold"],
                  "huber_ratio": evaluation["fit_parents"]["huber_ratio"] <= config["huber_ratio_threshold"],
                  "training_static_range": static_hits == 0,
                  "fit_static_range": evaluation["fit_parents"]["static_limit_hits"] == 0,
                  "evaluation_static_range": evaluation["static_limit_hits"] == 0,
                  "both_heads": {c["stm"] for c in selected} == {0, 1}}
        # A tiny-fit success is never a strength or deployment gate.
        torch.save({"diagnostic_only": True, "stage": "19-B", "arm": arm,
                    "config": vars(model.config), "model_state": model.state_dict(),
                    "fit_manifest_identity": file_identity(manifest_path), "initial_master_sha256": initial,
                    "batch_schedule_sha256": batch_hash, "optimizer_steps": config["steps"],
                    "samples_exposed": config["steps"]*config["batch_size"], "quant_forward": quantized}, out/"model.pt")
        report.update(complete=True, optimizer_steps=config["steps"], samples_exposed=config["steps"]*config["batch_size"],
                      final_master_sha256=master_hash(model),
                      checkpoint=file_identity(out/"model.pt"), curve=curve, gradients=gradients, activations=activations,
                      parameter_bounds=bounds,
                      master_projection_element_hits=projection_hits, training_static_limit_hits=static_hits,
                      evaluation=evaluation, fit_gate={"passed": all(checks.values()), "checks": checks},
                      elapsed_seconds=time.monotonic()-started)
        write_json(out/"report.json", report)
        print(json.dumps({"complete": True, "arm": arm, "fit_gate": report["fit_gate"],
                          "fit_mse_ratio": evaluation["fit_parents"]["mse_ratio"],
                          "fit_huber_ratio": evaluation["fit_parents"]["huber_ratio"],
                          "elapsed_seconds": report["elapsed_seconds"]}))
        return report
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic()-started)
        write_json(out/"report.json", report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare-b")
    prep.add_argument("--analysis-dir", type=Path, default=ROOT/"data/m5/distill19/analysis_initial")
    prep.add_argument("--out-dir", type=Path, default=ROOT/"data/m5/distill19/fit_b")
    command = sub.add_parser("fit-b")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--arm", choices=("float", "qat"), required=True)
    command.add_argument("--out-dir", type=Path, required=True)
    command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.mode == "prepare-b":
        prepare(args.analysis_dir.resolve(), args.out_dir.resolve())
    else:
        fit(args.manifest.resolve(), args.arm, args.out_dir.resolve(), args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
