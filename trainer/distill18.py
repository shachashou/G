"""Ticket 18 staged preparation and fixed-training-set preflight (no matches)."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import statistics
import sys
from types import SimpleNamespace

import binpack
import move_correlation as mc
import targets
import teacher_labels as tl
import validation_freeze as vf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import file_identity, source_identity, write_json

DEFAULT_REFERENCES = {
    "v1": "weights/nnue_v1.nnue", "small": "data/m5/m5_cmp_small_fidelity/candidate.nnue",
    "large": "data/m5/m5_cmp_large_fidelity/candidate.nnue", "opt_l": "data/m5/m5_opt_l_fidelity/candidate.nnue"}
HYPERPARAMETERS = {"epochs": 300, "batch_size": 1024, "max_steps": 1600, "checkpoint_policy": "last",
                   "seed": 42, "lr": 0.1, "hidden_lr": 0.001, "quant_forward": True,
                   "wdl_weight": 0, "quant_reg": 0, "score_clip": 0, "limit": None}
SOURCE_FILES = ("trainer/teacher_labels.py", "trainer/targets.py", "trainer/dataset.py", "trainer/nnue.py",
                "trainer/weights_format.py", "trainer/train.py", "trainer/export.py", "trainer/diagnose.py",
                "trainer/validation_freeze.py", "trainer/distill18.py", "trainer/distill18_fidelity.py",
                "trainer/move_correlation.py", "tools/distill18_summary.py", "engine/src/nnue_probe.cpp",
                "tools/run_distill18.ps1", "tools/distill18_screen.py", "engine/src/eval.cpp", "engine/src/nnue.cpp", "engine/src/nnue.h")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataset_args(directory: Path, arm="d"):
    frozen_path = directory / "frozen_validation.json"
    frozen = vf.load_frozen(frozen_path)
    target_path = directory / f"target_{arm}.json"
    return SimpleNamespace(data=frozen["data"]["path"], frozen_validation=frozen_path,
                           train_manifest=directory / "train_manifest.json", target_contract=target_path,
                           target=targets.load(target_path), limit=None, score_clip=0)


def describe_labels(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty label statistics")
    return {"records": len(values), "min": ordered[0], "max": ordered[-1],
            "mean": statistics.mean(values),
            "quantiles": {str(p): ordered[int((len(ordered)-1)*p)] for p in (0.05, 0.5, 0.9, 0.95, 0.99)}}


def prepare(args):
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "experiment_plan.json").exists():
        raise ValueError("experiment already prepared; use baselines/preflight/seal, do not overwrite")
    old = vf.load_frozen(args.parent_frozen)
    data = tl.verify_identity(old["data"])
    original_manifest = read(args.train_manifest)
    if original_manifest["frozen"]["sha256"] != file_identity(args.parent_frozen)["sha256"]:
        raise ValueError("parent train manifest/freeze mismatch")
    paths = [data, *(Path(e["path"]) for e in original_manifest["files"])]
    teacher_path = out / "labels" / "teacher_manifest.json"
    if not teacher_path.exists():
        teacher_path = tl.build(paths, args.probe, out / "labels", args.timeout)
    store = tl.LabelStore(teacher_path)
    if store.manifest["probe"]["sha256"] != file_identity(args.probe)["sha256"]:
        raise ValueError("cached labels use a different probe")
    tl.verify_identity(store.manifest["eval_source"])
    # Rescan, do not transplant the old manifest's exclusions or just edit its hash.
    with binpack.BinpackReader(data) as reader:
        game_map = reader.game_index_map()
        held = set(old["split"]["validation_games"])
        keys = vf.validation_keys(reader, game_map, held)
        validation_ids = vf.frozen_validation_record_ids(game_map, old)
    entries, labels, metadata = [], {"d": [], "s": []}, []
    for original in original_manifest["files"]:
        path = Path(original["path"]).resolve()
        values, mask, cache = store.for_source(path)
        with binpack.BinpackReader(path) as reader:
            game_map = reader.game_index_map()
            skip = held if cache["source"]["sha256"] == old["data"]["sha256"] else set()
            leaks = vf.scan_for_leaks(reader, game_map, keys, skip)
            excluded = {e["train_game"] for e in leaks} | skip
            allowed = set(original["games"]) - excluded
            entry = {"path": str(path), "source": file_identity(path), "games": sorted(allowed),
                     "excluded_games": sorted(excluded), "d4_equivalent_positions": len(leaks)}
            count, invalid = 0, 0
            for i, game in enumerate(game_map):
                if game not in allowed:
                    continue
                record = reader[i]
                if not mask[i]:
                    invalid += 1
                    continue
                labels["d"].append(values[i])
                labels["s"].append(record.score)
                metadata.append((record.stm, record.ply//10, abs(record.score) == 32767, record.wdl))
                count += 1
            entry.update(valid_records=count, terminal_records=invalid)
            entries.append(entry)
    gain = targets.choose_gain(max(abs(v) for values in labels.values() for v in values))
    pool_sha = targets.pool_identity(entries, store)
    contract_ids, baselines = {}, {}
    val_values, val_mask, _ = store.for_source(data)
    with binpack.BinpackReader(data) as reader:
        val_labels = {"d": [val_values[i] for i in validation_ids if val_mask[i]],
                      "s": [reader[i].score for i in validation_ids if val_mask[i]]}
    for arm, source in (("d", "handcrafted"), ("s", "search")):
        constants = targets.fit_constants(labels[arm], gain)
        contract = {"schema": 1, "name": f"{source}-linear", "label_source": source,
                    "perspective": "side-to-move", "units": "engine-score", "loss": "smooth-l1-beta1",
                    "format_version": 2, "output_gain": gain, "loss_normalizer": gain, "static_limit": 900000,
                    "teacher_manifest": file_identity(teacher_path), "train_pool_sha256": pool_sha,
                    "constants": constants, "label_profile": describe_labels(labels[arm])}
        contract["per_head"] = {str(h): describe_labels([v for v, m in zip(labels[arm], metadata) if m[0] == h]) for h in {m[0] for m in metadata}}
        contract["per_ply"] = {str(p): describe_labels([v for v, m in zip(labels[arm], metadata) if m[1] == p]) for p in {m[1] for m in metadata}}
        contract["saturated_search_buckets"] = {str(s): describe_labels([v for v, m in zip(labels[arm], metadata) if m[2] == s]) for s in {m[2] for m in metadata}}
        path = out / f"target_{arm}.json"
        write_json(path, contract)
        contract_ids[arm] = file_identity(path)
        baselines[arm] = targets.metrics([constants["mse_constant"]]*len(val_labels[arm]), val_labels[arm], gain, constants)
    prior_counts = [sum(m[3] == k for m in metadata) for k in range(3)]
    baselines["wdl_train_priors"] = [(n+1)/(sum(prior_counts)+3) for n in prior_counts]
    metric = mc.freeze_cases(data, old, args.probe, out / "move_cases.jsonl")
    write_json(out / "metric_contract.json", metric)
    frozen = copy.deepcopy(old)
    frozen.update(schema=2, parent_frozen_sha256=file_identity(args.parent_frozen)["sha256"],
                  split_identity={"parent": file_identity(args.parent_frozen), "validation_record_ids": validation_ids,
                                  "sha256": targets.digest({"data": old["data"]["sha256"], "games": sorted(held), "records": validation_ids})},
                  target_contracts=contract_ids, move_metric_contract=file_identity(out / "metric_contract.json"),
                  baselines=baselines, train_pool_sha256=pool_sha)
    write_json(out / "frozen_validation.json", frozen)
    write_json(out / "train_manifest.json", {"schema": 2, "frozen": file_identity(out / "frozen_validation.json"),
                                            "parent": file_identity(args.train_manifest), "files": entries,
                                            "valid_records": len(labels["d"]), "pool_sha256": pool_sha})
    references = dict(DEFAULT_REFERENCES)
    if args.reference:
        references = dict(item.split("=", 1) for item in args.reference)
    if len(references) != 4:
        raise ValueError("exactly four named historical references required")
    references = {name: file_identity(Path(path)) for name, path in references.items()}
    plan = {"schema": 1, "complete": True, "source": source_identity(), "probe": file_identity(args.probe),
            "code_files": [file_identity(ROOT / name) for name in SOURCE_FILES],
            "inputs": {name: file_identity(out / name) for name in ("frozen_validation.json", "train_manifest.json",
                         "target_d.json", "target_s.json", "metric_contract.json", "move_cases.jsonl")},
            "references": references, "hyperparameters": HYPERPARAMETERS,
            "screening_seed": 20261118, "screening_games": 20, "formal_games_authorised": False,
            "thresholds": metric["thresholds"], "bootstrap_seed": 20261118, "bootstrap_repeats": 10000}
    write_json(out / "experiment_plan.json", plan)
    print(f"prepared {len(labels['d'])} training records, gain={gain}, primary={metric['primary_positions']}/{len(validation_ids)}", flush=True)
    if getattr(args, "with_baselines", False):
        baselines_stage(args)
    return plan


def verify_plan(directory):
    plan = read(directory / "experiment_plan.json")
    if plan.get("schema") != 1 or plan.get("complete") is not True or plan["hyperparameters"] != HYPERPARAMETERS:
        raise ValueError("invalid or altered experiment plan")
    for identity in [plan["probe"], *plan["inputs"].values(), *plan["references"].values(), *plan["code_files"]]:
        tl.verify_identity(identity)
    metric = read(directory / "metric_contract.json")
    if plan["thresholds"] != metric["thresholds"] or plan["thresholds"] != {"min_top1": 0.20, "min_improvement": 0.10, "teacher_fraction": 0.50}:
        raise ValueError("predeclared thresholds changed")
    return plan, metric


def baselines_stage(args):
    out = args.out_dir.resolve()
    plan, metric = verify_plan(out)
    destination = out / "historical_baselines.json"
    if destination.exists():
        raise ValueError("historical baselines already exist")
    reports = {}
    for name, identity in {"handcrafted": None, **plan["references"]}.items():
        path = out / f"historical_{name}.json"
        reports[name] = mc.assess_cases(Path(plan["probe"]["path"]), Path(identity["path"]) if identity else None,
             "nnue" if identity else "handcrafted", "avx2", out / "move_cases.jsonl", out / "metric_contract.json", path, args.timeout)
        print(f"{name}: top1={reports[name]['results']['top1_tie_adjusted']:.6f}", flush=True)
    teacher = reports["handcrafted"]["results"]
    value = {"schema": 2, "complete": True, "plan": file_identity(out / "experiment_plan.json"),
             "reports": {name: file_identity(out / f"historical_{name}.json") for name in reports},
             "teacher_top1": teacher["top1_tie_adjusted"], "random_top1": teacher["random_top1"],
             "best_historical_top1": max(r["results"]["top1_tie_adjusted"] for n, r in reports.items() if n != "handcrafted"),
             "sensitivity_passed": teacher["top1_tie_adjusted"] >= .20 and teacher["top1_tie_adjusted"]-teacher["random_top1"] >= .10,
             "coverage_passed": metric["coverage_passed"]}
    write_json(destination, value)
    return value


def preflight(args):
    import torch
    import features
    import nnue
    import train
    import diagnose
    directory = args.out_dir.resolve()
    verify_plan(directory)
    result_dir = directory / "preflight"
    result_dir.mkdir(exist_ok=False)
    write_json(result_dir / "report.json", {"complete": False})
    options = dataset_args(directory)
    train_set, val_set, _, _ = train.build_datasets(options)
    try:
        parts = train_set.datasets
        pool = [(p, i) for p, part in enumerate(parts) for i in range(len(part))]
        rng = random.Random(42)
        for attempt in range(10):
            selected = rng.sample(pool, min(64, len(pool)))
            samples, ids = [], []
            for p, i in selected:
                part = parts[p]
                record = part.reader[part.indices[i]]
                samples.append((record.feature_indices(), record.stm, part.label(record), record.wdl))
                ids.append({"data": str(part.path), "record_id": record.index})
            if len({s[1] for s in samples}) == 2 and statistics.pvariance(s[2] for s in samples) > 0:
                break
        else:
            raise ValueError("cannot select non-degenerate two-head training sample")
    finally:
        train_set.close()
        val_set.close()
    torch.set_num_threads(1)
    device = train.resolve_device(args.device)
    x, stm, label, _ = diagnose.tensors(samples, device)
    constant = torch.mean(label)
    base_mse = float(((label-constant)/options.target["loss_normalizer"]).square().mean())
    arms = {}
    for name, quantized in (("float", False), ("ste", True)):
        model = nnue.NNUEModel(seed=42, config=nnue.ModelConfig(format_version=2, output_gain=options.target["output_gain"])).to(device)
        optimiser = train.make_optimizer(model, .1, .001)
        curve = []
        for step in range(1, 1001):
            prediction = model(x, stm, quantized=quantized)
            loss, _ = targets.losses(prediction, label, options.target)
            if not torch.isfinite(loss):
                raise ValueError("non-finite preflight loss")
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            if step == 1 or step % 100 == 0:
                with torch.no_grad():
                    values = model(x, stm, quantized=quantized).cpu().tolist()
                curve.append({"step": step, **targets.metrics(values, [s[2] for s in samples], options.target["loss_normalizer"])})
        arms[name] = {"curve": curve, "activations": diagnose.activation_stats(model, x, quantized),
                      "gradients": {n: {"l2": float(p.grad.norm()), "max": float(p.grad.abs().max())}
                                    for n, p in model.named_parameters() if p.grad is not None},
                      "mse_ratio": curve[-1]["normalised_mse"] / base_mse}
        write_json(result_dir / f"{name}.json", arms[name])
    report = {"complete": True, "plan": file_identity(directory / "experiment_plan.json"),
              "target": file_identity(options.target_contract), "samples": ids, "resampling_attempts": attempt,
              "device": str(device), "sample_baseline_mse": base_mse, "arms": arms,
              "passed": arms["ste"]["mse_ratio"] <= .25, "diagnostic_only": True}
    write_json(result_dir / "report.json", report)
    print(f"preflight passed={report['passed']}, STE MSE ratio={arms['ste']['mse_ratio']:.6f}")
    return report


def seal(args):
    directory = args.out_dir.resolve()
    _, metric = verify_plan(directory)
    baseline, pre = read(directory / "historical_baselines.json"), read(directory / "preflight/report.json")
    for identity in baseline["reports"].values():
        report = read(tl.verify_identity(identity))
        tl.verify_identity(report["per_case"])
    plan_sha = file_identity(directory / "experiment_plan.json")["sha256"]
    for report in (baseline, pre):
        if report.get("complete") is not True or report["plan"]["sha256"] != plan_sha:
            raise ValueError("stage A inputs incomplete or plan mismatch")
    passed = bool(metric["coverage_passed"] and baseline["sensitivity_passed"] and pre["passed"])
    write_json(directory / "stage_a.json", {"complete": True, "passed": passed,
              "plan": file_identity(directory / "experiment_plan.json"),
              "baselines": file_identity(directory / "historical_baselines.json"),
              "preflight": file_identity(directory / "preflight/report.json")})
    print(f"stage A passed={passed}; training {'eligible' if passed else 'blocked'}")
    return passed


def add_prepare_args(parser):
    parser.add_argument("--parent-frozen", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reference", action="append", help="four NAME=PATH references; defaults to tickets 08/15/17")
    parser.add_argument("--with-baselines", action="store_true")


def metric_audit(args):
    """Bounded read-only model audit before annotating the entire training pool."""
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = vf.load_frozen(args.parent_frozen)
    metric = mc.freeze_cases(Path(frozen["data"]["path"]), frozen, args.probe, out/"move_cases.jsonl")
    write_json(out/"metric_contract.json", metric)
    references = dict(DEFAULT_REFERENCES) if not args.reference else dict(item.split("=", 1) for item in args.reference)
    if len(references) != 4:
        raise ValueError("exactly four historical references required")
    reports = {}
    for name, path in {"handcrafted": None, **references}.items():
        reports[name] = mc.assess_cases(args.probe, Path(path) if path else None, "nnue" if path else "handcrafted",
            "avx2", out/"move_cases.jsonl", out/"metric_contract.json", out/f"{name}.json", args.timeout)
        print(f"{name}: top1={reports[name]['results']['top1_tie_adjusted']:.6f}", flush=True)
    h = reports["handcrafted"]["results"]
    sensitive = h["top1_tie_adjusted"] >= .2 and h["top1_tie_adjusted"]-h["random_top1"] >= .1
    result = {"complete": True, "coverage_passed": metric["coverage_passed"], "sensitivity_passed": sensitive,
              "passed": metric["coverage_passed"] and sensitive, "metric_contract": file_identity(out/"metric_contract.json"),
              "parent_frozen": file_identity(args.parent_frozen), "teacher_top1": h["top1_tie_adjusted"],
              "random_top1": h["random_top1"], "primary_positions": metric["primary_positions"],
              "primary_games": metric["primary_games"], "reports": {n: file_identity(out/f"{n}.json") for n in reports},
              "no_training_or_matches": True}
    write_json(out/"audit.json", result)
    return result


def assess(args):
    import export
    import diagnose
    out = args.out_dir.resolve()
    verify_plan(out)
    options = dataset_args(out, args.arm)
    checkpoint = ROOT / "weights" / f"m5_distill18_{args.arm}.pt"
    arm_dir = out / args.arm
    arm_dir.mkdir(exist_ok=True)
    export.main(["--checkpoint", str(checkpoint), "--data", options.data,
                 "--frozen-validation", str(options.frozen_validation), "--output", str(arm_dir/"candidate.nnue"),
                 "--report", str(arm_dir/"export_report.json"), "--no-calibrate"])
    return diagnose.main(["fidelity", "--checkpoint", str(checkpoint), "--weights", str(arm_dir/"candidate.nnue"),
                          "--export-report", str(arm_dir/"export_report.json"), "--data", options.data,
                          "--frozen-validation", str(options.frozen_validation), "--probe", str(ROOT/"build/engine/Release/nnue_probe.exe"),
                          "--all-validation", "--out-dir", str(arm_dir/"fidelity")])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "baselines", "preflight", "seal", "metric-audit", "assess"):
        command = sub.add_parser(mode)
        command.add_argument("--out-dir", type=Path, required=True)
        command.add_argument("--timeout", type=float, default=120)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        if mode == "prepare":
            add_prepare_args(command)
        if mode == "metric-audit":
            command.add_argument("--parent-frozen", type=Path, required=True)
            command.add_argument("--probe", type=Path, required=True)
            command.add_argument("--reference", action="append")
        if mode == "assess":
            command.add_argument("--arm", choices=("d", "s"), required=True)
    args = parser.parse_args(argv)
    result = {"prepare": prepare, "baselines": baselines_stage, "preflight": preflight, "seal": seal,
              "metric-audit": metric_audit, "assess": assess}[args.mode](args)
    return 2 if args.mode == "seal" and not result else 0


if __name__ == "__main__":
    sys.exit(main())
