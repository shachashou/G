"""Ticket 19 stage C: independent float/QAT fits on child positions.

Self-contained on the NEW (m5_round1) foundation: builds teacher labels via the
C++ handcrafted evaluator, freezes validation candidate cases for the extended
D4 audit, samples primary parents from the clean pool, and trains two arms on
their legal non-terminal children with the SAME pointwise Huber loss. Never
touches ticket-18 artifacts or starts matches.
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

import binpack
import diagnose
import local_ranking_diagnosis as ld
import move_correlation as mc
import nnue
import targets
import teacher_labels as tl
import train
import validation_freeze as vf
import weights_format as wf
from distill18_fidelity import model_predictions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import file_identity, source_identity, write_json

CONFIG = {"parents": 64, "per_head": 32, "sample_parents": 256, "steps": 1000,
          "batch_size": 64, "initial_seed": 42, "batch_seed": 42, "lr": .1,
          "hidden_lr": .001, "weight_decay": 1e-5, "loss": "smooth-l1-beta1",
          "wdl_weight": 0, "quant_reg": 0, "d4": False, "checkpoint_policy": "last",
          "projection": "every-step-W64-b-int16", "mse_ratio_threshold": .25,
          "huber_ratio_threshold": .25, "excess_threshold": .75, "disagreement_threshold": .10,
          "sample_seed": 20261119, "smoke_test": False}


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


def project_master(model):
    clipped = 0
    for name in ("W0", "b0", "W1", "b1", "W2", "b2"):
        value = getattr(model, name)
        lo, hi = (-wf.WEIGHT_CLAMP, wf.WEIGHT_CLAMP) if name.startswith("W") else (wf.INT16_MIN, wf.INT16_MAX)
        clipped += int(((value < lo) | (value > hi)).sum().item())
        value.data.clamp_(lo, hi)
    return clipped


def select_primary(cases, config):
    selected, heads, games = [], {0: 0, 1: 0}, set()
    for case in cases:
        if case["split"] != "train_sample" or not case["eligible_for_followup_training"]:
            continue
        if case["bucket"] != "primary":
            continue
        head = case["stm"]
        if heads[head] >= config["per_head"]:
            continue
        if case["game_key"] in games:
            raise ValueError("C parents must come from distinct complete games")
        selected.append(case)
        heads[head] += 1
        games.add(case["game_key"])
        if len(selected) == config["parents"]:
            break
    if heads != {0: config["per_head"], 1: config["per_head"]} or len(selected) != config["parents"]:
        raise ValueError("insufficient balanced clean primary parents; no automatic resampling")
    return selected


def child_samples(selected, store, probe, timeout):
    """For each selected primary parent, freeze all legal non-terminal children
    and their C++ handcrafted H label (own-stm perspective)."""
    per_parent = []
    boards, stms = [], []
    meta = []
    for case in selected:
        child_boards = list(ld.child_boards(case))
        for move, board in zip(case["moves"], child_boards):
            boards.append(board)
            stms.append(1 - case["stm"])
            meta.append((case["key"], move))
    teacher = ld.evaluate(probe, boards, stms, timeout=timeout)
    cursor = 0
    for case in selected:
        n = len(case["moves"])
        entries = []
        for move, board, h in zip(case["moves"], boards[cursor:cursor+n], teacher[cursor:cursor+n]):
            entries.append({"move": move, "board": board, "stm": 1 - case["stm"],
                            "teacher": h, "sparse": ld.sparse(board)})
        per_parent.append(entries)
        cursor += n
    return per_parent


def prepare(args):
    if args.out_dir.exists():
        raise ValueError("refusing to overwrite C output directory")
    args.out_dir.mkdir(parents=True)
    data = Path(args.data).resolve()
    frozen = vf.load_frozen(args.frozen_validation)
    vf.verify_frozen_data(frozen, data)
    probe = Path(args.probe).resolve()
    store = tl.LabelStore(args.teacher_manifest)
    manifest = read(args.train_manifest)
    # Build train cases from the schema-1 manifest + teacher labels.
    sources = ld.select_training_sources(manifest, store, CONFIG["sample_parents"], CONFIG["sample_seed"])
    train_cases = ld.make_cases(sources, manifest, store, probe, args.timeout)
    # Validation candidate cases for the extended D4 audit.
    metric = mc.freeze_cases(data, frozen, probe, args.out_dir / "move_cases.jsonl")
    write_json(args.out_dir / "metric_contract.json", metric)
    val_cases = [{"board": c["board"], "stm": c["stm"], "bucket": c["bucket"],
                  "moves": c["moves"], "key": c["record_id"]} for c in rows(args.out_dir / "move_cases.jsonl")]
    overlap = ld.extended_overlap(train_cases, val_cases)
    for case in train_cases:
        case["split"] = "train_sample"
    selected = select_primary(train_cases, CONFIG)
    if any(v for case in selected for v in case["extended_overlap"].values()):
        raise ValueError("selected C parent has an extended D4 overlap")
    per_parent = child_samples(selected, store, probe, args.timeout)
    # Sealed child pool: uniform parent -> uniform child sampling surface.
    flat = [e for entries in per_parent for e in entries]
    child_labels = [e["teacher"] for e in flat]
    if len(child_labels) < args.parents:
        raise ValueError("too few child positions for a meaningful fit")
    gain = targets.choose_gain(max(abs(v) for v in child_labels))
    constants = targets.fit_constants(child_labels, gain)
    baseline = targets.metrics([constants["mse_constant"]] * len(child_labels), child_labels, gain, constants)
    if baseline["baseline_mse"] <= 0 or baseline["baseline_huber"] <= 0:
        raise ValueError("degenerate child labels; no automatic redraw")
    manifest_out = {"schema": 1, "complete": True, "stage": "C", "diagnostic_only": True,
                    "source": source_identity(), "config": {**CONFIG, "output_gain": gain,
                    "loss_normalizer": gain, "smoke_test": args.smoke_test},
                    "data": file_identity(data), "frozen": file_identity(args.frozen_validation),
                    "train_manifest": file_identity(args.train_manifest),
                    "teacher_manifest": file_identity(args.teacher_manifest),
                    "probe": file_identity(probe), "metric_contract": file_identity(args.out_dir / "metric_contract.json"),
                    "selected_parents": selected, "children": per_parent,
                    "child_labels": child_labels, "constants": constants, "training_baseline": baseline,
                    "extended_d4": overlap}
    write_json(args.out_dir / "c_manifest.json", manifest_out)
    print(json.dumps({"complete": True, "parents": len(selected), "children": len(flat),
                      "gain": gain, "primary_validation": metric["primary_positions"],
                      "extended_d4": overlap["counts"]}, ensure_ascii=False))
    return manifest_out


def schedule(config, per_parent):
    rng = random.Random(config["batch_seed"])
    offsets = []
    cursor = 0
    for entries in per_parent:
        offsets.append(cursor)
        cursor += len(entries)
    batches = []
    for _ in range(config["steps"]):
        indices = []
        for _ in range(config["batch_size"]):
            p = rng.randrange(len(per_parent))
            c = rng.randrange(len(per_parent[p]))
            indices.append(offsets[p] + c)
        batches.append(indices)
    return batches


def fit(manifest_path, arm, out, requested_device="auto"):
    if arm not in ("float", "qat") or out.exists():
        raise ValueError("unknown arm or refusing to overwrite C fit output")
    manifest = read(manifest_path)
    if manifest.get("stage") != "C" or not manifest.get("complete"):
        raise ValueError("invalid C manifest")
    config = manifest["config"]
    torch.set_num_threads(1)
    torch.manual_seed(config["initial_seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = train.resolve_device(requested_device)
    model = nnue.NNUEModel(seed=config["initial_seed"], config=nnue.ModelConfig(format_version=2, output_gain=config["output_gain"]))
    model.to(device)
    # Build tensors for the whole child pool once.
    per_parent = manifest["children"]
    flat = [e for entries in per_parent for e in entries]
    samples = [(e["sparse"], e["stm"], e["teacher"], 0) for e in flat]
    x, stm, labels, _ = diagnose.tensors(samples, device)
    batches = schedule(config, per_parent)
    batch_t = torch.tensor(batches, dtype=torch.long, device=device)
    optimiser = train.make_optimizer(model, config["lr"], config["hidden_lr"])
    quantized = arm == "qat"
    contract = {"loss_normalizer": config["loss_normalizer"]}
    out.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema": 1, "complete": False, "stage": "C", "diagnostic_only": True, "arm": arm,
              "trained_quantized": quantized, "manifest": file_identity(manifest_path),
              "from_scratch": True, "device": train.device_info(device), "tf32": False}
    write_json(out / "report.json", report)
    curve, projection_hits, static_hits = [], 0, 0
    try:
        for step in range(config["steps"]):
            indices = batch_t[step]
            prediction = model(x[indices], stm[indices], quantized=quantized)
            loss, _ = targets.losses(prediction, labels[indices], contract)
            if not torch.isfinite(loss):
                raise ValueError("non-finite child fit loss")
            static_hits += int((prediction.detach().abs() >= wf.STATIC_LIMIT).sum().item())
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            projection_hits += project_master(model)
            if step == 0 or (step + 1) % 100 == 0 or step == config["steps"] - 1:
                with torch.no_grad():
                    p = model(x, stm, quantized=quantized).cpu().tolist()
                metric = targets.metrics(p, [s[2] for s in samples], config["loss_normalizer"], manifest["constants"])
                curve.append({"step": step + 1, **metric})
        evaluation = evaluate_child(model, quantized, manifest)
        if quantized and evaluation["child_fit"]["normalised_mse"] != curve[-1]["normalised_mse"]:
            raise ValueError("QAT child fit differs between training device and CPU evaluation")
        checks = {"child_mse_ratio": evaluation["child_fit"]["mse_ratio"] <= config["mse_ratio_threshold"],
                  "child_huber_ratio": evaluation["child_fit"]["huber_ratio"] <= config["huber_ratio_threshold"],
                  "teacher_best_excess": evaluation["local"]["teacher_best_excess"] >= config["excess_threshold"],
                  "pair_disagreement": evaluation["local"]["macro_pair_disagreement"] <= config["disagreement_threshold"],
                  "training_static_range": static_hits == 0}
        torch.save({"diagnostic_only": True, "stage": "19-C", "arm": arm,
                    "config": vars(model.config), "model_state": model.state_dict(),
                    "manifest_identity": file_identity(manifest_path), "quant_forward": quantized}, out / "model.pt")
        report.update(complete=True, optimizer_steps=config["steps"],
                      samples_exposed=config["steps"] * config["batch_size"],
                      curve=curve, evaluation=evaluation,
                      master_projection_element_hits=projection_hits, training_static_limit_hits=static_hits,
                      c_gate={"passed": all(checks.values()), "checks": checks},
                      elapsed_seconds=time.monotonic() - started)
        write_json(out / "report.json", report)
        print(json.dumps({"complete": True, "arm": arm, "c_gate": report["c_gate"],
                          "child_mse_ratio": evaluation["child_fit"]["mse_ratio"],
                          "child_huber_ratio": evaluation["child_fit"]["huber_ratio"],
                          "teacher_best_excess": evaluation["local"]["teacher_best_excess"],
                          "pair_disagreement": evaluation["local"]["macro_pair_disagreement"],
                          "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))
        return report
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
        write_json(out / "report.json", report)
        raise


@torch.no_grad()
def evaluate_child(model, quantized, manifest):
    config = manifest["config"]
    per_parent = manifest["children"]
    flat = [e for entries in per_parent for e in entries]
    samples = [(e["sparse"], e["stm"], e["teacher"], 0) for e in flat]
    model = model.cpu()
    values = model_predictions(model, samples, quantized)
    # Child point-value fit vs the same-sample child constants.
    child_fit = targets.metrics(values, [e["teacher"] for e in flat], config["loss_normalizer"], manifest["constants"])
    # Local ranking on each selected parent (negate both child H and prediction
    # into the parent's move perspective, exactly as ld.local_scores does).
    local_rows = []
    cursor = 0
    for case, entries in zip(manifest["selected_parents"], per_parent):
        n = len(entries)
        h = [e["teacher"] for e in entries]
        preds = values[cursor:cursor + n]
        stat = ld.local_scores(h, preds, case["moves"].index(case["teacher_move"]))
        local_rows.append(stat)
        cursor += n
    local = ld.local_summary([{"local": {"trained_forward": r}} for r in local_rows])["trained_forward"]
    return {"child_fit": child_fit, "local": local, "eval_positions": len(flat),
            "static_limit_hits": sum(abs(v) >= wf.STATIC_LIMIT for v in values)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare-c")
    prep.add_argument("--data", type=Path, required=True)
    prep.add_argument("--frozen-validation", type=Path, required=True)
    prep.add_argument("--train-manifest", type=Path, required=True)
    prep.add_argument("--teacher-manifest", type=Path, required=True)
    prep.add_argument("--probe", type=Path, required=True)
    prep.add_argument("--out-dir", type=Path, required=True)
    prep.add_argument("--parents", type=int, default=64)
    prep.add_argument("--sample-parents", type=int, default=256)
    prep.add_argument("--sample-seed", type=int, default=20261119)
    prep.add_argument("--timeout", type=float, default=120)
    prep.add_argument("--smoke-test", action="store_true")
    command = sub.add_parser("fit-c")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--arm", choices=("float", "qat"), required=True)
    command.add_argument("--out-dir", type=Path, required=True)
    command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.mode == "prepare-c":
        if args.parents < 2 or args.parents % 2:
            parser.error("--parents must be an even integer >= 2")
        CONFIG["parents"] = args.parents
        CONFIG["per_head"] = args.parents // 2
        CONFIG["sample_parents"] = args.sample_parents
        CONFIG["sample_seed"] = args.sample_seed
        if args.smoke_test:
            CONFIG.update(smoke_test=True, steps=10)
        prepare(args)
    else:
        fit(args.manifest.resolve(), args.arm, args.out_dir.resolve(), args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
