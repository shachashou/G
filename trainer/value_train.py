#!/usr/bin/env python3
"""Train the new value network from committed pool samples and export it (ticket 06).

Closed loop implemented here:

    SamplePool (frozen train/validation split)
      -> black-absolute targets + 450 absolute-colour features
      -> 450-1024-64-32-1, single output, MSE or SmoothL1
      -> checkpoint (model + optimizer + step + data identity + config)
      -> GMVALUE1 quantised model + fidelity report against frozen tolerances

Checkpoints are resumable and record the source identity; the exporter also
compares the training model, the quantised model (integer weights dequantised
in float) and the dependency-free integer reference forward.

CLI::

    python trainer/value_train.py --pool data/pool/pool.sqlite3 \\
        --output-dir weights/value-run1 --steps 200 --batch-size 32 \\
        --lr 1e-3 --loss mse --seed 1

This proves the pipeline runs and its numbers are sane; it is not a strength
promotion claim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import torch

import value_data
import value_format as vf
from value_model import ValueConfig, ValueNet, value_loss

CHECKPOINT_FORMAT = "GMVALUE-CHECKPOINT1"
ARCH = [vf.N_FEATURES, vf.N_ACC, vf.N_H1, vf.N_H2, 1]


@dataclass
class TrainConfig:
    pool: str
    output_dir: str
    seed: int = 1
    steps: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    loss: str = "mse"
    smooth_l1_beta: float = 1.0
    weight_decay: float = 0.0
    device: str = "auto"
    val_every: int = 20
    checkpoint_every: int = 50
    checkpoint_min_interval_s: float = 60.0
    log_every: int = 10
    resume: str | None = None
    ea: int = vf.EA_DEFAULT
    act_max: float = vf.ACT_MAX_DEFAULT
    export: bool = True
    max_minutes: float | None = None
    initial_model: str | None = None
    lineage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name in ("cpu", "cuda"):
        return name
    raise ValueError(f"unknown device {name!r}")


def model_id(model: ValueNet) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:32]


def evaluate(model: ValueNet, samples, device: str, batch_size: int,
             loss_kind: str, beta: float) -> dict:
    model.eval()
    total_loss = 0.0
    total_abs = 0.0
    count = 0
    total_weight = 0.0
    with torch.no_grad():
        for offset in range(0, len(samples), batch_size):
            chosen = samples[offset:offset + batch_size]
            x, y = value_data.tensors(chosen)
            x, y = x.to(device), y.to(device)
            weights = torch.tensor([s.weight for s in chosen], device=device)
            prediction = model(x)
            loss = value_loss(prediction, y, loss_kind, beta, weights)
            total_loss += float(loss) * float(weights.sum())
            total_weight += float(weights.sum())
            total_abs += float(torch.abs(prediction - y).sum())
            count += x.shape[0]
    model.train()
    return {"loss": total_loss / max(1e-12, total_weight),
            "mae": total_abs / max(1, count),
            "samples": count}


def _checkpoint_payload(*, model: ValueNet, optimizer,
                     step: int, config: TrainConfig, identity: dict,
                     history: list[dict], device: str, training_positions: dict) -> dict:
    return {
        "format": CHECKPOINT_FORMAT,
        "arch": ARCH,
        "rule": "renju",
        "feature_spec": "absolute-color-450",
        "value_contract": "black-absolute-tanh-v1",
        "config": config.to_dict(),
        "data_identity": identity,
        "training_positions": training_positions,
        "lineage": config.lineage,
        "step": int(step),
        "device": device,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history[-500:],
        "created_at": _now(),
    }


def _save_checkpoint(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(**values), path)


def _load_checkpoint(path: str | Path, config: TrainConfig) -> dict:
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or blob.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not a value-network checkpoint")
    for key, expected in (("arch", ARCH), ("rule", "renju"),
                          ("feature_spec", "absolute-color-450"),
                          ("value_contract", "black-absolute-tanh-v1")):
        if blob.get(key) != expected:
            raise ValueError(
                f"checkpoint {key} {blob.get(key)!r} does not match {expected!r}")
    return blob


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[index]


def _diff_metrics(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right) or not left:
        raise ValueError("cannot compare empty or mismatched value lists")
    diffs = [abs(a - b) for a, b in zip(left, right)]
    return {"count": len(diffs),
            "mean_abs": sum(diffs) / len(diffs),
            "p95_abs": _p95(diffs),
            "max_abs": max(diffs)}


def _tolerance_pass(metrics: dict, tolerance: dict) -> bool:
    return all(metrics[key] <= limit for key, limit in tolerance.items())


def export_model(model: ValueNet, bundle: value_data.DataBundle,
                 config: TrainConfig, *, steps: int,
                 train_metrics: dict, val_metrics: dict,
                 output_dir: str | Path) -> dict:
    """Writes the quantised model and a fidelity report."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = 1 << config.ea
    quantized = vf.quantize_parameters(model.flat_parameters(),
                                       ea=config.ea, act_max=config.act_max)
    # Freeze the positions actually used by this export, rather than relying
    # on the mutable pool remaining available when a candidate is evaluated.
    exclusion_path = out_dir / "training_positions.json"
    exclusion_path.write_text(json.dumps(value_data.training_positions(bundle),
                                         sort_keys=True), encoding="utf-8")
    exclusion_sha = hashlib.sha256(exclusion_path.read_bytes()).hexdigest()
    metadata = {
        "model_id": model_id(model),
        "arch": ARCH,
        "rule": "renju",
        "feature_spec": "absolute-color-450",
        "value_contract": "black-absolute-tanh-v1",
        "label_versions": bundle.identity.get("label_versions", []),
        "training": {
            "seed": config.seed,
            "steps": int(steps),
            "batch_size": config.batch_size,
            "lr": config.lr,
            "loss": config.loss,
            "smooth_l1_beta": config.smooth_l1_beta,
            "device": config.device,
            "data_identity": bundle.identity,
            "evaluation_exclusion": {"file": exclusion_path.name,
                                     "sha256": exclusion_sha},
            "lineage": config.lineage,
            "created_at": _now(),
        },
    }
    model_path = out_dir / "model.gmvalue"
    info = vf.write_value_weights(model_path, quantized, metadata,
                                  ea=config.ea, act_max=config.act_max)
    weights, loaded_meta = vf.read_value_weights(model_path)

    validation = bundle.validation
    x, y = value_data.tensors(validation)
    device = next(model.parameters()).device
    x = x.to(device)
    model.eval()
    with torch.no_grad():
        float_values = model(x).cpu().tolist()
    quant_model = ValueNet.from_quantized(weights).to(device)
    quant_model.eval()
    with torch.no_grad():
        quant_values = quant_model(x).cpu().tolist()
    integer_values = [vf.value_from_int_z(
        vf.infer_int(weights, sample.sparse()), weights)
        for sample in validation]
    targets = y.tolist()

    export_metrics = _diff_metrics(float_values, integer_values)
    quant_metrics = _diff_metrics(quant_values, integer_values)
    target_metrics = _diff_metrics(targets, integer_values)
    report = {
        "format": "GMVALUE-FIDELITY1",
        "created_at": _now(),
        "model": info,
        "metadata": loaded_meta,
        "architecture": ARCH,
        "rule": "renju",
        "feature_spec": "absolute-color-450",
        "value_contract": "black-absolute-tanh-v1",
        "fixed_point": {"ea": config.ea, "scale": scale,
                        "act_max_int": weights["act_max_int"]},
        "data_identity": bundle.identity,
        "training_metrics": {"train": train_metrics, "validation": val_metrics,
                             "steps": int(steps)},
        "fidelity": {
            "float_vs_integer": export_metrics,
            "quantised_float_vs_integer": quant_metrics,
            "target_vs_integer_mae": target_metrics,
            "value_tolerance": vf.VALUE_TOLERANCE,
            "integer_tolerance": vf.INTEGER_TOLERANCE,
            "export_pass": _tolerance_pass(export_metrics, vf.VALUE_TOLERANCE),
            "integer_pass": _tolerance_pass(quant_metrics,
                                            vf.INTEGER_TOLERANCE),
        },
        "validation_samples": [
            {"target": t, "float": f, "integer": i}
            for t, f, i in zip(targets, float_values, integer_values)],
    }
    report_path = out_dir / "fidelity.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return {"model": info, "report": str(report_path),
            "passed": report["fidelity"]["export_pass"]
                      and report["fidelity"]["integer_pass"],
            "fidelity": report["fidelity"]}


def train_value(config: TrainConfig, *, stop_event=None, progress=None,
                data_bundle: value_data.DataBundle | None = None) -> dict:
    """Runs (or resumes) training and exports the deployment model."""
    started = time.monotonic()
    bundle = data_bundle if data_bundle is not None else value_data.load_pool(config.pool)
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    rng = random.Random(config.seed)
    model_config = ValueConfig(ea=config.ea, act_max=config.act_max)
    model = ValueNet(seed=config.seed, config=model_config).to(device)
    if config.initial_model:
        if config.resume:
            raise ValueError("initial_model and resume are mutually exclusive")
        weights, metadata = vf.read_value_weights(config.initial_model)
        if (weights["ea"] != config.ea or weights["act_max_int"] != round(config.act_max * (1 << config.ea))):
            raise ValueError("initial model fixed-point configuration mismatch")
        manifest = metadata.get("training", {}).get("evaluation_exclusion", {})
        position_path = Path(config.initial_model).parent / "training_positions.json"
        if (manifest.get("file") != position_path.name
                or hashlib.sha256(position_path.read_bytes()).hexdigest() != manifest.get("sha256")):
            raise ValueError("initial model lacks verified training provenance")
        prior = json.loads(position_path.read_text(encoding="utf-8"))
        bundle = replace(bundle, previous_positions=tuple(prior["d4_keys"]),
                         previous_sources=tuple(prior["source_ids"]))
        model = ValueNet.from_quantized(weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    start_step = 0
    history: list[dict] = []
    resumed_from = None
    if config.resume:
        checkpoint = _load_checkpoint(config.resume, config)
        prior = checkpoint.get("training_positions", {})
        if prior.get("format") != "GMVALUE-TRAINING-POSITIONS1":
            raise ValueError("checkpoint lacks frozen training positions; restart from zero")
        bundle = replace(bundle, previous_positions=tuple(prior["d4_keys"]),
                         previous_sources=tuple(prior["source_ids"]))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint.get("step", 0))
        history = list(checkpoint.get("history", []))
        resumed_from = str(config.resume)

    positions = value_data.training_positions(bundle)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "train_history.jsonl"
    if not config.resume:
        history_path.write_text("", encoding="utf-8")
    elif history:
        history_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in history)
            + "\n", encoding="utf-8")

    steps = max(start_step, config.steps)
    train_loss = None
    val_metrics = {"loss": None, "mae": None, "samples": 0}
    best_val = math.inf
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    stopped = False
    stop_reason = ""
    step = start_step
    completed_steps = start_step
    samples_exposed = 0
    pending_best = None
    last_checkpoint_write = last_best_write = time.monotonic()
    pending_history = []
    last_history_write = time.monotonic()

    def append_history(entry):
        nonlocal last_history_write
        pending_history.append(json.dumps(entry, ensure_ascii=False) + "\n")
        if time.monotonic() - last_history_write >= 2.0:
            with history_path.open("a", encoding="utf-8") as handle:
                handle.writelines(pending_history)
            pending_history.clear()
            last_history_write = time.monotonic()
    # Bound the cache for unrestricted CLI datasets; autonomous snapshots are
    # bounded separately. Avoid encoding 450 features in Python every step.
    cached = (value_data.CachedTrainingBatches(bundle.train, device)
              if len(bundle.train) * vf.N_FEATURES * 4 <= 256 * 1024 * 1024 else None)
    for step in range(start_step, steps):
        if stop_event is not None and stop_event.is_set():
            stopped, stop_reason = True, "stopped"
            break
        if config.max_minutes is not None \
                and (time.monotonic() - started) / 60.0 >= config.max_minutes:
            stopped, stop_reason = True, "time_limit"
            break
        if cached is not None:
            x, y, weights = cached.sample(config.batch_size, rng)
        else:
            chosen = [bundle.train[rng.randrange(len(bundle.train))]
                      for _ in range(min(config.batch_size, len(bundle.train)))]
            x, y = value_data.tensors(chosen, augment=True, rng=rng)
            x, y = x.to(device), y.to(device)
            weights = torch.tensor([s.weight for s in chosen], device=device)
        model.train()
        optimizer.zero_grad()
        prediction = model(x)
        loss = value_loss(prediction, y, config.loss, config.smooth_l1_beta, weights)
        loss.backward()
        optimizer.step()
        completed_steps = step + 1
        samples_exposed += x.shape[0]
        if ((step + 1) % max(1, config.log_every) == 0
                or (step + 1) % max(1, config.val_every) == 0 or step + 1 == steps):
            train_loss = float(loss.detach())
        if (step + 1) % max(1, config.log_every) == 0 or step + 1 == steps:
            entry = {"step": step + 1, "train_loss": train_loss,
                     "elapsed_s": round(time.monotonic() - started, 3)}
            if progress is not None:
                progress(dict(entry, val_loss=val_metrics["loss"],
                              val_mae=val_metrics["mae"], stopped=stopped))
            history.append(entry)
            append_history(entry)
        if (step + 1) % max(1, config.val_every) == 0 or step + 1 == steps:
            val_metrics = evaluate(model, bundle.validation, device,
                                   config.batch_size, config.loss,
                                   config.smooth_l1_beta)
            entry = {"step": step + 1, "val_loss": val_metrics["loss"],
                     "val_mae": val_metrics["mae"],
                     "elapsed_s": round(time.monotonic() - started, 3)}
            history.append(entry)
            append_history(entry)
            if progress is not None:
                progress(dict(entry, train_loss=train_loss, stopped=stopped))
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                pending_best = copy.deepcopy(_checkpoint_payload(model=model, optimizer=optimizer,
                                 step=step + 1, config=config,
                                 identity=bundle.identity, history=history,
                                 device=device, training_positions=positions))
            if (pending_best is not None
                    and time.monotonic() - last_best_write >= config.checkpoint_min_interval_s):
                torch.save(pending_best, best_path)
                pending_best = None
                last_best_write = time.monotonic()
        if ((step + 1) % max(1, config.checkpoint_every) == 0 and step + 1 < steps
                and time.monotonic() - last_checkpoint_write >= config.checkpoint_min_interval_s):
            _save_checkpoint(last_path, model=model, optimizer=optimizer,
                             step=step + 1, config=config,
                             identity=bundle.identity, history=history,
                             device=device, training_positions=positions)
            last_checkpoint_write = time.monotonic()

    if pending_history:
        with history_path.open("a", encoding="utf-8") as handle:
            handle.writelines(pending_history)
    if pending_best is not None:
        torch.save(pending_best, best_path)
    _save_checkpoint(last_path, model=model, optimizer=optimizer,
                     step=completed_steps, config=config,
                     identity=bundle.identity, history=history,
                     device=device, training_positions=positions)

    summary = {
        "format": "GMVALUE-TRAIN1",
        "stopped": stopped,
        "stop_reason": stop_reason,
        "steps": completed_steps,
        "start_step": start_step,
        "samples_exposed": samples_exposed,
        "resumed_from": resumed_from,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "train_loss": train_loss,
        "validation": val_metrics,
        "device": device,
        "seed": config.seed,
        "loss": config.loss,
        "data_identity": bundle.identity,
        "arch": ARCH,
        "label_versions": bundle.identity.get("label_versions", []),
        "checkpoint": {"last": str(last_path),
                       "best": str(best_path) if best_path.exists() else None},
        "history": str(history_path),
    }
    if config.export:
        summary["export"] = export_model(
            model, bundle, config, steps=completed_steps,
            train_metrics={"loss": train_loss},
            val_metrics=val_metrics, output_dir=config.output_dir)
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", default="data/pool/pool.sqlite3")
    parser.add_argument("--output-dir", default="weights/value-run1")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", choices=("mse", "smooth_l1"), default="mse")
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--val-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--ea", type=int, default=vf.EA_DEFAULT)
    parser.add_argument("--act-max", type=float, default=vf.ACT_MAX_DEFAULT)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=None)
    args = parser.parse_args(argv)
    config = TrainConfig(
        pool=args.pool, output_dir=args.output_dir, seed=args.seed,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        loss=args.loss, smooth_l1_beta=args.smooth_l1_beta,
        weight_decay=args.weight_decay, device=args.device,
        val_every=args.val_every, checkpoint_every=args.checkpoint_every,
        log_every=args.log_every, resume=args.resume, ea=args.ea,
        act_max=args.act_max, export=not args.no_export,
        max_minutes=args.max_minutes)
    try:
        summary = train_value(config)
    except (ValueError, RuntimeError) as exc:
        print(f"value_train: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("export") and not summary["export"]["passed"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
