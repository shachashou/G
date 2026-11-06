"""Versioned linear score supervision shared by freeze, train and diagnostics."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

import weights_format as wf
from teacher_labels import LabelStore, file_identity, verify_identity


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def pool_identity(entries, store: LabelStore) -> str:
    rows = []
    for entry in entries:
        _, _, cache = store.for_source(entry["path"])
        actual = file_identity(entry["path"])
        if actual["sha256"] != cache["source"]["sha256"]:
            raise ValueError("training source changed after label build")
        rows.append({"data": actual["sha256"], "games": sorted(entry["games"]),
                     "mask": cache["mask"]["sha256"]})
    return digest(sorted(rows, key=lambda r: r["data"]))


def choose_gain(max_abs: float) -> int:
    if not math.isfinite(max_abs) or max_abs < 0:
        raise ValueError("invalid training label range")
    for gain in wf.OUTPUT_GAINS:
        if max_abs <= 720 * gain:
            return gain
    raise ValueError("training label range exceeds predeclared gain capacity")


def load(path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value.get("schema") != 1 or value.get("name") not in ("handcrafted-linear", "search-linear")
            or value.get("label_source") != ("handcrafted" if value["name"] == "handcrafted-linear" else "search")
            or value.get("perspective") != "side-to-move" or value.get("units") != "engine-score"
            or value.get("loss") != "smooth-l1-beta1" or value.get("format_version") != 2
            or value.get("output_gain") not in wf.OUTPUT_GAINS
            or value.get("loss_normalizer") != value["output_gain"]
            or value.get("static_limit") != wf.STATIC_LIMIT):
        raise ValueError("invalid linear target contract")
    verify_identity(value["teacher_manifest"])
    return value


def losses(prediction, label, contract):
    import torch.nn.functional as F
    residual = (prediction - label) / contract["loss_normalizer"]
    return F.smooth_l1_loss(residual, residual.new_zeros(residual.shape), beta=1), residual.square().mean()


def huber(v):
    a = abs(v)
    return 0.5 * a * a if a < 1 else a - 0.5


def fit_constants(labels, normalizer):
    if not labels:
        raise ValueError("empty training labels")
    lo, hi = min(labels), max(labels)
    for _ in range(60):
        middle = (lo + hi) / 2
        gradient = sum(max(-1, min(1, (middle - y) / normalizer)) for y in labels)
        if gradient > 0:
            hi = middle
        else:
            lo = middle
    return {"mse_constant": statistics.mean(labels), "huber_constant": (lo + hi) / 2,
            "training_records": len(labels), "fitted_on": "entire valid training manifest"}


def metrics(predictions, labels, normalizer, constants=None):
    if len(predictions) != len(labels) or not labels or not all(math.isfinite(float(v)) for v in [*predictions, *labels]):
        raise ValueError("invalid metric predictions/labels")
    residuals = [float(p) - y for p, y in zip(predictions, labels)]
    mse = statistics.mean((r / normalizer) ** 2 for r in residuals)
    result = {"records": len(labels), "mae": statistics.mean(abs(r) for r in residuals),
              "rmse": math.sqrt(statistics.mean(r*r for r in residuals)), "normalised_mse": mse,
              "huber": statistics.mean(huber(r / normalizer) for r in residuals),
              "output_min": min(predictions), "output_max": max(predictions),
              "static_limit_hits": sum(abs(p) >= wf.STATIC_LIMIT for p in predictions)}
    if constants:
        base_mse = statistics.mean(((constants["mse_constant"] - y) / normalizer)**2 for y in labels)
        base_huber = statistics.mean(huber((constants["huber_constant"] - y) / normalizer) for y in labels)
        result.update(constants=constants, baseline_mse=base_mse, baseline_huber=base_huber,
                      mse_ratio=mse/base_mse if base_mse else None,
                      huber_ratio=result["huber"]/base_huber if base_huber else None)
    return result


def checkpoint_datasets(payload, data, frozen_path):
    """Rebuild and verify the ACTUAL training pool; never silently resplit."""
    from types import SimpleNamespace
    import train
    if payload.get("target_contract") is None:
        raise ValueError("checkpoint has no linear target contract")
    contract_path = verify_identity(payload["target_contract_identity"])
    if load(contract_path) != payload["target_contract"]:
        raise ValueError("embedded checkpoint target differs from sealed contract")
    if frozen_path is None or file_identity(frozen_path)["sha256"] != payload["frozen_validation_identity"]["sha256"]:
        raise ValueError("checkpoint requires its original derived freeze")
    manifest_path = verify_identity(payload["train_manifest_identity"])
    if file_identity(data)["sha256"] != payload["data_identity"]["sha256"]:
        raise ValueError("checkpoint source identity mismatch")
    arguments = dict(payload["train_args"])
    arguments.update(data=str(data), frozen_validation=Path(frozen_path), train_manifest=manifest_path,
                     target_contract=contract_path, target=payload["target_contract"])
    train_set, val_set, frozen, manifest = train.build_datasets(SimpleNamespace(**arguments))
    train.assert_split_disjoint(train_set, val_set)
    return train_set, val_set, frozen, manifest
