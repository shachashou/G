#!/usr/bin/env python3
"""Ticket 17 stage A: how far did each layer actually move during training?

`NNUEModel.reset_parameters()` seeds its OWN ``torch.Generator``, so the exact
initial weights can be rebuilt from a checkpoint's ``config`` plus the training
``--seed`` without replaying DataLoader / D4 augmentation RNG streams. This is a
read-only diagnostic: it never modifies a checkpoint or a weight file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import nnue

GROUPS = {
    "accumulator": ("W0", "b0"),
    "l1": ("W1", "b1"),
    "output_head": ("W2", "b2"),
    "wdl_bias": ("wdl_bias",),
}

TRAIN_ARG_KEYS = (
    "lr", "hidden_lr", "wdl_weight", "quant_reg", "quant_forward",
    "max_steps", "epochs", "batch_size", "checkpoint_policy", "score_clip",
)


def _identity(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _stats(before: torch.Tensor, after: torch.Tensor) -> dict:
    delta = after - before
    l2_before = float(before.norm())
    l2_delta = float(delta.norm())
    return {
        "numel": before.numel(),
        "l2_before": l2_before,
        "l2_delta": l2_delta,
        "relative": l2_delta / l2_before if l2_before > 0 else None,
        "max_abs_delta": float(delta.abs().max()),
        "mean_abs_delta": float(delta.abs().mean()),
        "changed_fraction": float((delta != 0).float().mean()),
        "min_before": float(before.min()),
        "max_before": float(before.max()),
        "min_after": float(after.min()),
        "max_after": float(after.max()),
    }


def drift_report(checkpoint: Path, seed: int) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = nnue.ModelConfig(**payload["config"])
    initial = nnue.NNUEModel(seed=seed, config=config).state_dict()
    trained = payload["model_state"]
    missing = sorted(set(initial) - set(trained))
    extra = sorted(set(trained) - set(initial))
    if missing or extra:
        raise SystemExit(f"state_dict mismatch: missing={missing} extra={extra}")
    before = {k: initial[k].detach().float() for k in initial}
    after = {k: trained[k].detach().float() for k in initial}
    layers = {k: _stats(before[k], after[k]) for k in sorted(before)}
    groups = {}
    for name, keys in GROUPS.items():
        present = [k for k in keys if k in before]
        b = torch.cat([before[k].flatten() for k in present])
        a = torch.cat([after[k].flatten() for k in present])
        groups[name] = _stats(b, a) | {"layers": present}
    args = payload.get("train_args", {})
    return {
        "schema": "weight_drift_v1",
        "checkpoint": _identity(checkpoint) | {
            "epoch": payload.get("epoch"),
            "val_loss": payload.get("val_loss"),
        },
        "seed": seed,
        "config": payload["config"],
        "train_args": {k: args.get(k) for k in TRAIN_ARG_KEYS},
        "layers": layers,
        "groups": groups,
        "notes": ("initial state rebuilt with NNUEModel(seed=...).reset_parameters(); "
                  "relative = ||delta||_2 / ||initial||_2 (None when initial norm is 0)"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = drift_report(args.checkpoint, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    for name, stats in report["groups"].items():
        relative = stats["relative"]
        rel_text = "n/a" if relative is None else f"{relative:.6f}"
        print(f"{name:12s} relative={rel_text:>10s} "
              f"max_abs_delta={stats['max_abs_delta']:.6f} "
              f"changed={stats['changed_fraction']:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
