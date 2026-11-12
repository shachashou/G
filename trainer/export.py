#!/usr/bin/env python3
"""Ticket 08: export quantised NNUE weights + self-verification.

  1. Read a training checkpoint and round the master weights to int8/int16.
  2. Compare the PyTorch float forward, the integer-equivalent STE forward and
     the dependency-free integer reference forward on sampled positions.
  3. Calibrate the two output-head biases with the median float/int difference.
  4. Write `GMNNUE01` v1 weights, read them back (magic/version/hash/dims) and
     replay the integer forward to prove the file is self-consistent.
  5. Write a JSON report with the float/int parity error -> the tolerance used
     by ticket 09's C++ scalar-vs-PyTorch alignment test.

Example::

  .venv\\Scripts\\python.exe trainer\\export.py ^
      --checkpoint weights\\nnue_v1.pt --data data\\raw\\round1\\train.binpack ^
      --output weights\\nnue_v1.nnue --report weights\\export_report.json
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

import binpack
import features
import nnue
import validation_freeze as vf
import weights_format as wf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import file_identity


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True,
                        help="binpack file used for parity sampling")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "weights" / "nnue_v1.nnue")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "weights" / "export_report.json")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--frozen-validation", type=Path, default=None,
                        help="ticket 13: parity/calibration sampling is "
                             "restricted to the frozen train side")
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be positive")
    return args


def load_model(checkpoint: Path) -> tuple[nnue.NNUEModel, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = nnue.ModelConfig(**payload["config"])
    model = nnue.NNUEModel(config=config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def sample_positions(path: Path, count: int,
                     allowed_ids: list[int] | None = None
                     ) -> list[tuple[list[int], int, float]]:
    """Evenly spaced sample; ``allowed_ids`` (ticket 13) restricts it to the
    frozen train side so no validation position can calibrate the heads."""
    with binpack.BinpackReader(path) as reader:
        pool = (list(range(reader.available)) if allowed_ids is None
                else list(allowed_ids))
        samples = min(count, len(pool))
        if samples == 0:
            raise SystemExit("binpack file has no records to sample")
        stride = max(1, len(pool) // samples)
        result = []
        for offset in range(0, len(pool), stride):
            if len(result) >= samples:
                break
            record = reader[pool[offset]]
            result.append((record.feature_indices(), record.stm,
                           float(record.score)))
        return result


def int_evals(weights: dict, samples) -> list[int]:
    return [wf.infer_int(weights, sample[0], sample[1]) for sample in samples]


def error_stats(a: list[float], b: list[float]) -> dict:
    diffs = [abs(float(x) - float(y)) for x, y in zip(a, b)]
    return {
        "samples": len(diffs),
        "mae": round(sum(diffs) / max(1, len(diffs)), 5),
        "max": round(max(diffs), 5) if diffs else 0.0,
        "exact_matches": sum(1 for d in diffs if d == 0),
    }


def calibrate_bias(weights: dict, float_eval, integer_eval, heads) -> list[int]:
    """Training/calibration-set correction, reusable by independent diagnostics."""
    bias = []
    for head in (0, 1):
        deltas = [float(f) - float(i) for f, i, h in zip(float_eval, integer_eval, heads)
                  if h == head]
        gain = weights.get("output_gain", 1)
        scaled = statistics.median(deltas) / gain if deltas else 0
        delta = (int(round(statistics.median(deltas))) if deltas else 0) if weights.get("format_version", 1) == 1 else int(math.copysign(math.floor(abs(scaled) + 0.5), scaled))
        original = weights["b2"][head]
        weights["b2"][head] = max(wf.INT16_MIN, min(wf.INT16_MAX, original + delta))
        bias.append(weights["b2"][head] - original)
    return bias


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model, payload = load_model(args.checkpoint)
    calibrate = not args.no_calibrate
    frozen_identity = None
    allowed_ids = None
    if args.frozen_validation is not None:
        frozen = vf.load_frozen(args.frozen_validation)
        vf.verify_frozen_data(frozen, args.data)
        frozen_identity = file_identity(args.frozen_validation)
        recorded = payload.get("frozen_validation_identity") or {}
        if recorded.get("sha256") != frozen_identity["sha256"]:
            raise SystemExit("checkpoint was not trained with this frozen "
                             "validation file")
        with binpack.BinpackReader(args.data) as reader:
            allowed_ids = vf.frozen_train_record_ids(reader.game_index_map(),
                                                     frozen)
    weights = model.quantized_weights()
    samples = sample_positions(args.data, args.samples, allowed_ids)

    x = torch.tensor([features.dense_vector(indices) for indices, _, _ in samples],
                     dtype=torch.float32)
    stm = torch.tensor([s for _, s, _ in samples], dtype=torch.long)
    with torch.no_grad():
        float_eval = model(x, stm, quantized=False).cpu().tolist()
        quant_eval = model(x, stm, quantized=True).cpu().tolist()

    raw_int = int_evals(weights, samples)
    trained_quantized = bool(payload.get("train_args", {}).get("quant_forward", False))
    calibration_reference = quant_eval if trained_quantized else float_eval
    before = error_stats(float_eval, raw_int)
    # The STE integer-equivalent forward must match the reference exactly.
    ste_vs_int = error_stats(quant_eval, raw_int)

    head_bias = [0, 0]
    if calibrate:
        head_bias = calibrate_bias(weights, calibration_reference, raw_int, [s[1] for s in samples])

    # Parity is measured against a model rebuilt from the FINAL exported
    # weights (including the calibrated output biases).
    final_model = nnue.NNUEModel.from_quantized_weights(weights, model.config)
    with torch.no_grad():
        final_float = final_model(x, stm, quantized=False).cpu().tolist()
        final_quant = final_model(x, stm, quantized=True).cpu().tolist()
    integer_eval = int_evals(weights, samples)
    parity = error_stats(final_float, integer_eval)
    final_ste = error_stats(final_quant, integer_eval)

    if ste_vs_int["max"] != 0:
        raise ValueError("STE/integer mismatch")
    blob = wf.pack_weights(weights, qa=model.config.qa, qb=model.config.qb,
                           score_scale=model.config.score_scale, weight_clamp=model.config.weight_clamp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(blob)
    reloaded, meta = wf.read_weights(args.output)
    replay = int_evals(reloaded, samples)
    if replay != integer_eval:
        raise SystemExit("round-trip failed: integer forward differs after read-back")

    report = {
        "complete": True,
        "checkpoint_identity": file_identity(args.checkpoint),
        "original_vs_final": error_stats(float_eval, integer_eval),
        "trained_forward_vs_final": error_stats(calibration_reference, integer_eval),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_val_loss": payload.get("val_loss"),
        "frozen_validation_identity": frozen_identity,
        "calibration_sampling": ("frozen train side only (validation games "
                                 "and D4-leaking games excluded)"
                                 if frozen_identity is not None
                                 else "whole file (legacy behaviour)"),
        "weights_file": str(args.output.resolve()),
        "weights_meta": meta,
        "weights_summary": wf.weights_summary(reloaded),
        "quantisation": {
            "qa": wf.QA, "qb": wf.QB, "score_scale": wf.SCORE_SCALE,
            "weight_clamp": wf.WEIGHT_CLAMP,
            "head_bias_calibration": head_bias,
            "calibrated": calibrate,
            "calibration_reference": "trained_ste" if trained_quantized else "trained_float",
        },
        "parity": {
            "before_calibration": before,
            "float_vs_integer": parity,
            "ste_forward_vs_integer": ste_vs_int,
            "final_ste_forward_vs_integer": final_ste,
            "suggested_cpp_tolerance": max(2, int(parity["max"] + 2)),
        },
        "self_check": {
            "magic": blob[:8].decode("ascii"),
            "version": meta["version"],
            "output_gain": meta["output_gain"],
            "file_size": len(blob),
            "expected_file_size": wf.FILE_LEN,
            "read_back_replay_matches": True,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    print(f"weights: {args.output} ({len(blob):,} bytes, magic {blob[:8].decode()})")
    print(f"parity : float vs integer MAE {parity['mae']:.4f}, "
          f"max {parity['max']:.4f}; STE vs integer max {ste_vs_int['max']:.4f}")
    print(f"bias calibration: {head_bias}; ticket-09 tolerance suggestion "
          f"<= {report['parity']['suggested_cpp_tolerance']}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
