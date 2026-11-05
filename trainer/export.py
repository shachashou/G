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
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

import binpack
import features
import nnue
import weights_format as wf

ROOT = Path(__file__).resolve().parents[1]


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
    return parser.parse_args(argv)


def load_model(checkpoint: Path) -> tuple[nnue.NNUEModel, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = nnue.ModelConfig(**payload["config"])
    model = nnue.NNUEModel(config=config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def sample_positions(path: Path, count: int) -> list[tuple[list[int], int, float]]:
    with binpack.BinpackReader(path) as reader:
        total = reader.available
        samples = min(count, total)
        if samples == 0:
            raise SystemExit("binpack file has no records to sample")
        stride = max(1, total // samples)
        result = []
        for index in range(0, total, stride):
            if len(result) >= samples:
                break
            record = reader[index]
            result.append((record.feature_indices(), record.stm,
                           float(record.score)))
        return result


def int_evals(weights: dict, samples) -> list[int]:
    return [wf.infer_int(weights, indices, stm) for indices, stm, _ in samples]


def error_stats(a: list[float], b: list[float]) -> dict:
    diffs = [abs(float(x) - float(y)) for x, y in zip(a, b)]
    return {
        "samples": len(diffs),
        "mae": round(sum(diffs) / max(1, len(diffs)), 5),
        "max": round(max(diffs), 5) if diffs else 0.0,
        "exact_matches": sum(1 for d in diffs if d == 0),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model, payload = load_model(args.checkpoint)
    weights = model.quantized_weights()
    samples = sample_positions(args.data, args.samples)

    x = torch.tensor([features.dense_vector(indices) for indices, _, _ in samples],
                     dtype=torch.float32)
    stm = torch.tensor([s for _, s, _ in samples], dtype=torch.long)
    with torch.no_grad():
        float_eval = model(x, stm, quantized=False).cpu().tolist()
        quant_eval = model(x, stm, quantized=True).cpu().tolist()

    raw_int = int_evals(weights, samples)
    before = error_stats(float_eval, raw_int)
    # The STE integer-equivalent forward must match the reference exactly.
    ste_vs_int = error_stats(quant_eval, raw_int)

    head_bias = [0, 0]
    if not args.no_calibrate:
        for head in (0, 1):
            deltas = [float(f) - float(i)
                      for f, i, (_, s, _) in zip(float_eval, raw_int, samples)
                      if s == head]
            if deltas:
                head_bias[head] = int(round(statistics.median(deltas)))
        weights["b2"] = [
            max(wf.INT16_MIN, min(wf.INT16_MAX,
                                  weights["b2"][h] + head_bias[h]))
            for h in (0, 1)
        ]

    # Parity is measured against a model rebuilt from the FINAL exported
    # weights (including the calibrated output biases).
    final_model = nnue.NNUEModel.from_quantized_weights(weights, model.config)
    with torch.no_grad():
        final_float = final_model(x, stm, quantized=False).cpu().tolist()
        final_quant = final_model(x, stm, quantized=True).cpu().tolist()
    integer_eval = int_evals(weights, samples)
    parity = error_stats(final_float, integer_eval)
    final_ste = error_stats(final_quant, integer_eval)

    blob = wf.pack_weights(weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(blob)
    reloaded, meta = wf.read_weights(args.output)
    replay = int_evals(reloaded, samples)
    if replay != integer_eval:
        raise SystemExit("round-trip failed: integer forward differs after read-back")

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_val_loss": payload.get("val_loss"),
        "weights_file": str(args.output.resolve()),
        "weights_meta": meta,
        "weights_summary": wf.weights_summary(reloaded),
        "quantisation": {
            "qa": wf.QA, "qb": wf.QB, "score_scale": wf.SCORE_SCALE,
            "weight_clamp": wf.WEIGHT_CLAMP,
            "head_bias_calibration": head_bias,
            "calibrated": not args.no_calibrate,
        },
        "parity": {
            "before_calibration": before,
            "float_vs_integer": parity,
            "ste_forward_vs_integer": ste_vs_int,
            "final_ste_forward_vs_integer": final_ste,
            "suggested_cpp_tolerance": max(2, int(parity["max"] + 2)),
        },
        "self_check": {
            "magic": wf.MAGIC.decode("ascii"),
            "version": wf.VERSION,
            "file_size": len(blob),
            "expected_file_size": wf.FILE_LEN,
            "read_back_replay_matches": True,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    print(f"weights: {args.output} ({len(blob):,} bytes, magic {wf.MAGIC.decode()})")
    print(f"parity : float vs integer MAE {parity['mae']:.4f}, "
          f"max {parity['max']:.4f}; STE vs integer max {ste_vs_int['max']:.4f}")
    print(f"bias calibration: {head_bias}; ticket-09 tolerance suggestion "
          f"<= {report['parity']['suggested_cpp_tolerance']}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
