"""Ticket 20: does a float-trained model's ranking survive int8 quantization?

Loads a diagnostic checkpoint, runs the same children through BOTH the
continuous float forward and the quantized integer (STE) forward, and compares
the move-ranking metrics. If ranking survives quantization, the ticket-20
blocker is the STE TRAINING path; if it collapses, the blocker is quantization
itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

import local_ranking_diagnosis as ld
import nnue
import weights_format as wf
from distill18_fidelity import model_predictions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import file_identity, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@torch.no_grad()
def run(checkpoint_path, manifest_path, out_path):
    if out_path.exists():
        raise ValueError("refusing to overwrite diagnosis output")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not ckpt.get("diagnostic_only"):
        raise ValueError("refusing non-diagnostic checkpoint")
    manifest = read(manifest_path)
    if manifest.get("stage") != "C" or not manifest.get("complete"):
        raise ValueError("invalid C manifest")
    model = nnue.NNUEModel(config=nnue.ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    per_parent = manifest["children"]
    flat = [e for entries in per_parent for e in entries]
    samples = [(e["sparse"], e["stm"], e["teacher"], 0) for e in flat]
    float_preds = model_predictions(model, samples, False)
    int_preds = model_predictions(model, samples, True)

    def local_summary(preds):
        rows = []
        cursor = 0
        for case, entries in zip(manifest["selected_parents"], per_parent):
            n = len(entries)
            h = [e["teacher"] for e in entries]
            rows.append(ld.local_scores(h, preds[cursor:cursor + n], case["moves"].index(case["teacher_move"])))
            cursor += n
        return ld.local_summary([{"local": {"x": r}} for r in rows])["x"]

    float_local = local_summary(float_preds)
    int_local = local_summary(int_preds)
    # Weight round-trip distance: how far master weights are from their rounded ints.
    weights = model.quantized_weights()
    drift = {}
    for name in ("W0", "b0", "W1", "b1", "W2", "b2"):
        p = getattr(model, name).detach()
        q = torch.tensor(weights[name], dtype=torch.float32).reshape(p.shape)
        drift[name] = {"mae": float((p - q).abs().mean()), "max": float((p - q).abs().max())}
    report = {"schema": 1, "complete": True, "diagnostic_only": True,
              "checkpoint": file_identity(checkpoint_path), "manifest": file_identity(manifest_path),
              "arm": ckpt.get("arm"), "stage": ckpt.get("stage"),
              "float_local": float_local, "integer_local": int_local,
              "weight_roundtrip_drift": drift,
              "static_limit_hits": {"float": sum(abs(v) >= wf.STATIC_LIMIT for v in float_preds),
                                    "integer": sum(abs(v) >= wf.STATIC_LIMIT for v in int_preds)}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, report)
    print(json.dumps({"arm": ckpt.get("arm"),
                      "float_disagreement": round(float_local["macro_pair_disagreement"], 4),
                      "integer_disagreement": round(int_local["macro_pair_disagreement"], 4),
                      "float_excess": round(float_local["teacher_best_excess"], 4),
                      "integer_excess": round(int_local["teacher_best_excess"], 4)}, ensure_ascii=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.checkpoint.resolve(), args.manifest.resolve(), args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
