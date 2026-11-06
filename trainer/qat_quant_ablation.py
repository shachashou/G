"""Ticket 20 ablation: isolate which quantization step collapses ranking.

Runs the same children through four forward variants of ONE float-trained
model, crossing two knobs: weight rounding (int8) and round_div integer
division. Reports move-ranking for each to localize the collapse.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import file_identity, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def round_div_int(x: torch.Tensor, d: int) -> torch.Tensor:
    """Exact forward of nnue.RoundDivSte (truncate to int, round half away from zero)."""
    n = x.to(torch.int64)
    sign = torch.where(n >= 0, torch.ones_like(n), -torch.ones_like(n))
    quotient = sign * ((2 * torch.abs(n) + d) // (2 * d))
    return quotient.to(torch.float32)


@torch.no_grad()
def variant_forward(model, x, stm, round_weights: bool, round_div: bool) -> list:
    c = model.config
    w0, b0, w1, b1, w2, b2 = model._tensors(round_weights)
    head = stm.long()
    acc = (x @ w0 + b0).clamp(wf.INT16_MIN, wf.INT16_MAX)
    a = acc.clamp(0, c.qa)
    z = a @ w1
    zq = (round_div_int(z, c.qa) if round_div else z / c.qa) + b1
    a2 = zq.clamp(0, c.qb)
    y = (a2 * w2[head]).sum(dim=-1)
    gain = c.output_gain
    eval_score = (round_div_int(y * gain, c.qb) if round_div else y * gain / c.qb) + gain * b2[head]
    if c.format_version == 2:
        eval_score = eval_score.clamp(-wf.STATIC_LIMIT, wf.STATIC_LIMIT)
    return eval_score.tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("refusing to overwrite ablation output")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not ckpt.get("diagnostic_only"):
        raise ValueError("refusing non-diagnostic checkpoint")
    manifest = read(args.manifest)
    model = nnue.NNUEModel(config=nnue.ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    per_parent = manifest["children"]
    flat = [e for entries in per_parent for e in entries]
    x, stm = [], []
    for e in flat:
        vec = [0] * wf.N_FEATURES
        for i in e["sparse"]:
            vec[i] = 1.0
        x.append(vec)
        stm.append(e["stm"])
    x = torch.tensor(x)
    stm = torch.tensor(stm, dtype=torch.long)

    def summary(preds):
        rows = []
        cursor = 0
        for case, entries in zip(manifest["selected_parents"], per_parent):
            n = len(entries)
            h = [e["teacher"] for e in entries]
            rows.append(ld.local_scores(h, preds[cursor:cursor + n], case["moves"].index(case["teacher_move"])))
            cursor += n
        return ld.local_summary([{"local": {"x": r}} for r in rows])["x"]

    variants = {
        "float_W_float_div": (False, False),
        "rounded_W_float_div": (True, False),
        "float_W_round_div": (False, True),
        "rounded_W_round_div": (True, True),  # full integer forward
    }
    report = {"schema": 1, "complete": True, "diagnostic_only": True,
              "checkpoint": file_identity(args.checkpoint), "manifest": file_identity(args.manifest),
              "arm": ckpt.get("arm"), "variants": {}}
    for name, (rw, rd) in variants.items():
        preds = variant_forward(model, x, stm, rw, rd)
        report["variants"][name] = {
            "disagreement": summary(preds)["macro_pair_disagreement"],
            "excess": summary(preds)["teacher_best_excess"],
            "static_limit_hits": sum(abs(v) >= wf.STATIC_LIMIT for v in preds)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    for name, v in report["variants"].items():
        print(f"{name}: disagreement={v['disagreement']:.4f} excess={v['excess']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
