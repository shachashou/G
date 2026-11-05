"""Read-only checkpoint probe for M5 same-set fit; no optimizer steps."""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "trainer"), str(ROOT / "tools")]
import torch
import binpack
import diagnose
import export
import nnue
from run_support import file_identity, write_json

torch.set_num_threads(1)
data = ROOT / "data/raw/bench-d4/train.binpack"
results = {}
for name in sys.argv[1:]:
    folder = ROOT / "data/m5" / name
    report = json.loads((folder / "report.json").read_text())
    if not report["complete"] or report["data"]["sha256"] != file_identity(data)["sha256"]:
        raise ValueError("incomplete experiment or dataset identity mismatch")
    with binpack.BinpackReader(data) as reader:
        samples = diagnose.samples_at(reader, report["manifest"]["record_indices"],
                                      report["config"]["score_clip"])
    x, stm, scores, labels = diagnose.tensors(samples)
    targets = torch.tanh(scores / 400).tolist()
    groups = defaultdict(list)
    for sample, target in zip(samples, targets):
        groups[(sample[1], tuple(sample[0]))].append(target)
    irreducible = sum(sum((v - statistics.mean(group)) ** 2 for v in group)
                      for group in groups.values()) / len(samples)
    result = {"same_set_constant_mse": statistics.pvariance(targets),
              "duplicate_positions": len(samples) - len(groups),
              "score_sign_vs_decisive_wdl_disagreements": sum(
                  (s[2] > 0 and s[3] == 0) or (s[2] < 0 and s[3] == 2) for s in samples),
              "conflicting_duplicate_mse_lower_bound": irreducible, "arms": {}}
    for arm in ("float", "ste"):
        model, _ = export.load_model(folder / f"{arm}.pt")
        p = model(x, stm, quantized=arm == "ste")
        mse, ce = nnue.mixed_losses(p, scores, labels, model.config.score_scale,
                                    model.config.wdl_temperature, model.wdl_bias)
        reg = model.quantization_regularizer()
        params = list(model.parameters())
        sources = {"mse": mse, "weighted_ce": ce * report["config"]["wdl_weight"],
                   "weighted_regularizer": reg * report["config"]["quant_reg"]}
        gradient = {}
        for source, loss in sources.items():
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            gradient[source] = torch.cat([
                (torch.zeros_like(param) if grad is None else grad).flatten()
                for param, grad in zip(params, grads)])
        task = gradient["mse"] + gradient["weighted_ce"]
        reg_grad = gradient["weighted_regularizer"]
        denom = float(task.norm() * reg_grad.norm())
        task_denom = float(gradient["mse"].norm() * gradient["weighted_ce"].norm())
        with torch.no_grad():
            w0, b0, w1, b1, _, _ = model._tensors(arm == "ste")
            acc = (x @ w0 + b0).clamp(-32768, 32767)
            z = acc.clamp(0, model.config.qa) @ w1
            hidden = (nnue.RoundDivSte.apply(z, model.config.qa) if arm == "ste"
                      else z / model.config.qa) + b1
            low, high = float((hidden <= 0).float().mean()), float((hidden >= 64).float().mean())
        result["arms"][arm] = {
            "mse": float(mse.detach()), "ce": float(ce.detach()),
            "regularizer": float(reg.detach()),
            "gradient_norms": {k: float(g.norm()) for k, g in gradient.items()},
            "task_gradient_norm": float(task.norm()),
            "task_vs_regularizer_cosine": float(task @ reg_grad) / denom if denom else None,
            "mse_vs_weighted_ce_cosine": float(gradient["mse"] @ gradient["weighted_ce"]) / task_denom if task_denom else None,
            "true_forward_hidden_clip": {"low": low, "high": high, "active": 1 - low - high},
            "outputs": {"min": float(p.min().detach()), "max": float(p.max().detach())}}
    results[name] = result
write_json(ROOT / "data/m5/fit_inspection.json", results)
print(json.dumps(results, indent=2, allow_nan=False))
