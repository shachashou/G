"""Ticket 23: read-only diagnosis of candidate-search-score ranking signal.

For a frozen sample of primary parent positions, evaluate every legal
non-terminal candidate child with (a) the handcrafted static evaluator H and
(b) depth-2 / depth-4 search scores S, both mapped to the parent's move
perspective and saturated to int16 exactly like datagen labels. Report whether
S carries a ranking signal that differs from H (deciding whether to invest in
recording candidate search scores in datagen). Never trains / plays matches.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import teacher_labels as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import file_identity, write_json

SCORE_MAX = 32767  # binpack.cpp encodeScore / kScoreMax


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def clamp_score(v: int) -> int:
    return max(-SCORE_MAX, min(SCORE_MAX, int(v)))


def child_board(case, move: int) -> str:
    b = list(case["board"])
    if b[move] != ".":
        raise ValueError("candidate occupied")
    b[move] = "x" if case["stm"] == 0 else "o"
    return "".join(b)


def probe_scores(probe: Path, boards_stm: list[tuple[str, int]], mode: str,
                 extra: list[str], timeout: float) -> list[int]:
    """One subprocess for the whole batch; probe is line-streaming."""
    payload = "".join(f"{s} {b}\n" for b, s in boards_stm)
    import subprocess
    proc = subprocess.run([str(probe), mode, *extra], input=payload,
                          text=True, capture_output=True, timeout=timeout)
    if proc.returncode:
        raise ValueError(f"probe {mode} failed: {proc.stderr.strip()[:300]}")
    values = [int(v) for v in proc.stdout.splitlines()]
    if len(values) != len(boards_stm):
        raise ValueError(f"probe {mode} count mismatch: {len(values)} != {len(boards_stm)}")
    return values


def argmax_set(vals: list[int]) -> set:
    m = max(vals)
    return {i for i, v in enumerate(vals) if v == m}


def top1_hit(ref: list[int], pred: list[int]) -> float:
    """|argmax(ref) ∩ argmax(pred)| / |argmax(pred)| (uniform tie credit)."""
    rp, pp = argmax_set(ref), argmax_set(pred)
    return len(rp & pp) / len(pp)


def pair_disagreement(ref: list[int], pred: list[int]) -> float | None:
    """Reversal rate of pred vs ref over unequal ref pairs (tie=0.5)."""
    n = len(ref)
    total, wrong = 0, 0.0
    for i in range(n):
        for j in range(i):
            dh = ref[i] - ref[j]
            if dh == 0:
                continue
            de = pred[i] - pred[j]
            wrong += 0.5 if de == 0 else float(dh * de < 0)
            total += 1
    return wrong / total if total else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path,
                        default=ROOT / "data/m5/t19/c_real/move_cases.jsonl")
    parser.add_argument("--probe", type=Path,
                        default=ROOT / "build/engine/Release/nnue_probe.exe")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--parents", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--depths", default="2,4")
    parser.add_argument("--tt-mb", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise ValueError("refusing to overwrite diagnosis directory")
    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    if not depths or any(d < 1 or d > 8 for d in depths):
        raise ValueError("--depths must be 1..8")

    cases_all = read_jsonl(args.cases)
    usable = [c for c in cases_all
              if c["bucket"] == "primary" and len(c["moves"]) >= 2
              and all(r == 0 for r in c["child_results"])]
    import random
    rng = random.Random(args.seed)
    if len(usable) < args.parents:
        raise ValueError(f"only {len(usable)} usable parents, need {args.parents}")
    sampled = rng.sample(usable, args.parents)
    print(f"usable={len(usable)} sampled={len(sampled)}", flush=True)

    # Flat child list with parent offsets.
    flat: list[tuple[str, int]] = []
    offsets = []
    for case in sampled:
        offsets.append(len(flat))
        for move in case["moves"]:
            flat.append((child_board(case, move), 1 - case["stm"]))
    n_children = len(flat)
    print(f"children={n_children} probing H + {depths} ...", flush=True)
    started = time.monotonic()

    H = probe_scores(args.probe, flat, "handcrafted", [], args.timeout)
    S = {d: probe_scores(args.probe, flat, "searchscore",
                         ["--depth", str(d), "--tt-mb", str(args.tt_mb)], args.timeout)
         for d in depths}

    # Parent-perspective values, saturated like datagen labels.
    def per_parent(values):
        out = []
        for k, case in enumerate(sampled):
            s, e = offsets[k], (offsets[k + 1] if k + 1 < len(offsets) else n_children)
            out.append([-clamp_score(v) for v in values[s:e]])
        return out

    Hp = per_parent(H)
    Sp = {d: per_parent(S[d]) for d in depths}

    # Diagnostics.
    def spread_stats(vecs):
        non_tie = sum(1 for v in vecs if len(set(v)) > 1) / len(vecs)
        sat_rate = sum(1 for v in vecs for x in v if abs(x) >= SCORE_MAX) / max(1, sum(len(v) for v in vecs))
        sds = [statistics.pstdev(v) for v in vecs if len(v) >= 2]
        allv = sorted(x for v in vecs for x in v)
        q = {str(p): allv[int((len(allv) - 1) * p)] for p in (0.05, 0.5, 0.95)}
        return {"non_all_tie_ratio": round(non_tie, 4),
                "saturation_ratio": round(sat_rate, 4),
                "mean_within_parent_std": round(statistics.mean(sds), 2) if sds else 0,
                "value_quantiles": q}

    def compare(ref, pred):
        hits = [top1_hit(r, p) for r, p in zip(ref, pred)]
        dis = [pair_disagreement(r, p) for r, p in zip(ref, pred)]
        usable_dis = [d for d in dis if d is not None]
        return {"top1_agreement": round(statistics.mean(hits), 4),
                "pair_disagreement": round(statistics.mean(usable_dis), 4) if usable_dis else None,
                "informative_parents": len(usable_dis)}

    report = {
        "schema": 1, "complete": False, "diagnostic_only": True,
        "source": {"cases": file_identity(args.cases), "probe": file_identity(args.probe),
                   "seed": args.seed, "parents": len(sampled), "children": n_children,
                   "depths": depths, "tt_mb": args.tt_mb},
        "H_spread": spread_stats(Hp),
        "S_spread": {str(d): spread_stats(Sp[d]) for d in depths},
        "H_vs_S": {str(d): compare(Hp, Sp[d]) for d in depths},
        "cross_depth": {},
    }
    if len(depths) >= 2:
        for i in range(len(depths)):
            for j in range(i + 1, len(depths)):
                d1, d2 = depths[i], depths[j]
                report["cross_depth"][f"{d1}_vs_{d2}"] = compare(Sp[d1], Sp[d2])

    # Pre-declared decision (ticket 23 §4), using the deepest depth.
    dmax = max(depths)
    top1 = report["H_vs_S"][str(dmax)]["top1_agreement"]
    dis = report["H_vs_S"][str(dmax)]["pair_disagreement"]
    non_tie = report["S_spread"][str(dmax)]["non_all_tie_ratio"]
    sat = report["S_spread"][str(dmax)]["saturation_ratio"]
    if top1 >= 0.85 and dis is not None and dis <= 0.10:
        decision = "STOP (search-score ranking ~= static H; changing datagen unlikely to help)"
    elif top1 <= 0.60 and non_tie >= 0.70 and sat <= 0.25:
        decision = "INVEST (search scores carry new, well-spread ranking signal)"
    else:
        decision = "INCONCLUSIVE (report all metrics; define a follow-up diagnosis)"
    report["decision"] = {"code": decision, "depth": dmax,
                          "top1_agreement": top1, "pair_disagreement": dis,
                          "non_all_tie_ratio": non_tie, "saturation_ratio": sat}
    report["complete"] = True
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)

    args.out_dir.mkdir(parents=True)
    rows = []
    for k, case in enumerate(sampled):
        rows.append({"record_id": case["record_id"], "game_id": case["game_id"],
                     "stm": case["stm"], "ply": case["ply"], "moves": case["moves"],
                     "teacher_move": case["teacher_move"], "H": Hp[k],
                     **{f"S{d}": Sp[d][k] for d in depths}})
    (args.out_dir / "cases.jsonl").write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")
    write_json(args.out_dir / "report.json", report)
    print(json.dumps({"complete": True, "decision": report["decision"],
                      "H_vs_S": report["H_vs_S"], "elapsed_seconds": report["elapsed_seconds"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
