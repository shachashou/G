"""Ticket 22: train a float model to rank the search's recorded move #1.

The self-play data already records, for every position, the move the depth-4
search chose. That move is the search's top pick, i.e. a "this child is best"
signal far stronger than the handcrafted static evaluator. We train a float
model with a pairwise ranking loss (recorded move beats every other candidate)
and export it as float16 (v3), the two fixes the ticket-20/21 diagnosis found.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

import binpack
import diagnose
import local_ranking_diagnosis as ld
import nnue
import teacher_labels as tl
import train
import weights_format as wf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import file_identity, write_json


def sample_parents(data_path, manifest_path, count, seed):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    allowed = {str(Path(e["path"]).resolve()).lower(): set(e["games"])
               for e in manifest["files"]}
    key = str(Path(data_path).resolve()).lower()
    games = allowed.get(key, set())
    by_stm = {0: [], 1: []}
    with binpack.BinpackReader(data_path) as reader:
        mapping = reader.game_index_map()
        for i, game in enumerate(mapping):
            if game in games:
                by_stm[reader[i].stm].append(reader[i])
    rng = random.Random(seed)
    rng.shuffle(by_stm[0])
    rng.shuffle(by_stm[1])
    per_head = count // 2
    if len(by_stm[0]) < per_head or len(by_stm[1]) < per_head:
        raise ValueError("not enough allowed records for balanced sample")
    selected = by_stm[0][:per_head] + by_stm[1][:per_head]
    rng.shuffle(selected)
    return selected


def build_cases(records, probe):
    rows = tl.probe_rows(probe, "candidates", records, timeout=180)
    cases = []
    for record, row in zip(records, rows):
        moves = [r["move"] for r in row["candidates"]]
        if len(moves) < 2 or record.move not in moves:
            continue
        cases.append({"board": tl.board_text(record), "stm": record.stm,
                      "moves": moves, "recorded": record.move,
                      "score": record.score})
    return cases


def child_boards(case):
    boards = []
    for move in case["moves"]:
        b = list(case["board"])
        b[move] = "x" if case["stm"] == 0 else "o"
        boards.append("".join(b))
    return boards


def build_tensors(cases, device):
    # Efficient scatter: one zero matrix, set the sparse feature columns to 1.
    # Avoids diagnose.tensors() building a 450-wide Python list per position.
    flat = []
    offsets = []
    recorded_idx = []
    cursor = 0
    for case in cases:
        boards = child_boards(case)
        offsets.append(cursor)
        recorded_idx.append(case["moves"].index(case["recorded"]))
        for b in boards:
            flat.append((ld.sparse(b), 1 - case["stm"]))
        cursor += len(boards)
    n = len(flat)
    rows, cols, stm_list = [], [], []
    for i, (sp, s) in enumerate(flat):
        for c in sp:
            rows.append(i)
            cols.append(c)
        stm_list.append(s)
    x = torch.zeros(n, wf.N_FEATURES, dtype=torch.float32, device=device)
    if rows:
        ri = torch.tensor(rows, dtype=torch.long, device=device)
        ci = torch.tensor(cols, dtype=torch.long, device=device)
        x[ri, ci] = 1.0
    stm = torch.tensor(stm_list, dtype=torch.long, device=device)
    # Parent tensors + pointwise search-score targets (tanh space, matches train.py).
    parent_sp = [ld.sparse(c["board"]) for c in cases]
    parent_stm = [c["stm"] for c in cases]
    px = torch.zeros(len(cases), wf.N_FEATURES, dtype=torch.float32, device=device)
    for i, sp in enumerate(parent_sp):
        if sp:
            px[i, torch.tensor(sp, dtype=torch.long, device=device)] = 1.0
    pstm = torch.tensor(parent_stm, dtype=torch.long, device=device)
    target = torch.tensor([math.tanh(c["score"] / 400.0) for c in cases],
                          dtype=torch.float32, device=device)
    return x, stm, offsets, recorded_idx, px, pstm, target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--parents", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pointwise-weight", type=float, default=0.0,
                        help="weight of the search-score pointwise MSE term; "
                             "0 = pure ranking, >0 = joint loss")
    parser.add_argument("--hidden-lr", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = train.resolve_device(args.device)
    probe = Path(args.probe).resolve()

    print("sampling parents ...", flush=True)
    records = sample_parents(args.data, args.train_manifest, args.parents, args.seed)
    cases = build_cases(records, probe)
    print(f"built {len(cases)} cases from {len(records)} parents", flush=True)
    if len(cases) < 100:
        raise SystemExit("too few usable cases")

    x, stm, offsets, recorded_idx, px, pstm, target = build_tensors(cases, device)
    n_parents = len(offsets)
    n_children = len(x)
    # Fixed subset for periodic top-1 diagnostics.
    log_rng = random.Random(args.seed + 2)
    probe_parents = sorted(log_rng.sample(range(n_parents), min(400, n_parents)))

    model = nnue.NNUEModel(seed=args.seed, config=nnue.ModelConfig())
    model.to(device)
    optimiser = train.make_optimizer(model, lr=0.1, hidden_lr=args.hidden_lr)
    rng = random.Random(args.seed + 1)
    started = time.monotonic()
    curve = []
    print(f"training {args.steps} steps ...", flush=True)
    for step in range(1, args.steps + 1):
        picks = [rng.randrange(n_parents) for _ in range(args.batch)]
        idx = []
        for p in picks:
            base = offsets[p]
            n = (offsets[p + 1] if p + 1 < n_parents else len(x)) - base
            rec = base + recorded_idx[p]
            cand = [base + c for c in range(n) if base + c != rec]
            if not cand:
                continue
            idx.append((rec, rng.choice(cand)))
        if not idx:
            continue
        # Forward ONLY the sampled (recorded, candidate) pairs, not all children.
        sel = torch.tensor([v for pair in idx for v in pair],
                           dtype=torch.long, device=device)
        s_sel = -model(x[sel], stm[sel], quantized=False)  # move score = -child eval
        logits = s_sel[0::2] - s_sel[1::2]
        rank_loss = torch.nn.functional.softplus(-logits).mean()
        loss = rank_loss
        if args.pointwise_weight > 0:
            pidx = torch.tensor([rng.randrange(n_parents) for _ in range(args.batch)],
                                dtype=torch.long, device=device)
            p_pred = model(px[pidx], pstm[pidx], quantized=False)
            point_loss = (torch.tanh(p_pred / 400.0) - target[pidx]).square().mean()
            loss = loss + args.pointwise_weight * point_loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step == 1 or step % 200 == 0 or step == args.steps:
            # top-1 agreement on a fixed subset of parents (cheap, diagnostic).
            with torch.no_grad():
                agree = 0
                for p in probe_parents:
                    base = offsets[p]
                    n = (offsets[p + 1] if p + 1 < n_parents else len(x)) - base
                    s_local = -model(x[base:base + n], stm[base:base + n]).cpu()
                    if int(s_local.argmax()) == recorded_idx[p]:
                        agree += 1
            top1 = agree / len(probe_parents)
            curve.append({"step": step, "loss": float(loss.item()), "top1": round(top1, 4)})
            print(f"step {step}: loss={loss.item():.4f} top1={top1:.4f}", flush=True)

    # save a float checkpoint in the same shape train.py/export.py expect
    payload = {
        "diagnostic_only": False, "config": vars(model.config),
        "model_state": model.state_dict(), "epoch": args.steps,
        "val_loss": float(curve[-1]["loss"] if curve else 0.0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.history, {"complete": True, "curve": curve, "top1_final": curve[-1]["top1"] if curve else None,
                              "device": train.device_info(device), "parents": n_parents,
                              "elapsed_seconds": time.monotonic() - started})
    print(json.dumps({"complete": True, "top1_final": curve[-1]["top1"] if curve else None,
                      "parents": n_parents, "elapsed_seconds": round(time.monotonic() - started, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
