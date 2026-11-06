#!/usr/bin/env python3
"""Ticket 08: train the NNUE net on self-play binpack data.

Loss = MSE(tanh(eval/400), tanh(score/400)) + wdl_weight * CE(WDL)
       + quant_reg * distance-to-nearest-integer regularizer.

Example (project venv, RTX 4080 SUPER auto-detected)::

  .venv\\Scripts\\python.exe trainer\\train.py ^
      --data data\\raw\\round1\\train.binpack ^
      --output weights\\nnue_v1.pt --history weights\\training_history.json ^
      --epochs 10 --batch-size 1024

Every epoch writes train/validation loss into the history JSON so the
convergence curve can be reviewed later. The best checkpoint is kept.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import dataset as ds
import nnue
import validation_freeze as vf
import targets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import file_identity


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="binpack file (shards merged)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "weights" / "nnue_v1.pt")
    parser.add_argument("--history", type=Path,
                        default=ROOT / "weights" / "training_history.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden-lr", type=float,
                        help="optional W0/b0/W1/b1 learning rate; output keeps --lr")
    parser.add_argument("--wdl-weight", type=float, default=0.2)
    parser.add_argument("--wdl-temperature", type=float, default=100.0)
    parser.add_argument("--quant-reg", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="keep 0 on Windows (mmap + fork)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap samples per split for a quick smoke run")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="ticket 15: cap TOTAL optimizer updates across "
                             "epochs so two data-volume arms receive identical "
                             "compute; history records the actual step count")
    parser.add_argument("--checkpoint-policy", choices=("best", "last"),
                        default="best",
                        help="ticket 15: 'last' saves the model after the final "
                             "optimizer step so equal-step arms export the same "
                             "amount of training; default keeps the best-val "
                             "checkpoint of ticket 08")
    parser.add_argument("--quant-forward", action="store_true",
                        help="QAT mode: integer-equivalent STE forward")
    parser.add_argument("--score-clip", type=float, default=0.0,
                        help="clip saturated +/-32767 search scores to +/-N "
                             "(0 disables; loss stays MSE(tanh(score/400)))")
    parser.add_argument("--frozen-validation", type=Path, default=None,
                        help="ticket-13 frozen validation JSON: fixes the "
                             "validation games and applies the D4-leakage "
                             "train-game exclusions; --val-fraction/--seed "
                             "no longer drive the split")
    parser.add_argument("--train-manifest", type=Path, default=None,
                        help="ticket-13 train manifest (per-file include "
                             "lists from validation_freeze.py check); adds "
                             "training files without touching the frozen "
                             "validation split")
    parser.add_argument("--target-contract", type=Path)
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be >= 1")
    return args


def training_losses(model, prediction, score, wdl, args):
    contract = getattr(args, "target", None)
    if contract:
        huber, mse = targets.losses(prediction, score, contract)
        return huber, mse, prediction.new_zeros(())
    mse, ce = nnue.mixed_losses(prediction, score, wdl, model.config.score_scale,
                                model.config.wdl_temperature, model.wdl_bias)
    return mse + args.wdl_weight * ce, mse, ce


def make_optimizer(model: nnue.NNUEModel, lr: float,
                   hidden_lr: float | None = None):
    """Keep large hidden sums from saturating while the output scale learns."""
    if hidden_lr is None:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    hidden_names = {"W0", "b0", "W1", "b1"}
    return torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if n in hidden_names],
         "lr": hidden_lr},
        {"params": [p for n, p in model.named_parameters() if n not in hidden_names],
         "lr": lr}], weight_decay=1e-5)


def collate_batch(batch):
    """Stack the 450-dim lists explicitly; default_collate transposes lists.

    `BinpackDataset` stays dependency-free (returns plain Python lists), so the
    collation into tensors belongs here.
    """
    x = torch.tensor([item[0] for item in batch], dtype=torch.float32)
    stm = torch.tensor([item[1] for item in batch], dtype=torch.long)
    score = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    wdl = torch.tensor([item[3] for item in batch], dtype=torch.long)
    return x, stm, score, wdl


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def device_info(device: torch.device) -> dict:
    return {
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0)
        if device.type == "cuda" else None,
    }


def evaluate(model: nnue.NNUEModel, loader: DataLoader, device: torch.device,
             args: argparse.Namespace) -> dict:
    model.eval()
    total, mse_sum, ce_sum, count = 0.0, 0.0, 0.0, 0
    predictions, labels, heads = [], [], []
    with torch.no_grad():
        for x, stm, score, wdl in tqdm(loader, desc="eval", leave=False,
                                       unit="batch"):
            x = x.float().to(device)
            stm = stm.long().to(device)
            score = score.float().to(device)
            wdl = wdl.long().to(device)
            eval_score = model(x, stm, quantized=args.quant_forward)
            primary, mse, ce = training_losses(model, eval_score, score, wdl, args)
            batch = len(x)
            total += float(primary) * batch
            mse_sum += float(mse) * batch
            ce_sum += float(ce) * batch
            count += batch
            if getattr(args, "target", None):
                predictions.extend(eval_score.cpu().tolist())
                labels.extend(score.cpu().tolist())
                heads.extend(stm.cpu().tolist())
    if count == 0:
        raise SystemExit("validation split is empty; lower --val-fraction or add games")
    result = {"val_loss": total / count, "val_mse": mse_sum / count,
              "val_ce": ce_sum / count, "val_records": count}
    if getattr(args, "target", None):
        c = args.target
        result["linear_metrics"] = targets.metrics(predictions, labels, c["loss_normalizer"], c["constants"])
        result["linear_metrics"]["per_head"] = {
            str(h): targets.metrics([p for p, s in zip(predictions, heads) if s == h],
                                    [y for y, s in zip(labels, heads) if s == h], c["loss_normalizer"])
            for h in set(heads)}
    return result


def run_training(args: argparse.Namespace, train_set,
                 val_set: ds.BinpackDataset, device: torch.device,
                 info: dict, frozen: dict | None = None,
                 manifest: dict | None = None) -> int:
    if isinstance(train_set, ds.MultiBinpackDataset):
        n_games = 0
        print(f"data: {len(train_set):,} train records from "
              f"{len(train_set.datasets)} manifest file(s) / "
              f"{len(val_set):,} frozen val records")
    else:
        n_games = train_set.game_index_map[-1] + 1 if len(train_set.game_index_map) else 0
        print(f"data: {len(train_set):,} train / {len(val_set):,} val records, "
              f"{n_games} games ({len(train_set.validation_games)} held out)")

    loader_kwargs = {"num_workers": args.num_workers,
                     "pin_memory": device.type == "cuda",
                     "collate_fn": collate_batch}
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, **loader_kwargs)

    target = getattr(args, "target", None)
    config = nnue.ModelConfig(wdl_temperature=args.wdl_temperature,
                              format_version=2 if target else 1,
                              output_gain=target["output_gain"] if target else 1)
    model = nnue.NNUEModel(seed=args.seed, config=config).to(device)
    optimiser = make_optimizer(model, args.lr, args.hidden_lr)
    history: list[dict] = []
    data_identity = file_identity(args.data)
    frozen_identity = (file_identity(args.frozen_validation)
                       if args.frozen_validation is not None else None)
    manifest_identity = (file_identity(args.train_manifest)
                         if args.train_manifest is not None else None)
    train_files = ([p["path"] for p in manifest["files"]] if manifest is not None
                   else [str(Path(args.data).resolve())])
    best_val, best_epoch = float("inf"), 0
    steps = 0
    samples_exposed = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.history.parent.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(1, args.epochs + 1), desc="epoch", unit="epoch"):
        model.train()
        started = time.perf_counter()
        total, mse_sum, ce_sum, count = 0.0, 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False,
                    unit="batch")
        for x, stm, score, wdl in pbar:
            x = x.float().to(device)
            stm = stm.long().to(device)
            score = score.float().to(device)
            wdl = wdl.long().to(device)

            eval_score = model(x, stm, quantized=args.quant_forward)
            primary, mse, ce = training_losses(model, eval_score, score, wdl, args)
            regularizer = model.quantization_regularizer()
            loss = primary + args.quant_reg * regularizer
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            batch = len(x)
            total += float(primary) * batch
            mse_sum += float(mse) * batch
            ce_sum += float(ce) * batch
            count += batch
            steps += 1
            samples_exposed += batch
            pbar.set_postfix(loss=f"{total / max(1, count):.4f}", steps=steps)
            if args.max_steps is not None and steps >= args.max_steps:
                break

        val = evaluate(model, val_loader, device, args)
        row = {
            "epoch": epoch,
            "train_loss": total / max(1, count),
            "train_mse": mse_sum / max(1, count),
            "train_ce": ce_sum / max(1, count),
            "seconds": round(time.perf_counter() - started, 3),
            "optimizer_steps": steps,
            "samples_exposed": samples_exposed,
            **val,
        }
        history.append(row)
        print(f"epoch {epoch:3d} | train {row['train_loss']:.5f} "
              f"(mse {row['train_mse']:.5f} ce {row['train_ce']:.5f}) | "
              f"val {row['val_loss']:.5f} "
              f"(mse {row['val_mse']:.5f} ce {row['val_ce']:.5f}) | "
              f"{row['seconds']:.1f}s")

        improved = row["val_loss"] < best_val
        if improved:
            best_val, best_epoch = row["val_loss"], epoch
        if improved or args.checkpoint_policy == "last":
            torch.save({
                "model_state": model.state_dict(),
                "config": vars(model.config),
                "epoch": epoch,
                "val_loss": row["val_loss"],
                "history": history,
                "train_args": vars(args),
                "device": info,
                "data_identity": data_identity,
                "frozen_validation_identity": frozen_identity,
                "train_manifest_identity": manifest_identity,
                "target_contract": target,
                "target_contract_identity": file_identity(args.target_contract) if target else None,
            }, args.output)

        if args.max_steps is not None and steps >= args.max_steps:
            break

    report = {
        "complete": args.max_steps is None or steps == args.max_steps,
        "target_contract": target,
        "target_contract_identity": file_identity(args.target_contract) if target else None,
        "data": str(Path(args.data).resolve()),
        "data_identity": data_identity,
        "device": info,
        "config": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "max_steps": args.max_steps,
            "checkpoint_policy": args.checkpoint_policy,
            "hidden_lr": args.hidden_lr,
            "wdl_weight": args.wdl_weight,
            "wdl_temperature": args.wdl_temperature,
            "quant_reg": args.quant_reg,
            "val_fraction": args.val_fraction, "seed": args.seed,
            "quant_forward": args.quant_forward, "limit": args.limit,
            "score_clip": args.score_clip,
            "frozen_validation": (str(args.frozen_validation)
                                  if args.frozen_validation else None),
            "train_manifest": (str(args.train_manifest)
                               if args.train_manifest else None),
        },
        "dataset": {
            "train_records": len(train_set), "val_records": len(val_set),
            "games": n_games,
            "validation_games": sorted(val_set.validation_games),
            "train_files": train_files,
            "frozen_validation_identity": frozen_identity,
            "train_manifest_identity": manifest_identity,
        },
        "optimisation": {
            "optimizer_steps": steps,
            "samples_exposed": samples_exposed,
            "cap_reached": (args.max_steps is not None
                            and steps >= args.max_steps),
        },
        "epochs": history,
        "loss_curve": [row["val_loss"] for row in history],
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "converged": bool(history and history[-1]["val_loss"] < history[0]["val_loss"]),
        "checkpoint": str(args.output.resolve()),
    }
    args.history.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    if args.max_steps is not None and steps < args.max_steps:
        raise SystemExit(
            f"--max-steps {args.max_steps} not reached ({steps} steps in "
            f"{len(history)} epochs); increase --epochs or lower the cap")
    print(f"best val loss {best_val:.5f} @ epoch {best_epoch}; "
          f"converged={report['converged']}")
    print(f"checkpoint: {args.output}")
    print(f"history:    {args.history}")
    return 0


def build_datasets(args: argparse.Namespace):
    """(train_set, val_set, frozen, manifest) honouring the ticket-13 split."""
    frozen = manifest = None
    if args.frozen_validation is not None:
        frozen = vf.load_frozen(args.frozen_validation)
        vf.verify_frozen_data(frozen, args.data)
    if args.train_manifest is not None:
        if frozen is None:
            raise SystemExit("--train-manifest requires --frozen-validation")
        manifest = json.loads(args.train_manifest.read_text(encoding="utf-8"))
        if manifest.get("schema") not in (1, 2):
            raise SystemExit(f"train manifest schema "
                             f"{manifest.get('schema')!r} != {vf.SCHEMA}")
        frozen_sha = file_identity(args.frozen_validation)["sha256"]
        if manifest.get("frozen", {}).get("sha256") != frozen_sha:
            raise SystemExit("train manifest was not generated for this "
                             "frozen validation file; regenerate it with "
                             "validation_freeze.py check")

    contract = getattr(args, "target", None)
    store = None
    if contract:
        from teacher_labels import LabelStore
        if frozen is None or manifest is None or frozen.get("schema") != 2:
            raise ValueError("linear targets require derived freeze and train manifest")
        if file_identity(args.target_contract)["sha256"] not in [v["sha256"] for v in frozen["target_contracts"].values()]:
            raise ValueError("target contract is not bound to this freeze")
        store = LabelStore(contract["teacher_manifest"]["path"])
        if targets.pool_identity(manifest["files"], store) != contract["train_pool_sha256"]:
            raise ValueError("training pool differs from target contract")
    elif frozen and frozen.get("schema") == 2:
        raise ValueError("derived freeze requires --target-contract")
    extra = {"target_contract": contract, "label_store": store}

    if manifest is not None:
        parts = []
        for entry in manifest["files"]:
            include = set(entry["games"])
            if not include:
                raise SystemExit(f"manifest entry {entry['path']} keeps no "
                                 f"games; refusing an empty training file")
            parts.append(ds.BinpackDataset(
                entry["path"], "train", validation_games=set(),
                augment=True, max_records=args.limit,
                score_clip=args.score_clip, include_games=include, **extra))
        train_set: object = ds.MultiBinpackDataset(parts)
        val_games = set(frozen["split"]["validation_games"])
    elif frozen is not None:
        val_games = set(frozen["split"]["validation_games"])
        train_set = ds.BinpackDataset(
            args.data, "train", validation_games=val_games,
            exclude_games=set(frozen["leakage"]["excluded_train_games"]),
            augment=True, max_records=args.limit, score_clip=args.score_clip)
    else:
        train_set = ds.BinpackDataset(
            args.data, "train", args.val_fraction, args.seed,
            augment=True, max_records=args.limit, score_clip=args.score_clip)
        # Without a frozen split the validation games MUST come from the same
        # draw as the train set, otherwise the two datasets split independently
        # and can overlap (regression caught by m5_tests).
        val_games = train_set.validation_games

    val_set = ds.BinpackDataset(
        args.data, "val", validation_games=val_games,
        augment=False, max_records=args.limit, score_clip=args.score_clip, **extra)
    return train_set, val_set, frozen, manifest


def assert_split_disjoint(train_set, val_set: ds.BinpackDataset) -> None:
    """Game-id check within one file; cross-file separation is guaranteed by
    the D4 scan in validation_freeze.py, not by game numbering."""
    parts = (train_set.datasets if isinstance(train_set, ds.MultiBinpackDataset)
             else [train_set])
    for part in parts:
        if part.path.resolve() == val_set.path.resolve():
            overlap = part.game_ids & val_set.game_ids
            if overlap:
                raise ValueError(
                    f"game-level leakage: {len(overlap)} game(s) of "
                    f"{part.path} appear in both splits")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.target = targets.load(args.target_contract) if args.target_contract else None
    if args.target and (args.score_clip != 0 or args.wdl_weight != 0 or args.quant_reg != 0):
        raise ValueError("linear targets require score_clip/wdl_weight/quant_reg = 0")
    if args.target and args.max_steps == 1600 and args.limit is None:
        if args.frozen_validation is None:
            raise ValueError("linear training requires --frozen-validation")
        from distill18 import verify_plan, read
        from teacher_labels import verify_identity
        directory = args.frozen_validation.parent
        plan, _ = verify_plan(directory)
        stage = read(directory / "stage_a.json")
        if stage.get("passed") is not True or stage["plan"]["sha256"] != file_identity(directory / "experiment_plan.json")["sha256"]:
            raise ValueError("stage A has not passed; training blocked")
        verify_identity(stage["baselines"])
        verify_identity(stage["preflight"])
        if any(getattr(args, k) != v for k, v in plan["hyperparameters"].items()):
            raise ValueError("training hyperparameters differ from sealed plan")
    if any(v is not None and (not math.isfinite(v) or v <= 0)
           for v in (args.lr, args.hidden_lr)):
        raise SystemExit("learning rates must be finite and positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    info = device_info(device)
    print(f"device: {info['device']} | torch {info['torch']} | "
          f"CUDA {info['cuda_available']}"
          + (f" | {info['cuda_name']}" if info["cuda_name"] else ""))
    if args.score_clip > 0:
        print(f"score_clip: +/-{args.score_clip:g} (saturated search scores "
              f"clipped before the tanh loss)")

    train_set, val_set, frozen, manifest = build_datasets(args)
    if frozen is not None:
        print(f"frozen validation: {args.frozen_validation} "
              f"({frozen['split']['validation_records']} records, "
              f"{len(frozen['split']['validation_games'])} games)")
    if manifest is not None:
        print(f"train manifest: {args.train_manifest} "
              f"({len(manifest['files'])} files)")
    try:
        assert_split_disjoint(train_set, val_set)
        return run_training(args, train_set, val_set, device, info,
                            frozen=frozen, manifest=manifest)
    finally:
        # Windows cannot delete a file while an mmap handle is alive.
        train_set.close()
        val_set.close()


if __name__ == "__main__":
    raise SystemExit(main())
