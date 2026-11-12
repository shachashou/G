#!/usr/bin/env python3
"""Value-network training/export tests (ticket 06).

Covers the contract rather than strength: architecture, target conversion,
D4 isolation, a real (small) training run on a committed pool, checkpoint
resume, the GMVALUE1 format and its rejection paths, and the frozen fidelity
tolerances.

Usage: python value_train_tests.py [tools dir]
"""

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "trainer"))

import torch  # noqa: E402

import value_data  # noqa: E402
import value_format as vf  # noqa: E402
import value_train  # noqa: E402
from value_fixture import build_pool, make_board, record  # noqa: E402
from value_model import ValueConfig, ValueNet, value_loss  # noqa: E402
from sample_pool import SamplePool  # noqa: E402

failures = 0


def check(cond, msg):
    global failures
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures += 1


def evaluate(model, samples):
    model.eval()
    with torch.no_grad():
        x, y = value_data.tensors(samples)
        prediction = model(x)
        return (float(value_loss(prediction, y, "mse")),
                float(torch.abs(prediction - y).mean()))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="value-train-tests-"))
    try:
        return run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(tmp: Path) -> int:
    pool_path = build_pool(tmp / "pool.sqlite3")

    print("[model architecture]")
    model = ValueNet(seed=1, config=ValueConfig())
    shapes = {name: tuple(t.shape) for name, t in model.named_parameters()}
    check(shapes["W0"] == (450, 1024) and shapes["b0"] == (1024,),
          "first layer is 450 -> 1024")
    check(shapes["W1"] == (1024, 64) and shapes["W2"] == (64, 32),
          "hidden layers are 1024 -> 64 -> 32")
    check(shapes["W3"] == (32, 1) and shapes["b3"] == (1,),
          "single output head, no side-to-move head")
    check(model.parameter_count() == 529537,
          f"parameter count is the fixed architecture ({model.parameter_count()})")
    x = torch.zeros(3, 450)
    with torch.no_grad():
        values = model(x)
    check(tuple(values.shape) == (3,), "forward returns one value per position")
    check(bool((values.abs() <= 1.0).all()), "value stays inside [-1, 1]")

    print("[target contract]")
    cells, side = make_board(__import__("random").Random(1), 8)
    teacher = record(cells, "white")
    if teacher["side_to_move"] != "white":
        teacher["side_to_move"] = "white"
    target = value_data.record_target(teacher)
    check(abs(target + teacher["value"]) < 1e-12,
          "white-to-move teacher value is negated into black-absolute target")
    teacher_black = dict(teacher, side_to_move="black")
    check(abs(value_data.record_target(teacher_black) - teacher["value"]) < 1e-12,
          "black-to-move teacher value is kept")
    result = dict(teacher, label_source="selfplay_result", result_value=-0.5)
    check(value_data.record_target(result) == -0.5,
          "self-play result is already black-absolute")
    try:
        value_data.record_target(dict(teacher, label_source="none"))
        check(False, "sample without supervision is rejected")
    except ValueError:
        check(True, "sample without supervision is rejected")

    print("[D4 isolation]")
    bundle = value_data.load_pool(pool_path)
    check(bundle.identity["train_samples"] > 0
          and bundle.identity["validation_samples"] > 0,
          "pool split loads train and validation samples")
    original = bundle.train[0]
    rotated = list(original.cells)
    perm = __import__("features").PERMUTATIONS[1]
    turned = [0] * 225
    for cell, value in enumerate(rotated):
        turned[perm[cell]] = value
    check(value_data.canonical_key(original.cells, original.side)
          == value_data.canonical_key(turned, original.side),
          "D4 rotation keeps the canonical position identity")
    twin = value_data.Sample(cells=tuple(turned), side=original.side,
                             target=original.target,
                             source_id="leak", label_source="teacher",
                             label_version="teacher-v1")
    try:
        value_data.check_d4_isolation([twin], [original])
        check(False, "D4-equivalent training sample is rejected")
    except ValueError:
        check(True, "D4-equivalent training sample is rejected")

    print("[real training run]")
    base = ValueNet(seed=5, config=ValueConfig())
    initial_loss, initial_mae = evaluate(base, bundle.validation)
    config = value_train.TrainConfig(
        pool=str(pool_path), output_dir=str(tmp / "run"),
        seed=5, steps=40, batch_size=8, lr=3e-3, loss="mse", device="cpu",
        val_every=10, checkpoint_every=20, log_every=10)
    summary = value_train.train_value(config)
    check(summary["steps"] == 40, "training ran the requested steps")
    check(summary["stopped"] is False, "training completed without a stop")
    check(summary["train_loss"] is not None
          and summary["train_loss"] == summary["train_loss"],
          "training loss is finite")
    check(summary["validation"]["samples"] == len(bundle.validation),
          "validation covered the frozen validation split")
    check(summary["validation"]["mae"] < initial_mae,
          f"validation MAE improved ({initial_mae:.3f} -> "
          f"{summary['validation']['mae']:.3f})")
    check(Path(summary["checkpoint"]["last"]).is_file(),
          "last checkpoint exists")
    check(Path(summary["checkpoint"]["best"]).is_file(),
          "best checkpoint exists")
    history = Path(summary["history"]).read_text(encoding="utf-8").splitlines()
    check(len(history) > 0, "training history was written")
    check(summary["data_identity"]["pool"].endswith("pool.sqlite3"),
          "summary records the data identity")

    print("[export format and fidelity]")
    export = summary["export"]
    check(export["passed"] is True, "fidelity passed the frozen tolerances")
    model_path = Path(export["model"]["path"])
    check(model_path.is_file() and model_path.stat().st_size > 1_000_000,
          "GMVALUE1 model file was written")
    weights, metadata = vf.read_value_weights(model_path)
    check(metadata["arch"] == [450, 1024, 64, 32, 1],
          "export declares the 450-1024-64-32-1 architecture")
    check(metadata["rule"] == "renju"
          and metadata["feature_spec"] == "absolute-color-450"
          and metadata["value_contract"] == "black-absolute-tanh-v1",
          "export declares rule, features and value contract")
    check(metadata["training"]["data_identity"]["train_samples"]
          == bundle.identity["train_samples"],
          "export records the training data identity")
    check(isinstance(vf.infer_int(weights, bundle.validation[0].sparse()), int),
          "integer reference forward returns a fixed-point score")
    report = json.loads(Path(export["report"]).read_text(encoding="utf-8"))
    fidelity = report["fidelity"]
    check(fidelity["value_tolerance"] == vf.VALUE_TOLERANCE,
          "report records the frozen value tolerance")
    check(fidelity["export_pass"] and fidelity["integer_pass"],
          "report marks export and integer fidelity as passed")
    check(fidelity["float_vs_integer"]["mean_abs"] <= vf.VALUE_TOLERANCE["mean_abs"],
          "float-vs-integer mean error is within tolerance")
    check(fidelity["quantised_float_vs_integer"]["max_abs"]
          <= vf.INTEGER_TOLERANCE["max_abs"],
          "quantised float and integer reference agree")

    print("[checkpoint resume]")
    resumed = value_train.TrainConfig(
        pool=str(pool_path), output_dir=str(tmp / "resume"),
        seed=5, steps=60, batch_size=8, lr=3e-3, loss="mse", device="cpu",
        val_every=10, checkpoint_every=20, log_every=20,
        resume=summary["checkpoint"]["last"])
    resumed_summary = value_train.train_value(resumed)
    check(resumed_summary["start_step"] == 40,
          "resume starts from the saved step")
    check(resumed_summary["steps"] == 60, "resume continues to the new target")
    check(resumed_summary["resumed_from"].endswith("last.pt"),
          "summary records the checkpoint it resumed from")

    print("[format rejection]")
    old = b"GMNNUE01" + b"\x00" * 56
    try:
        vf.unpack_value_weights(old)
        check(False, "old GMNNUE01 file is rejected")
    except vf.ValueFormatError:
        check(True, "old GMNNUE01 file is rejected")
    blob = bytearray(model_path.read_bytes())
    blob[-1] ^= 0xFF
    try:
        vf.unpack_value_weights(bytes(blob))
        check(False, "corrupted body checksum is rejected")
    except vf.ValueFormatError:
        check(True, "corrupted body checksum is rejected")
    bad_meta = copy.deepcopy(metadata)
    bad_meta["arch"] = [450, 1024, 15, 2]
    try:
        vf.pack_value_weights(weights, bad_meta)
        check(False, "metadata with the old architecture is rejected")
    except vf.ValueFormatError:
        check(True, "metadata with the old architecture is rejected")
    checkpoint = torch.load(summary["checkpoint"]["last"],
                            map_location="cpu", weights_only=False)
    checkpoint["arch"] = [450, 1024, 15, 2]
    bad_path = tmp / "bad_checkpoint.pt"
    torch.save(checkpoint, bad_path)
    try:
        value_train._load_checkpoint(bad_path, config)
        check(False, "checkpoint with the old architecture is rejected")
    except ValueError:
        check(True, "checkpoint with the old architecture is rejected")

    print("[pool without validation]")
    train_only = build_pool(tmp / "no-val.sqlite3", validation_games=0)
    try:
        value_data.load_pool(train_only)
        check(False, "pool without validation samples is refused")
    except ValueError as exc:
        check("validation" in str(exc),
              "pool without validation samples is refused")

    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
