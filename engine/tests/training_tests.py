#!/usr/bin/env python3
"""Ticket 08 tests: feature/D4 spec, weight format, game-level split, training.

The standard-library half always runs (no torch/numpy needed): it validates the
cross-language contracts used by ticket 09. The torch half runs automatically
once the training dependencies are installed and is reported as skipped
otherwise.

Usage::

  .venv\\Scripts\\python.exe engine\\tests\\training_tests.py
"""
from __future__ import annotations

import importlib.util
import json
import random
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "trainer"))

import binpack  # noqa: E402
import dataset  # noqa: E402
import features  # noqa: E402
import weights_format as wf  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)
        print(f"FAIL: {message}")


def has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def write_synthetic(path: Path, games: int = 3, per_game: int = 3) -> None:
    """A tiny but structurally valid binpack file, built with the Python spec."""
    records = []
    for game in range(games):
        draw = game == 1
        winner = 0  # black wins the decisive games
        for r in range(per_game):
            ply = 3 + r
            stm = ply & 1
            wdl = 1 if draw else (2 if stm == winner else 0)
            black = bytearray(binpack.PLANE_BYTES)
            white = bytearray(binpack.PLANE_BYTES)
            for i in range(ply):
                cell = 1 + i * 7
                plane = black if i % 2 == 0 else white
                plane[cell >> 3] |= 1 << (cell & 7)
            flags = (wdl | (stm << 2)
                     | (binpack.GAME_START_BIT if r == 0 else 0)
                     | (binpack.GAME_END_BIT if r == per_game - 1 else 0))
            raw = bytearray(binpack.RECORD_SIZE)
            raw[0:binpack.PLANE_BYTES] = black
            raw[binpack.PLANE_BYTES:2 * binpack.PLANE_BYTES] = white
            struct.pack_into("<hBBH", raw, binpack.SCORE_OFFSET,
                             100 * game + 10 * r - 50, flags, ply, 200)
            records.append(bytes(raw))
    header = binpack.pack_header(binpack.Header(
        record_count=games * per_game, game_count=games,
        search_depth=4, shard_id=0, base_seed=1))
    path.write_bytes(header + b"".join(records))


def test_features() -> None:
    check(len(features.PERMUTATIONS) == 8, "eight D4 transforms")
    check(len(set(features.PERMUTATIONS)) == 8, "D4 transforms are distinct")
    check(features.PERMUTATIONS[0] == tuple(range(225)), "transform 0 is identity")
    for t, perm in enumerate(features.PERMUTATIONS):
        check(sorted(perm) == list(range(225)), f"transform {t} is a bijection")
    for t in range(8):
        for u in range(8):
            composed = tuple(features.PERMUTATIONS[t][features.PERMUTATIONS[u][c]]
                             for c in range(225))
            check(composed in features.PERMUTATIONS,
                  f"D4 closure for transforms {t},{u}")

    check(features.feature_index_of_color(7 * 15 + 7, 2) == 337,
          "white stone at (7,7) -> feature 337")
    check(features.split_feature(337) == (1, 112), "split_feature is the inverse")
    check(features.sparse_indices([1, 8], [15]) == [1, 8, 240],
          "sparse_indices assigns black/white planes")

    sample = [1, 8, 15, 240, 337]
    transformed = [tuple(features.transform_sparse(sample, t)) for t in range(8)]
    check(len(set(transformed)) == 8, "asymmetric position has 8 distinct D4 images")
    for indices in transformed:
        check(len(indices) == len(sample), "D4 preserves stone count")
        check(all(0 <= i < 450 for i in indices), "D4 stays inside 450 dims")
        check(len([i for i in indices if i < 225]) ==
              len([i for i in sample if i < 225]), "D4 preserves colour partition")

    tables = features.symmetry_table()
    original = features.dense_vector(sample)
    for t in range(8):
        augmented = features.dense_vector(features.transform_sparse(sample, t))
        check(all(augmented[i] == original[tables[t][i]] for i in range(450)),
              f"symmetry_table matches transform {t}")

    # Renju's five-in-a-row geometry is D4-invariant.
    def is_five_line(cells: list[int]) -> bool:
        points = {(c % 15, c // 15) for c in cells}
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            for cell in cells:
                x0, y0 = cell % 15, cell // 15
                if all((x0 + k * dx, y0 + k * dy) in points for k in range(5)):
                    return True
        return False

    five = [7 * 15 + x for x in range(3, 8)]
    for t in range(8):
        check(is_five_line([features.transform_cell(c, t) for c in five]),
              f"five-in-a-row stays a line under transform {t}")


def make_small_weights() -> dict:
    weights = {
        "W0": [0] * wf.W0_LEN, "b0": [0] * wf.B0_LEN,
        "W1": [0] * wf.W1_LEN, "b1": [0] * wf.B1_LEN,
        "W2": [0] * wf.W2_LEN, "b2": [0] * wf.B2_LEN,
    }
    weights["W0"][112 * wf.N_ACC + 0] = 5
    weights["b0"][0] = 3
    weights["W1"][0 * wf.N_L1 + 0] = 7
    weights["b1"][0] = 1
    weights["W2"][0 * wf.N_L1 + 0] = 11
    weights["b2"][0] = 2
    return weights


def test_round_div() -> None:
    check([wf.round_div(n, 127) for n in (-191, -64, -63, 0, 63, 64, 191)]
          == [-2, -1, 0, 0, 0, 1, 2], "round_div rounds half away from zero")
    check(wf.round_div(-1, 64) == 0 and wf.round_div(-33, 64) == -1,
          "round_div differs from C++ truncation on negatives")


def test_weight_format(tmp: Path) -> None:
    weights = make_small_weights()
    blob = wf.pack_weights(weights)
    check(len(blob) == wf.FILE_LEN, f"weight file is {wf.FILE_LEN} bytes")
    check(blob[:8] == wf.MAGIC, "weight magic")
    decoded, meta = wf.unpack_weights(blob)
    check(meta["version"] == 1 and meta["qa"] == wf.QA and meta["qb"] == wf.QB,
          "quantisation constants in header")
    check(list(decoded["W0"])[112 * wf.N_ACC] == 5, "W0 round-trips")

    # Hand-computed integer forward:
    # acc0 = 3+5 = 8; z0 = 8*7 = 56; zq0 = round_div(56,127)+1 = 1;
    # y = 1*11 = 11; eval = round_div(11,64)+2 = 2.
    indices = [112]
    check(wf.infer_int(decoded, indices, 0) == 2, "integer reference forward")
    check(wf._infer_python(decoded, indices, 0, wf.QA, wf.QB) == 2,
          "pure-Python reference forward")
    if wf._np is not None:
        check(wf._infer_numpy(decoded, indices, 0, wf.QA, wf.QB) == 2,
              "numpy reference forward agrees")

    def rejects(data: bytes, label: str) -> bool:
        try:
            wf.unpack_weights(data)
        except wf.WeightFormatError:
            return True
        return False

    bad = bytearray(blob)
    bad[0] = ord("X")
    check(rejects(bytes(bad), "magic"), "bad magic is rejected")
    bad = bytearray(blob)
    bad[8] = 9
    check(rejects(bytes(bad), "version"), "unsupported version is rejected")
    bad = bytearray(blob)
    bad[wf.HEADER_SIZE + 10] ^= 1
    check(rejects(bytes(bad), "checksum"), "checksum mismatch is rejected")
    bad = bytearray(blob)
    bad[wf.HEADER_STRUCT.size] = 1
    check(rejects(bytes(bad), "reserved"), "non-zero reserved bytes are rejected")
    bad = bytearray(blob)
    bad[16] = 8
    check(rejects(bytes(bad), "dims"), "wrong dimensions are rejected")

    out_of_range = dict(weights)
    out_of_range["W0"] = list(weights["W0"])
    out_of_range["W0"][0] = wf.WEIGHT_CLAMP + 1
    try:
        wf.pack_weights(out_of_range)
        check(False, "out-of-range int8 weight is rejected")
    except wf.WeightFormatError:
        pass

    path = tmp / "tiny.nnue"
    wf.write_weights(path, weights)
    reloaded, _ = wf.read_weights(path)
    check(wf.infer_int(reloaded, indices, 0) == 2, "file round-trip replay")


def test_game_split(tmp: Path) -> None:
    path = tmp / "synthetic.binpack"
    write_synthetic(path)
    stats = binpack.validate_file(path)
    check(not stats["errors"], f"synthetic binpack is valid: {stats['errors'][:3]}")
    check(stats["games"] == 3, "synthetic binpack has 3 games")

    val_a = dataset.choose_validation_games(10, 0.2, 7)
    val_b = dataset.choose_validation_games(10, 0.2, 7)
    check(val_a == val_b and len(val_a) == 2, "validation split is deterministic")
    check(dataset.choose_validation_games(10, 0.0, 7) == set(),
          "zero val_fraction holds out nothing")
    check(1 <= len(dataset.choose_validation_games(2, 0.9, 7)) <= 1,
          "validation never swallows the entire dataset")

    from array import array
    mapping = array("I", [0, 0, 0, 1, 1, 2, 2, 2])
    held = {1}
    train = dataset.split_record_indices(mapping, held, "train")
    val = dataset.split_record_indices(mapping, held, "val")
    check(list(train) == [0, 1, 2, 5, 6, 7] and list(val) == [3, 4],
          "every record of a game stays in one split")

    train_set = dataset.BinpackDataset(path, "train", 0.34, 7, augment=True)
    val_set = dataset.BinpackDataset(path, "val", 0.34, 7, augment=False)
    try:
        dataset.assert_disjoint(train_set, val_set)
        check(True, "no game-level leakage")
    except ValueError as exc:
        check(False, f"no game-level leakage: {exc}")
    check(len(train_set) + len(val_set) == stats["records"],
          "splits cover every record exactly once")
    random.seed(11)
    vector, stm, score, wdl = train_set[0]
    check(len(vector) == 450 and sum(vector) > 0, "sample is a 450-dim 0/1 vector")
    check(all(v in (0.0, 1.0) for v in vector), "sample contains only 0/1")
    check(stm in (0, 1) and wdl in (0, 1, 2) and isinstance(score, float),
          "labels are well formed")
    check(sum(train_set[0][0]) == sum(vector),
          "D4 augmentation preserves stone count")
    train_set.close()
    val_set.close()


def test_score_clip(tmp: Path) -> None:
    """Optional saturation clipping (ticket 10): default behaviour unchanged."""
    check(dataset.clip_score(32767, 0) == 32767,
          "score_clip=0 keeps saturated labels (ticket-08 behaviour)")
    check(dataset.clip_score(32767, 600) == 600,
          "positive saturated label is clipped")
    check(dataset.clip_score(-32767, 600) == -600,
          "negative saturated label is clipped")
    check(dataset.clip_score(120, 600) == 120,
          "ordinary label is untouched")
    check(dataset.clip_score(32767, -1) == 32767,
          "negative clip value disables clipping")

    raw = bytearray(binpack.RECORD_SIZE)
    raw[0] = 0x01  # one black stone at cell 0 -> ply 1
    flags = binpack.GAME_START_BIT | binpack.GAME_END_BIT
    struct.pack_into("<hBBH", raw, binpack.SCORE_OFFSET, 32767, flags, 1, 7)
    header = binpack.pack_header(binpack.Header(
        record_count=1, game_count=1, search_depth=4, shard_id=0,
        base_seed=1))
    path = tmp / "saturated.binpack"
    path.write_bytes(header + bytes(raw))

    plain = dataset.BinpackDataset(path, "train", 0.0, 7, augment=False)
    clipped = dataset.BinpackDataset(path, "train", 0.0, 7, augment=False,
                                     score_clip=600)
    try:
        check(plain[0][2] == 32767.0, "default dataset keeps the raw score")
        check(clipped[0][2] == 600.0, "dataset applies score_clip to samples")
        check(list(clipped.iter_samples())[0][2] == 600.0,
              "iter_samples applies the same clipping")
    finally:
        plain.close()
        clipped.close()


def test_torch(tmp: Path) -> None:
    import torch  # noqa: F401
    import nnue
    import train
    import export
    import weight_drift as wd

    synthetic = tmp / "torch.binpack"
    write_synthetic(synthetic, games=4, per_game=3)

    model = nnue.NNUEModel(seed=3)
    weights = model.quantized_weights()
    wf.normalise_weights(weights)
    indices, stm = [112, 337, 8], 1
    x = torch.tensor([features.dense_vector(indices)], dtype=torch.float32)
    s = torch.tensor([stm], dtype=torch.long)
    with torch.no_grad():
        quant_eval = float(model(x, s, quantized=True)[0])
        float_eval = float(model(x, s, quantized=False)[0])
    reference = wf.infer_int(weights, indices, stm)
    check(quant_eval == reference,
          f"STE forward equals integer spec ({quant_eval} vs {reference})")
    check(abs(float_eval - reference) < 20,
          f"float forward is close to integer spec ({float_eval} vs {reference})")

    rebuilt = nnue.NNUEModel.from_quantized_weights(weights, model.config)
    with torch.no_grad():
        check(float(rebuilt(x, s, quantized=True)[0]) == reference,
              "from_quantized_weights replays the same integers")
    mse, ce = nnue.mixed_losses(model(x, s),
                                torch.tensor([100.0]), torch.tensor([2]))
    check(torch.isfinite(mse) and torch.isfinite(ce), "mixed losses are finite")

    checkpoint = tmp / "ckpt.pt"
    history_path = tmp / "history.json"
    train.main(["--data", str(synthetic), "--output", str(checkpoint),
                "--history", str(history_path), "--epochs", "2",
                "--batch-size", "4", "--limit", "6", "--device", "cpu"])
    history = json.loads(history_path.read_text(encoding="utf-8"))
    check(len(history["epochs"]) == 2, "training writes two epochs")
    check(all(row["val_loss"] == row["val_loss"]
              for row in history["epochs"]), "validation losses are finite")
    check(history["dataset"]["train_records"] > 0 and
          history["dataset"]["val_records"] > 0, "both splits have samples")
    check(len(history["loss_curve"]) == 2 and "converged" in history,
          "history carries a reviewable validation loss curve")
    check(checkpoint.exists(), "best checkpoint is written")

    clip_history = tmp / "history_clip.json"
    train.main(["--data", str(synthetic), "--output", str(tmp / "clip.pt"),
                "--history", str(clip_history), "--epochs", "1",
                "--batch-size", "4", "--limit", "6", "--device", "cpu",
                "--score-clip", "600"])
    clip_report = json.loads(clip_history.read_text(encoding="utf-8"))
    check(clip_report["config"]["score_clip"] == 600,
          "--score-clip reaches the dataset and is recorded in history")
    check(all(row["val_loss"] == row["val_loss"]
              for row in clip_report["epochs"]),
          "clipped-label training runs without NaNs")

    steps_history = tmp / "history_steps.json"
    train.main(["--data", str(synthetic), "--output", str(tmp / "steps.pt"),
                "--history", str(steps_history), "--epochs", "10",
                "--batch-size", "4", "--limit", "6", "--max-steps", "5",
                "--checkpoint-policy", "last", "--device", "cpu"])
    steps_report = json.loads(steps_history.read_text(encoding="utf-8"))
    check(steps_report["config"]["max_steps"] == 5,
          "ticket 15 step cap is recorded in the training config")
    check(steps_report["config"]["checkpoint_policy"] == "last",
          "checkpoint policy is recorded in the training config")
    steps_ckpt = torch.load(tmp / "steps.pt", map_location="cpu",
                            weights_only=False)
    check(steps_ckpt["epoch"] == len(steps_report["epochs"]),
          "'last' policy exports the final optimizer step, not a best-val pick")
    check(steps_report["optimisation"]["optimizer_steps"] == 5 and
          steps_report["optimisation"]["cap_reached"],
          "training stops at exactly the requested optimizer updates")
    check(len(steps_report["epochs"]) < 10,
          "step cap stops the epoch loop early")
    check(steps_report["epochs"][-1]["optimizer_steps"] == 5 and
          steps_report["epochs"][-1]["samples_exposed"] ==
          steps_report["optimisation"]["samples_exposed"],
          "per-epoch step/exposure counters agree with the final report")
    try:
        train.main(["--data", str(synthetic), "--output", str(tmp / "short.pt"),
                    "--history", str(tmp / "history_short.json"), "--epochs", "1",
                    "--batch-size", "4", "--limit", "6", "--max-steps", "100",
                    "--device", "cpu"])
    except SystemExit:
        pass
    else:
        check(False, "an unreachable step cap must fail loudly")

    drift_seed = 7
    initial_model = nnue.NNUEModel(seed=drift_seed)
    drift_state = {k: v.detach().clone()
                   for k, v in initial_model.state_dict().items()}
    drift_state["W2"] = drift_state["W2"] + 3.0
    fake_ckpt = tmp / "drift_fake.pt"
    torch.save({"config": vars(initial_model.config),
                "model_state": drift_state, "epoch": 1,
                "val_loss": 0.5, "train_args": {"lr": 0.1}}, fake_ckpt)
    drift = wd.drift_report(fake_ckpt, drift_seed)
    check(drift["layers"]["W2"]["max_abs_delta"] == 3.0,
          "ticket 17 weight drift detects an exact known parameter change")
    check(all(drift["layers"][k]["l2_delta"] == 0.0
              for k in drift["layers"] if k != "W2"),
          "rebuilt seed matches every untouched initial layer exactly")
    check(drift["groups"]["output_head"]["l2_delta"] > 0 and
          drift["groups"]["accumulator"]["l2_delta"] == 0.0,
          "layer groups aggregate deltas correctly")

    weight_file = tmp / "exported.nnue"
    report_file = tmp / "export_report.json"
    export.main(["--checkpoint", str(checkpoint), "--data", str(synthetic),
                 "--output", str(weight_file), "--report", str(report_file),
                 "--samples", "6"])
    report = json.loads(report_file.read_text(encoding="utf-8"))
    check(report["self_check"]["read_back_replay_matches"],
          "exported weights read back and replay identically")
    check(report["self_check"]["file_size"] == wf.FILE_LEN,
          "exported weight file has the documented size")
    check(report["parity"]["float_vs_integer"]["mae"] < 50,
          "float/int parity error is small")
    check(weight_file.read_bytes()[:8] == wf.MAGIC, "exported file has the magic")
    final_weights, _ = wf.read_weights(weight_file)
    final_model = nnue.NNUEModel.from_quantized_weights(final_weights, model.config)
    with torch.no_grad():
        check(float(final_model(x, s, quantized=True)[0]) ==
              wf.infer_int(final_weights, indices, stm),
              "exported model still matches the integer spec")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="gomoku_training_",
                                    ignore_cleanup_errors=True) as raw_tmp:
        tmp = Path(raw_tmp)
        test_round_div()
        test_features()
        test_weight_format(tmp)
        test_game_split(tmp)
        test_score_clip(tmp)
        if has_torch():
            print("torch detected: running model / train / export tests")
            test_torch(tmp)
        else:
            print("torch not installed: model/train/export tests SKIPPED "
                  "(install trainer/requirements.txt to enable)")

    if FAILURES:
        print(f"\n{len(FAILURES)} training test(s) failed")
        return 1
    print("training_tests: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
