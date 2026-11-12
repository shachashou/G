#!/usr/bin/env python3
"""Ticket 13 regression tests: frozen, D4-leak-free validation split.

Standard-library half (always runs):
  * canonical position key is D4-invariant, colour/stm-sensitive
  * freeze detects mirror overlaps, excludes WHOLE train games, keeps the
    validation set itself unchanged, records baselines and identities
  * check handles cross-file game-number reuse via positions, and emits a
    train manifest whose include-lists never contain validation games
  * tampered data files are rejected against the frozen identity

Torch half (when installed): train.py honours --frozen-validation and
--train-manifest; extra files never change the frozen validation split.

Fixture note: with --val-fraction 0.5 --seed 42 over 4 games, the split is
deterministically {1, 2} (verified via dataset.choose_validation_games), so
the bridge pair X / mirror(X) is placed in games 0 (train) and 1 (val).
"""

import importlib.util
import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "trainer"), str(ROOT / "tools")]

import binpack  # noqa: E402
import features  # noqa: E402
import validation_freeze as vf  # noqa: E402
from dataset import choose_validation_games  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)
        print(f"FAIL: {message}")


def has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


# Positions are (black_cells, white_cells); ply parity fixes the side to move.
X = ([112], [113])
Y = ([3, 19], [34])
Z = ([130, 145], [125])
V = ([30, 31], [32])
V2 = ([45], [60])
V3 = ([90], [91])
V4 = ([92, 90], [93])
W = ([70], [71])
Y2 = ([12, 10], [13])
Z2 = ([53], [54])
Y3 = ([14], [15, 11])
Z3 = ([55, 50], [56])

DISTINCT = {"X": X, "Y": Y, "Z": Z, "V": V, "V2": V2, "V3": V3, "V4": V4,
            "W": W, "Y2": Y2, "Z2": Z2, "Y3": Y3, "Z3": Z3}


def transform_position(position, t: int):
    black, white = position
    indices = features.transform_sparse(
        list(black) + [225 + c for c in white], t)
    return ([i for i in indices if i < 225],
            [i - 225 for i in indices if i >= 225])


def position_indices(position) -> list[int]:
    black, white = position
    return list(black) + [225 + c for c in white]


def write_binpack(path: Path, games) -> None:
    body = bytearray()
    n_records = 0
    for game_records in games:
        for i, (black, white, score, wdl) in enumerate(game_records):
            raw = bytearray(binpack.RECORD_SIZE)

            def pack_plane(cells) -> bytes:
                plane = bytearray(binpack.PLANE_BYTES)
                for c in cells:
                    plane[c >> 3] |= 1 << (c & 7)
                return bytes(plane)

            raw[0:binpack.PLANE_BYTES] = pack_plane(black)
            raw[binpack.PLANE_BYTES:2 * binpack.PLANE_BYTES] = pack_plane(white)
            ply = len(black) + len(white)
            stm = ply & 1
            flags = wdl | (stm << 2)
            if i == 0:
                flags |= binpack.GAME_START_BIT
            if i == len(game_records) - 1:
                flags |= binpack.GAME_END_BIT
            occupied = set(black) | set(white)
            move = next(c for c in range(225) if c not in occupied)
            struct.pack_into("<hBBH", raw, binpack.SCORE_OFFSET, score, flags,
                             ply, move)
            body += raw
            n_records += 1
    header = binpack.pack_header(binpack.Header(
        record_count=n_records, game_count=len(games), search_depth=4,
        shard_id=0, base_seed=1))
    path.write_bytes(header + bytes(body))


def game(positions, score=100, wdl=2):
    return [(p[0], p[1], score + 7 * i, wdl) for i, p in enumerate(positions)]


def build_files(tmp: Path):
    main = tmp / "main.binpack"
    write_binpack(main, [
        game([X, Y]),                          # game 0: bridge, train side
        game([transform_position(X, 5), Z]),    # game 1: bridge, val side
        game([V, V2]),                          # game 2: val side, clean
        game([V3, V4]),                         # game 3: train side, clean
    ])
    extra = tmp / "extra.binpack"
    write_binpack(extra, [
        game([transform_position(X, 5), W]),    # game 0: leaks vs val game 1
        game([Y2, Z2], score=-60, wdl=0),       # game 1: clean
    ])
    return main, extra


def test_position_construction() -> None:
    keys: dict[tuple, str] = {}
    for name, position in DISTINCT.items():
        stm = (len(position[0]) + len(position[1])) & 1
        key = vf.canonical_position(position_indices(position), stm)
        check(key not in keys,
              f"fixtures {keys.get(key)} and {name} are D4-equivalent; "
              f"pick different positions")
        keys[key] = name
    bridge = vf.canonical_position(position_indices(transform_position(X, 5)),
                                   0)
    check(bridge == vf.canonical_position(position_indices(X), 0),
          "the bridge pair X/mirror(X) is D4-equivalent")


def test_canonical_position() -> None:
    indices = position_indices(X)
    key = vf.canonical_position(indices, 0)
    for t in range(features.NUM_SYMMETRIES):
        check(vf.canonical_position(
            features.transform_sparse(indices, t), 0) == key,
              f"D4 transform {t} preserves the canonical key")
    check(vf.canonical_position(indices, 1) != key,
          "side to move is part of the canonical key")
    check(vf.canonical_position(position_indices((X[1], X[0])), 0) != key,
          "colour exchange changes the key")
    check(vf.find_transform(indices, tuple(sorted(indices))) == 0,
          "identity transform is found")
    check(vf.canonical_position(position_indices(Y), 0) != key,
          "different positions have different keys")


def test_freeze_and_check(tmp: Path) -> None:
    main, extra = build_files(tmp)
    check(sorted(choose_validation_games(4, 0.5, 42)) == [1, 2],
          "fixture premise: seed 42 splits games {1, 2} into validation")
    frozen_path = tmp / "frozen.json"
    rc = vf.main(["freeze", "--data", str(main), "--val-fraction", "0.5",
                  "--seed", "42", "--score-clip", "600",
                  "--out", str(frozen_path)])
    check(rc == 0, "freeze exits 0")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    check(frozen["schema"] == 1, "freeze writes schema 1")
    check(frozen["split"]["games"] == 4, "freeze sees all games")
    check(frozen["split"]["validation_games"] == [1, 2],
          "validation games come from the deterministic split")
    check(frozen["split"]["validation_records"] == 4,
          "validation records are counted (2 per game)")
    check(frozen["leakage"]["d4_equivalent_positions"] == 1,
          "the single mirror-equivalent pair is detected")
    check(frozen["leakage"]["excluded_train_games"] == [0],
          "the whole bridge train game is excluded")
    check(frozen["split"]["train_records_after_exclusion"] == 2,
          "only the clean train game survives (2 records)")
    check(frozen["baseline"]["constant_mse_on_validation"] > 0.0,
          "train-side constant baseline is fitted and reported")
    check(len(frozen["data"]["sha256"]) == 64, "data identity is recorded")

    vf.verify_frozen_data(frozen, main)  # same bytes -> accepted
    tampered = tmp / "tampered.binpack"
    tampered.write_bytes(main.read_bytes() + b"\x00")
    try:
        vf.verify_frozen_data(frozen, tampered)
        check(False, "tampered data must be rejected by the frozen identity")
    except SystemExit:
        pass

    manifest_path = tmp / "manifest.json"
    rc = vf.main(["check", "--frozen", str(frozen_path),
                  "--train-data", str(main), "--train-data", str(extra),
                  "--manifest", str(manifest_path)])
    check(rc == 0, "check exits 0")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {Path(f["path"]).name: f for f in manifest["files"]}

    main_entry = entries["main.binpack"]
    check(main_entry["games"] == [3],
          "the manifest keeps only clean non-validation games of the main file")
    check(main_entry["excluded_games"] == [0],
          "the leaking train game is excluded from the include-list")
    extra_entry = entries["extra.binpack"]
    check(extra_entry["excluded_games"] == [0],
          "cross-file leaks are found by position, not by game numbering")
    check(extra_entry["games"] == [1], "the clean extra game is kept")

    clean = tmp / "clean.binpack"
    write_binpack(clean, [game([Y3]), game([Z3], score=-30, wdl=1)])
    check_path = tmp / "clean_check.json"
    rc = vf.main(["check", "--frozen", str(frozen_path),
                  "--train-data", str(clean), "--out", str(check_path)])
    check(rc == 0, "clean check exits 0")
    clean_report = json.loads(check_path.read_text(encoding="utf-8"))
    check(clean_report["files"][0]["excluded_games"] == [],
          "no overlap means nothing is excluded")


def test_frozen_record_ids(tmp: Path) -> None:
    """The shared helpers keep train.py and export.py on one frozen split."""
    main, _ = build_files(tmp)
    frozen_path = tmp / "frozen_ids.json"
    vf.main(["freeze", "--data", str(main), "--val-fraction", "0.5",
             "--seed", "42", "--out", str(frozen_path)])
    frozen = vf.load_frozen(frozen_path)
    with binpack.BinpackReader(main) as reader:
        game_map = reader.game_index_map()
        val_ids = vf.frozen_validation_record_ids(game_map, frozen)
        train_ids = vf.frozen_train_record_ids(game_map, frozen)
        val_games = set(frozen["split"]["validation_games"])
        excluded = set(frozen["leakage"]["excluded_train_games"])
        check(len(val_ids) == frozen["split"]["validation_records"],
              "frozen validation ids match the frozen record count")
        check(len(train_ids) == frozen["split"]["train_records_after_exclusion"],
              "frozen train ids match the frozen train record count")
        check(not (set(val_ids) & set(train_ids)),
              "frozen train/validation record ids are disjoint")
        check(set(game_map[i] for i in val_ids) == val_games,
              "validation ids map exactly to the frozen validation games")
        check(set(game_map[i] for i in train_ids).isdisjoint(val_games | excluded),
              "train ids exclude both validation and D4-leaking games")


def test_torch_integration(tmp: Path) -> None:
    import train

    main, extra = build_files(tmp)
    frozen_path = tmp / "frozen.json"
    vf.main(["freeze", "--data", str(main), "--val-fraction", "0.5",
             "--seed", "42", "--out", str(frozen_path)])
    manifest_path = tmp / "manifest.json"
    vf.main(["check", "--frozen", str(frozen_path), "--train-data", str(main),
             "--train-data", str(extra), "--manifest", str(manifest_path)])

    # Default (non-frozen) mode must keep one single draw for both splits.
    history0 = tmp / "history0.json"
    train.main(["--data", str(main), "--output", str(tmp / "d.pt"),
                "--history", str(history0), "--epochs", "1",
                "--batch-size", "2", "--device", "cpu",
                "--wdl-weight", "0", "--quant-reg", "0", "--seed", "7"])
    h0 = json.loads(history0.read_text(encoding="utf-8"))
    check(h0["dataset"]["train_records"] + h0["dataset"]["val_records"] == 8,
          "default mode keeps a single disjoint game split (all 8 records)")

    history1 = tmp / "history1.json"
    train.main(["--data", str(main), "--frozen-validation", str(frozen_path),
                "--output", str(tmp / "a.pt"), "--history", str(history1),
                "--epochs", "1", "--batch-size", "2", "--device", "cpu",
                "--wdl-weight", "0", "--quant-reg", "0", "--seed", "7"])
    h1 = json.loads(history1.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    check(h1["dataset"]["val_records"]
          == frozen["split"]["validation_records"],
          "frozen mode evaluates on the exact frozen validation records")
    check(h1["dataset"]["train_records"]
          == frozen["split"]["train_records_after_exclusion"],
          "leak-excluded games are not trained on")
    check(h1["dataset"]["validation_games"]
          == frozen["split"]["validation_games"],
          "history records the frozen validation games")
    check(h1["config"]["frozen_validation"] == str(frozen_path),
          "history records the frozen validation path")
    check(h1["dataset"]["frozen_validation_identity"] is not None,
          "history records the frozen validation identity")

    history2 = tmp / "history2.json"
    train.main(["--data", str(main), "--frozen-validation", str(frozen_path),
                "--train-manifest", str(manifest_path),
                "--output", str(tmp / "b.pt"), "--history", str(history2),
                "--epochs", "1", "--batch-size", "2", "--device", "cpu",
                "--wdl-weight", "0", "--quant-reg", "0", "--seed", "7"])
    h2 = json.loads(history2.read_text(encoding="utf-8"))
    check(h2["dataset"]["val_records"] == h1["dataset"]["val_records"],
          "manifest files never change the frozen validation split")
    check(h2["dataset"]["train_records"] == 4,
          "manifest adds the clean extra game's records (2 + 2)")
    check(len(h2["dataset"]["train_files"]) == 2,
          "history lists every manifest training file")

    frozen2_path = tmp / "frozen2.json"
    vf.main(["freeze", "--data", str(extra), "--val-fraction", "0.5",
             "--seed", "3", "--out", str(frozen2_path)])
    try:
        train.main(["--data", str(main),
                    "--frozen-validation", str(frozen2_path),
                    "--epochs", "1", "--device", "cpu"])
        check(False, "a freeze built from other bytes must be rejected")
    except SystemExit:
        pass


def main() -> int:
    test_position_construction()
    test_canonical_position()
    with tempfile.TemporaryDirectory(prefix="gomoku_vfreeze_",
                                     ignore_cleanup_errors=True) as raw_tmp:
        tmp = Path(raw_tmp)
        test_freeze_and_check(tmp)
        test_frozen_record_ids(tmp)
        if has_torch():
            test_torch_integration(tmp)
        else:
            print("torch not installed: train.py integration tests SKIPPED")
    if FAILURES:
        print(f"\n{len(FAILURES)} frozen-validation check(s) failed")
        return 1
    print("validation_freeze_tests: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
