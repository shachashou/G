"""Profile the existing 7,208-record dataset, including D4 orbit overlap."""
from __future__ import annotations

import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "trainer"), str(ROOT / "tools")]
import binpack
import dataset
import features
from run_support import file_identity, write_json

path = ROOT / "data/raw/bench-d4/train.binpack"
groups = defaultdict(list)
raw = set()
records = []
with binpack.BinpackReader(path) as reader:
    game_map = reader.game_index_map()
    val_games = dataset.choose_validation_games(reader.header.game_count, .1, 42)
    for record in reader:
        sparse = record.feature_indices()
        raw.add((record.stm, tuple(sparse)))
        canonical = min(tuple(sorted(features.transform_sparse(sparse, symmetry)))
                        for symmetry in range(8))
        key = record.stm, canonical
        target = math.tanh(record.score / 400)
        is_val = game_map[record.index] in val_games
        groups[key].append((target, is_val))
        records.append((key, record.score, record.ply, is_val))
train_keys = {key for key, group in groups.items() if any(not row[1] for row in group)}
val_keys = {key for key, group in groups.items() if any(row[1] for row in group)}
train_count = sum(not row[3] for row in records)
noise_sum = 0.0
for group in groups.values():
    train_targets = [target for target, is_val in group if not is_val]
    if train_targets:
        mean = statistics.mean(train_targets)
        noise_sum += sum((target - mean) ** 2 for target in train_targets)
report = {"data": file_identity(path), "records": len(records),
          "games": len(set(game_map)), "distinct_raw_positions": len(raw),
          "distinct_D4_orbits": len(groups), "training_orbits": len(train_keys),
          "validation_orbits": len(val_keys),
          "validation_records": sum(row[3] for row in records),
          "validation_records_with_training_orbit": sum(row[3] and row[0] in train_keys for row in records),
          "D4_invariant_train_mse_lower_bound": noise_sum / train_count,
          "saturated_scores": sum(abs(row[1]) == 32767 for row in records),
          "early_ply_3_to_7_records": sum(row[2] <= 7 for row in records),
          "scope": "Existing benchmark dataset only. Different game IDs do not prove distinct position orbits."}
write_json(ROOT / "data/m5/dataset_profile.json", report)
for key, value in report.items():
    print(f"{key}: {value}")
