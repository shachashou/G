"""Ticket-19 metric semantics and small real-probe end-to-end analysis."""
from __future__ import annotations

import json
from pathlib import Path
import random
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"trainer"), str(ROOT/"tools")]
import binpack
import distill18
import local_ranking_diagnosis as diagnosis
import move_correlation as mc
import local_ranking_fit as local_fit
import local_ranking_summary as local_summary
import teacher_labels as tl
import validation_freeze as vf
import weights_format as wf

PROBE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT/"build/engine/Release/nnue_probe.exe"
sys.argv = sys.argv[:1]


def plane(cells):
    result = bytearray(29)
    for i in cells:
        result[i//8] |= 1 << (i%8)
    return bytes(result)


def fixture(path):
    rng, games, index = random.Random(1919), [], 0
    for _ in range(12):
        cells = rng.sample(range(225), 9)
        game = []
        for ply in (6, 7):
            r = binpack.Record(index, plane(cells[:ply:2]), plane(cells[1:ply:2]),
                               (index%5-2)*250, 2, ply%2, ply == 6, ply == 7, ply, 0)
            response = tl.probe_rows(PROBE, "candidates", [r])[0]
            move = response["candidates"][0]["move"]
            flags = 2 | (r.stm << 2) | (8 if ply == 6 else 0) | (16 if ply == 7 else 0)
            game.append(r.black+r.white+struct.pack("<hBBH", r.score, flags, ply, move))
            index += 1
        games.extend(game)
    path.write_bytes(binpack.pack_header(binpack.Header(record_count=index, game_count=12, search_depth=4))+b"".join(games))


class LocalRankingTests(unittest.TestCase):
    def test_offset_is_not_a_ranking_error(self):
        result = diagnosis.local_scores([10, 0, -10], [110, 100, 90], 0)
        self.assertEqual(result["child_mae"], 100)
        self.assertEqual(result["centred_mse"], 0)
        self.assertEqual(result["teacher_best_hit"], 1)
        self.assertEqual(result["pair_disagreement"], 0)
        self.assertEqual(result["regret"], 0)
        # Search's recorded move can differ from the static evaluator's best.
        self.assertEqual(result["search_move_top1_credit"], 0)

    def test_constant_and_reversed_models(self):
        constant = diagnosis.local_scores([10, 0, -10], [5, 5, 5], 0)
        self.assertEqual(constant["teacher_best_hit"], 1/3)
        self.assertEqual(constant["teacher_best_random"], 1/3)
        self.assertEqual(constant["teacher_best_excess"], 0)
        self.assertEqual(constant["centred_mse_ratio"], 1)
        self.assertEqual(constant["pair_disagreement"], .5)
        reverse = diagnosis.local_scores([10, 0, -10], [-10, 0, 10], 0)
        self.assertEqual(reverse["pair_disagreement"], 1)
        self.assertEqual(reverse["regret"], 20)
        all_equal = diagnosis.local_scores([3, 3], [9, 1], 0)
        self.assertIsNone(all_equal["centred_mse_ratio"])
        result = diagnosis.local_summary([{"local": {"candidate": all_equal}}])["candidate"]
        self.assertEqual(result["constant_teacher_parents"], 1)
        self.assertIsNone(result["teacher_best_hit"])

    def test_extended_child_leakage_is_explicit(self):
        b = ["."]*225
        b[112], b[113] = "x", "o"
        shared = {"board": "".join(b), "stm": 0, "moves": [97, 98], "bucket": "primary"}
        val = [{**shared}]
        train = [{**shared, "board": "".join(b[:97]+["x"]+b[98:]), "stm": 1,
                  "moves": [96, 99], "game_key": "other-file:7"}]
        result = diagnosis.extended_overlap(train, val)
        self.assertEqual(result["counts"]["parent_vs_child"], 1)
        self.assertFalse(result["extended_position_separation_verified"])
        self.assertFalse(train[0]["eligible_for_followup_training"])
        duplicate = [{**shared, "game_key": "same-position"}]
        with self.assertRaisesRegex(ValueError, "parent D4 separation"):
            diagnosis.extended_overlap(duplicate, val)

    def test_small_pipeline_replay_and_identity_failure(self):
        import torch
        import nnue
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="local_diagnosis_") as temporary:
            tmp = Path(temporary)
            data, frozen, manifest = tmp/"data.binpack", tmp/"old.json", tmp/"manifest.json"
            fixture(data)
            vf.main(["freeze", "--data", str(data), "--val-fraction", ".25", "--seed", "42", "--out", str(frozen)])
            vf.main(["check", "--frozen", str(frozen), "--train-data", str(data), "--manifest", str(manifest)])
            reference = tmp/"reference.nnue"
            wf.write_weights(reference, nnue.NNUEModel(seed=42).quantized_weights())
            base = tmp/"ticket18"
            distill18.prepare(SimpleNamespace(out_dir=base, parent_frozen=frozen, train_manifest=manifest,
                               probe=PROBE, timeout=30, with_baselines=False, reference=[f"r{i}={reference}" for i in range(4)]))
            summary = {"experiment_complete": True, "plan": tl.file_identity(base/"experiment_plan.json"), "arms": {}}
            for arm in ("d", "s"):
                options = distill18.dataset_args(base, arm)
                model = nnue.NNUEModel(seed=42, config=nnue.ModelConfig(format_version=2, output_gain=options.target["output_gain"]))
                checkpoint, candidate, evidence = tmp/f"{arm}.pt", tmp/f"{arm}.nnue", tmp/f"{arm}.json"
                torch.save({"config": vars(model.config), "model_state": model.state_dict(),
                            "target_contract_identity": tl.file_identity(options.target_contract),
                            "frozen_validation_identity": tl.file_identity(base/"frozen_validation.json")}, checkpoint)
                wf.write_weights(candidate, model.quantized_weights())
                correlation = mc.assess_cases(PROBE, candidate, "nnue", "avx2", base/"move_cases.jsonl",
                                             base/"metric_contract.json", tmp/f"{arm}_correlation.json", 30)
                evidence.write_text(json.dumps({"checkpoint": tl.file_identity(checkpoint), "candidate": tl.file_identity(candidate),
                     "frozen_validation": tl.file_identity(base/"frozen_validation.json"),
                     "child_rankings": {"final_integer": correlation["results"]}}), encoding="utf-8")
                summary["arms"][arm] = {"fidelity": tl.file_identity(evidence), "candidate": tl.file_identity(candidate)}
            (base/"summary.json").write_text(json.dumps(summary), encoding="utf-8")
            before = (base/"frozen_validation.json").read_bytes()
            args = SimpleNamespace(experiment_dir=base, out_dir=tmp/"analysis", train_parents=4,
                                   seed=20261119, sample_manifest=None, timeout=30, threads=1)
            report = diagnosis.analyse(args)
            self.assertTrue(report["complete"])
            self.assertFalse(report["training_performed"])
            self.assertEqual(before, (base/"frozen_validation.json").read_bytes())
            self.assertTrue(all(v["exact"] for v in report["parity"].values()))
            sample = diagnosis.read(args.out_dir/"sample_manifest.json")
            self.assertEqual(len(sample["train_sources"]), 4)
            self.assertEqual(len({(r["source_sha256"], r["game_id"]) for r in sample["train_sources"]}), 4)
            self.assertEqual([r["stm"] for r in sample["train_sources"]], [0, 1, 0, 1])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                diagnosis.analyse(args)
            first = args.out_dir
            args.out_dir, args.sample_manifest = tmp/"replay", first/"sample_manifest.json"
            replay = diagnosis.analyse(args)
            self.assertEqual(report["splits"], replay["splits"])
            # Small synthetic control: two independently initialised fits,
            # identical data/batches/projection, different forward modes.
            cfg = {**local_fit.FIT_CONFIG, "parents": 4, "per_head": 2, "steps": 2, "batch_size": 2,
                   "output_gain": options.target["output_gain"], "loss_normalizer": options.target["output_gain"], "smoke_test": True}
            fit_dir = tmp/"fit_b"
            sealed = local_fit.prepare(first, fit_dir, cfg)
            manifest_path = fit_dir/"fit_manifest.json"
            float_report = local_fit.fit(manifest_path, "float", fit_dir/"float_initial", "cpu")
            qat_report = local_fit.fit(manifest_path, "qat", fit_dir/"qat_initial", "cpu")
            self.assertEqual(float_report["initial_master_sha256"], qat_report["initial_master_sha256"])
            self.assertEqual(float_report["batch_schedule_sha256"], qat_report["batch_schedule_sha256"])
            self.assertFalse(float_report["trained_quantized"])
            self.assertTrue(qat_report["trained_quantized"])
            self.assertNotEqual(float_report["final_master_sha256"], float_report["initial_master_sha256"])
            self.assertNotEqual(qat_report["final_master_sha256"], qat_report["initial_master_sha256"])
            self.assertEqual(qat_report["optimizer_steps"], 2)
            self.assertEqual(qat_report["samples_exposed"], 4)
            compare = local_summary.summarise(manifest_path, fit_dir/"float_initial/report.json",
                          fit_dir/"qat_initial/report.json", fit_dir/"summary.json")
            self.assertTrue(compare["complete"], compare.get("reason"))
            self.assertFalse(compare["official_stage_b_complete"])
            self.assertFalse(compare["eligible_for_screening"])
            altered_report = dict(qat_report)
            altered_report["batch_schedule_sha256"] = "wrong-schedule"
            (fit_dir/"qat_initial/report.json").write_text(json.dumps(altered_report), encoding="utf-8")
            compare = local_summary.summarise(manifest_path, fit_dir/"float_initial/report.json",
                          fit_dir/"qat_initial/report.json", fit_dir/"summary.json")
            self.assertFalse(compare["complete"])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                local_fit.fit(manifest_path, "qat", fit_dir/"qat_initial", "cpu")
            sample["train_sources"][0]["record_id"] += 1
            bad = tmp/"bad_sample.json"
            bad.write_text(json.dumps(sample), encoding="utf-8")
            args.out_dir, args.sample_manifest = tmp/"bad", bad
            with self.assertRaisesRegex(ValueError, "deterministic draw"):
                diagnosis.analyse(args)
            self.assertFalse(diagnosis.read(args.out_dir/"report.json")["complete"])
            with self.assertRaises(ValueError):
                diagnosis.evaluate(PROBE, ["."*225], [], timeout=10)

    def test_common_master_projection(self):
        import torch
        import nnue
        model = nnue.NNUEModel()
        with torch.no_grad():
            model.W0[0, 0] = 100
            model.b2[0] = -40000
        self.assertEqual(local_fit.project_master(model), 2)
        self.assertEqual(float(model.W0[0, 0]), 64)
        self.assertEqual(float(model.b2[0]), -32768)


if __name__ == "__main__":
    unittest.main()
