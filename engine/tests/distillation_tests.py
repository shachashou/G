"""Short ticket-18 integration checks; never runs experimental training/matches."""
from __future__ import annotations

import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "trainer"), str(ROOT / "tools")]
import binpack
import distill18
import move_correlation as mc
import targets
import teacher_labels as tl
import validation_freeze as vf
import weights_format as wf
import distill18_summary as summary
import distill18_screen as screen

PROBE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "build/engine/Release/nnue_probe.exe"
sys.argv = sys.argv[:1]


def plane(cells):
    raw = bytearray(29)
    for i in cells:
        raw[i//8] |= 1 << (i % 8)
    return bytes(raw)


def record(index, black, white, move=0, score=100, start=False, end=False):
    stm = 0 if len(black) == len(white) else 1
    return binpack.Record(index, plane(black), plane(white), score, 2, stm, start, end, len(black)+len(white), move)


def write_binpack(path, games):
    rows = []
    for game in games:
        for i, r in enumerate(game):
            flags = r.wdl | (r.stm << 2) | (8 if i == 0 else 0) | (16 if i == len(game)-1 else 0)
            rows.append(r.black + r.white + struct.pack("<hBBH", r.score, flags, r.ply, r.move))
    path.write_bytes(binpack.pack_header(binpack.Header(record_count=len(rows), game_count=len(games), search_depth=4)) + b"".join(rows))


class DistillationTests(unittest.TestCase):
    def test_ties_and_gate(self):
        stat = mc.tie_stats(1, [5, 5, 5, 5], [0, 1, 2, 3])
        self.assertEqual(stat["top1_credit"], .25)
        self.assertEqual(stat["top3_credit"], .75)
        cases = [{"moves": [0, 1], "teacher_move": 0, "record_id": 0, "game_id": 0, "stm": 0, "ply": 4},
                 {"moves": [0, 1, 2, 3], "teacher_move": 0, "record_id": 1, "game_id": 1, "stm": 1, "ply": 5}]
        result, rows = mc.aggregate_cases(cases, [0]*6)
        self.assertEqual(result["top1_tie_adjusted"], .375)
        self.assertEqual(result["random_top1"], .375)
        self.assertEqual(mc.tie_stats(0, [-2, 1], [0, 1])["top1_credit"], 1)
        self.assertFalse(summary.gate(.199999, .001, .4, [.1, .2], True)["passed"])
        self.assertFalse(summary.gate(.4, .01, .4, [0, .2], True)["passed"])
        self.assertTrue(summary.gate(.4, .01, .4, [.01, .2], True)["passed"])
        interval = mc.cluster_interval(rows, [rows], repeats=100)
        self.assertEqual(interval, [0, 0])
        with self.assertRaises(ValueError):
            mc.aggregate_cases(cases, [0]*5)
        with self.assertRaises(ValueError):
            mc.tie_stats(0, [float("nan")], [0])
        openings = {"positions": [[[i, 0], [i, 1]] for i in range(10)]}
        screen.reject_opening_overlap(openings, [])
        with self.assertRaises(ValueError):
            screen.reject_opening_overlap(openings, [("old", openings)])

    def test_summary_selection_and_rejections(self):
        with tempfile.TemporaryDirectory(prefix="distill18_summary_") as tmp:
            tmp = Path(tmp)
            def save(name, value):
                path = tmp/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")
                return tl.file_identity(path)
            pid = save("experiment_plan.json", {"test": "sealed plan fixture"})
            pre = save("preflight.json", {"complete": True})
            probe_id = tl.file_identity(PROBE)
            metric_id = save("metric.json", {})
            frozen_id = save("frozen.json", {})
            plan = {"inputs": {"metric_contract.json": metric_id, "frozen_validation.json": frozen_id},
                    "references": {}, "probe": probe_id, "hyperparameters": distill18.HYPERPARAMETERS,
                    "bootstrap_repeats": 100, "bootstrap_seed": 20261118}
            cases = [{"moves": list(range(10)), "teacher_move": 0, "record_id": i,
                      "game_id": i//2, "stm": i%2, "ply": 8} for i in range(8)]
            def correlation(name, weight_id, evaluations):
                result, rows = mc.aggregate_cases(cases, evaluations)
                rows_path = tmp/f"{name}.jsonl"
                rows_path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
                value = {"complete": True, "mode": "nnue", "backend": "avx2", "probe": probe_id,
                         "weights": weight_id, "metric_contract": metric_id, "per_case": tl.file_identity(rows_path),
                         "results": result, "static_limit_hits": 0}
                return save(f"{name}.json", value)
            refs = {}
            for n in range(4):
                weight = save(f"ref{n}.weight", {})
                plan["references"][str(n)] = weight
                refs[str(n)] = correlation(f"ref{n}", weight, [0]*80)
            base = save("baseline.json", {"reports": refs, "sensitivity_passed": True,
                        "best_historical_top1": .1, "teacher_top1": .5})
            save("stage_a.json", {"passed": True, "complete": True, "plan": pid, "baselines": base, "preflight": pre})
            for arm in ("d", "s"):
                target_id = save(f"target_{arm}.json", {})
                plan["inputs"][f"target_{arm}.json"] = target_id
                checkpoint_id = save(f"{arm}.pt", {})
                weight = save(f"{arm}.weight", {})
                # One hit and one miss per game keeps the clustered difference
                # positive for every possible game resample.
                evaluations = sum(([-1]+[0]*9 if i%2 == 0 else [1]+[0]*9 for i in range(8)), [])
                corr_id = correlation(arm, weight, evaluations)
                save(f"{arm}/history.json", {"complete": True, "config": distill18.HYPERPARAMETERS,
                     "optimisation": {"optimizer_steps": 1600, "cap_reached": True, "samples_exposed": 1638400},
                     "checkpoint": checkpoint_id["path"], "target_contract_identity": target_id})
                save(f"{arm}/fidelity/report.json", {"complete": True, "checkpoint": checkpoint_id,
                     "candidate": weight, "target_contract": target_id, "frozen_validation": frozen_id,
                     "move_correlation": corr_id, "cpp": {"exact": True, "all_primary_children": True}, "child_static_limit_hits": 0})
            with patch.object(summary.experiment, "verify_plan", return_value=(plan, {"coverage_passed": True})):
                result = summary.summarize(tmp, tmp/"summary.json")
                self.assertTrue(result["experiment_complete"], result["reason"])
                self.assertTrue(result["eligible_for_screening"])
                self.assertEqual(result["selected_arm"], "d")
                self.assertFalse(result["eligible_for_ticket16"])
                history = json.loads((tmp/"s/history.json").read_text())
                history["config"] = {**history["config"], "seed": 99}
                save("s/history.json", history)
                result = summary.summarize(tmp, tmp/"summary.json")
                self.assertFalse(result["eligible_for_screening"])
                self.assertIn("seed", result["reason"])

    def test_versioned_numerics(self):
        import torch
        import nnue
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="distill18_numeric_") as tmp:
            tmp = Path(tmp)
            for gain in (1, 2, 64, 1024):
                model = nnue.NNUEModel(config=nnue.ModelConfig(format_version=2, output_gain=gain))
                with torch.no_grad():
                    for p in model.parameters():
                        p.zero_()
                    model.b1[0] = 1
                    model.W2[0, 0] = 32
                    model.W2[1, 0] = -32
                    model.b2[0], model.b2[1] = -3, 5
                weights = model.quantized_weights()
                blob = wf.pack_weights(weights)
                decoded, meta = wf.unpack_weights(blob)
                self.assertEqual(meta["output_gain"], gain)
                path = tmp / "test.nnue"
                path.write_bytes(blob)
                expected = [wf.round_div(gain*32, 64)-3*gain, wf.round_div(-gain*32, 64)+5*gain]
                values = model(torch.zeros(2, 450), torch.tensor([0, 1]), True).tolist()
                self.assertEqual(values, expected)
                rebuilt = nnue.NNUEModel.from_quantized_weights(decoded)
                self.assertEqual(rebuilt.config.output_gain, gain)
                for h in (0, 1):
                    self.assertEqual(wf.infer_int(decoded, [], h), expected[h])
                    self.assertEqual(wf._infer_python(decoded, [], h, wf.QA, wf.QB), expected[h])
                for backend in ("scalar", "avx2"):
                    proc = subprocess.run([str(PROBE), "--weights", str(path), "--backend", backend, "infer"],
                                          input="0\n1\n", text=True, capture_output=True, check=True)
                    self.assertEqual([int(v) for v in proc.stdout.splitlines()], expected)
                corrupt = bytearray(blob)
                struct.pack_into("<I", corrupt, 52, 3)
                with self.assertRaises(wf.WeightFormatError):
                    wf.unpack_weights(corrupt)
                path.write_bytes(corrupt)
                self.assertNotEqual(subprocess.run([str(PROBE), "--weights", str(path), "infer"], capture_output=True).returncode, 0)
            with torch.no_grad():
                model.b2[:] = torch.tensor([32767, -32768])
            decoded, _ = wf.unpack_weights(wf.pack_weights(model.quantized_weights()))
            self.assertEqual([wf.infer_int(decoded, [], h) for h in (0, 1)], [900000, -900000])

    def test_cpp_candidates(self):
        cross = record(7, [111, 113, 97, 127], [0, 14, 210, 224])
        row = tl.probe_rows(PROBE, "candidates", [cross])[0]
        self.assertNotIn(112, [e["move"] for e in row["candidates"]])
        four = record(8, [108, 109, 110, 111], [0, 2, 4, 6])
        row = tl.probe_rows(PROBE, "candidates", [four])[0]
        self.assertEqual(next(e["result"] for e in row["candidates"] if e["move"] == 112), 1)
        overlap = record(9, [112], [112])
        with self.assertRaises(ValueError):
            tl.board_text(overlap)
        terminal = record(10, [108, 109, 110, 111, 112], [0, 2, 4, 6])
        self.assertEqual(tl.probe_rows(PROBE, "labels", [terminal])[0]["result"], 1)

    def test_pipeline_and_identity_failures(self):
        import torch
        import nnue
        import train
        import export
        import diagnose
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="distill18_pipeline_") as tmp:
            tmp = Path(tmp)
            data, oldfreeze, oldmanifest, out = tmp/"data.binpack", tmp/"old.json", tmp/"old_manifest.json", tmp/"experiment"
            rng, games, index = random.Random(987), [], 0
            for _ in range(16):
                cells = rng.sample(range(225), 12)
                game = []
                for ply in range(6, 10):
                    r = record(index, cells[:ply:2], cells[1:ply:2], score=(index%7-3)*250)
                    candidates = tl.probe_rows(PROBE, "candidates", [r])[0]["candidates"]
                    if not candidates or any(e["result"] for e in candidates):
                        self.fail("fixture unexpectedly terminal")
                    game.append(record(index, cells[:ply:2], cells[1:ply:2], candidates[0]["move"], (index%7-3)*250))
                    index += 1
                games.append(game)
            write_binpack(data, games)
            vf.main(["freeze", "--data", str(data), "--val-fraction", ".25", "--seed", "42", "--out", str(oldfreeze)])
            vf.main(["check", "--frozen", str(oldfreeze), "--train-data", str(data), "--manifest", str(oldmanifest)])
            before = oldfreeze.read_bytes()
            reference = tmp / "reference.nnue"
            wf.write_weights(reference, nnue.NNUEModel(seed=42).quantized_weights())
            args = SimpleNamespace(out_dir=out, parent_frozen=oldfreeze, train_manifest=oldmanifest,
                                   probe=PROBE, timeout=30, with_baselines=False,
                                   reference=[f"r{i}={reference}" for i in range(4)])
            distill18.prepare(args)
            self.assertEqual(before, oldfreeze.read_bytes())
            frozen = vf.load_frozen(out/"frozen_validation.json")
            self.assertEqual(frozen["split"], json.loads(before)["split"])
            self.assertFalse(json.loads((out/"metric_contract.json").read_text())["coverage_passed"])
            options = distill18.dataset_args(out)
            training, validation, _, _ = train.build_datasets(options)
            try:
                self.assertFalse(training.datasets[0].game_ids & validation.game_ids)
                self.assertEqual(len(training), options.target["constants"]["training_records"])
                store = tl.LabelStore(out/"labels/teacher_manifest.json")
                vals, _, _ = store.for_source(data)
                part = training.datasets[0]
                r = part.reader[part.indices[3]]
                self.assertEqual(part[3][2], vals[r.index])
            finally:
                training.close()
                validation.close()
            outd = out / "d"
            outd.mkdir()
            checkpoint, weights = tmp/"model.pt", outd/"candidate.nnue"
            argv = ["--data", str(data), "--output", str(checkpoint), "--history", str(outd/"history.json"),
                    "--frozen-validation", str(out/"frozen_validation.json"), "--train-manifest", str(out/"train_manifest.json"),
                    "--target-contract", str(out/"target_d.json"), "--epochs", "2", "--batch-size", "16", "--max-steps", "2",
                    "--checkpoint-policy", "last", "--quant-forward", "--wdl-weight", "0", "--quant-reg", "0", "--device", "cpu"]
            train.main(argv)
            export.main(["--checkpoint", str(checkpoint), "--data", str(data), "--frozen-validation", str(out/"frozen_validation.json"),
                         "--output", str(weights), "--report", str(outd/"export_report.json"), "--samples", "8"])
            self.assertEqual(wf.read_weights(weights)[1]["version"], 2)
            self.assertFalse(json.loads((outd/"export_report.json").read_text())["quantisation"]["calibrated"])
            diagnose.main(["fidelity", "--checkpoint", str(checkpoint), "--weights", str(weights), "--data", str(data),
                           "--export-report", str(outd/"export_report.json"), "--frozen-validation", str(out/"frozen_validation.json"),
                           "--probe", str(PROBE), "--all-validation", "--out-dir", str(outd/"fidelity")])
            report = json.loads((outd/"fidelity/report.json").read_text())
            self.assertTrue(report["cpp"]["exact"])
            self.assertTrue(report["cpp"]["all_primary_children"])
            self.assertEqual(report["manifest"]["assessment_records"], frozen["split"]["validation_records"])
            self.assertFalse(summary.summarize(out, out/"summary.json")["experiment_complete"])
            altered = tmp / "altered_target.json"
            c = options.target.copy()
            c["output_gain"] = 3
            altered.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                targets.load(altered)
            cache = store.manifest["files"][0]["labels"]["path"]
            Path(cache).write_bytes(b"broken")
            with self.assertRaises(ValueError):
                tl.LabelStore(out/"labels/teacher_manifest.json")


if __name__ == "__main__":
    unittest.main()
