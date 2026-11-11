"""Ticket 10: real outcomes, strict source isolation and blocked-teacher loop."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("tools", "trainer", "engine/tests")]
import history_evaluation as he
import selfplay_data as sp
from model_registry import ModelRegistry
from sample_pool import SamplePool, PoolError
from training_task import TaskConfig, TrainingTask
from value_fixture import build_pool
from value_train import TrainConfig, train_value
from value_data import load_pool, record_target
from value_model import value_loss
import value_format as vf
import torch

EXE = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else str(ROOT / "build/engine/gomoku_nnue.exe")


def make_incumbent(root, exe=EXE):
    root = Path(root)
    pool = build_pool(root / "bootstrap.sqlite3")
    trained = train_value(TrainConfig(pool=str(pool), output_dir=str(root / "bootstrap"),
                                      steps=4, batch_size=8, device="cpu"))
    assert trained["export"]["passed"]
    report = he.evaluate(root / "bootstrap/model.gmvalue", exe, registry=root / "registry",
        output_root=root / "bootstrap-evaluations", bootstrap=True,
        config=he.EvaluationConfig(pairs=1, movetime_ms=10, vcf_policy="off"))
    assert report["publication"], report.get("publication_error")
    return root / "registry"


class ResultEngine:
    mode = "black_win"
    def __init__(self, spec, timeout, stop):
        self.spec, self.stop = spec, stop
        _, self.meta = vf.read_value_weights(spec.argv[2])
    def start(self, *args):
        pass
    def send(self, *args):
        pass
    def recv_line(self):
        return (f'eval="nnue", eval_engine="gmvalue1-incremental-avx2", value_model_id="{self.meta["model_id"]}", '
            'value_model_arch="450-1024-64-32-1", value_model_rule="renju", '
            'value_model_feature_spec="absolute-color-450", value_model_contract="black-absolute-tanh-v1"')
    def move(self, cells):
        if self.mode == "interrupted":
            self.stop.set()
            raise InterruptedError("controlled stop")
        if self.mode in ("timeout", "crash"):
            raise he.EngineFailure(self.mode, "controlled transport anomaly")
        if self.mode == "illegal":
            return 99, 99
        black = cells.count(1) == cells.count(2)
        if self.mode == "draw":
            choices = [(x, y) for y in range(15) for x in range(15)
                       if (((x + 2*y) % 4 < 2) == black)]
        elif (self.mode == "black_win") == black:
            choices = [(x, 0 if black else 14) for x in range(5)]
        else:
            choices = [(x, 2 if black else 12) for x in (0, 3, 6, 9, 12)]
        return next(p for p in choices if cells[p[1]*15+p[0]] == 0)
    def close(self):
        pass


class SelfPlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tempfile.TemporaryDirectory(prefix="selfplay-base-")
        cls.registry = make_incumbent(Path(cls.base.name))
    @classmethod
    def tearDownClass(cls):
        cls.base.cleanup()
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="selfplay-case-")
        self.root = Path(self.temp.name)
        shutil.copytree(self.registry, self.root / "registry")
        self.cfg = TaskConfig(mode="selfplay", engine=EXE, pool=str(self.root / "mixed.sqlite3"),
            registry=str(self.root / "registry"), output_root=str(self.root / "tasks"),
            evaluations=str(self.root / "evaluations"), capacity=1000, validation_games=1,
            min_train_samples=1, min_validation_samples=1, min_new_samples=1,
            max_pending_samples=4, train_steps=4, batch_size=4, lr=0.00001, device="cpu",
            max_rounds=1, max_games=6, max_minutes=2, min_round_interval_s=0,
            collection_delay_ms=10, seed=32,
            selfplay={"opening_plies":0,"epsilon":0,"sample_every":1,"movetime_ms":10,
                      "vcf_policy":"off","filter_vcf":False},
            evaluation={"pairs":1,"movetime_ms":10,"vcf_policy":"off"})
        ResultEngine.mode = "black_win"
    def tearDown(self):
        self.temp.cleanup()
    def source(self, mode=None):
        task = TrainingTask(self.cfg)
        task.started = __import__('time').monotonic()
        task.directory.mkdir(parents=True)
        task.engine = Path(EXE)
        source = sp.SelfPlayDataSource(task)
        source.preflight()
        return task, source

    def test_black_white_and_draw_outcomes_are_absolute_and_replayable(self):
        with SamplePool(self.cfg.pool) as pool:
            for index, (mode, value) in enumerate(sp.RESULTS.items()):
                task, source = self.source()
                ResultEngine.mode = mode
                with patch.object(sp, "MatchEngine", ResultEngine):
                    result = source.produce(pool, index)
                record = json.loads(Path(result["game_record"]["path"]).read_text())
                self.assertEqual(record["status"], "completed", record)
                self.assertEqual(record["final_result"], mode)
                self.assertEqual(record["result_value"], value)
                rows = pool.read_batch(split="train") + pool.read_batch(split="validation")
                own = [r for r in rows if r["source_id"] in result["source_ids"]]
                self.assertTrue(own)
                self.assertEqual({record_target(r) for r in own}, {value})
                self.assertTrue(all(r["value"] is None and r["raw_score"] is None for r in own))
                self.assertTrue(all(r["final_result"] == mode for r in own))
                self.assertEqual(record["teacher_calls"], 0)
            self.assertFalse(pool.validate())

    def test_mixed_pool_filter_and_transition_weights(self):
        build_pool(self.cfg.pool, validation_games=1)
        with SamplePool(self.cfg.pool) as pool:
            before = list(pool.config()["validation"]["source_ids"])
            for i, mode in enumerate(("black_win", "white_win")):
                task, source = self.source()
                ResultEngine.mode = mode
                with patch.object(sp, "MatchEngine", ResultEngine):
                    source.produce(pool, i)
            self.assertTrue(set(before).issubset(pool.config()["validation"]["source_ids"]))
            stats = pool.stats(["selfplay_result"])
            self.assertGreater(stats["train_samples"], 0)
            self.assertGreater(stats["validation_samples"], 0)
            self.assertGreater(stats["total_pool_samples"], stats["pool_samples"])
        pure = load_pool(self.cfg.pool, label_sources=["selfplay_result"])
        self.assertEqual(pure.identity["label_sources"], ["selfplay_result"])
        self.assertTrue(all(s.label_source == "selfplay_result" for s in pure.train + pure.validation))
        mixed = load_pool(self.cfg.pool, label_sources=["teacher", "selfplay_result"],
                          source_weights={"teacher":.25,"selfplay_result":1})
        self.assertEqual(set(mixed.identity["label_sources"]), {"teacher", "selfplay_result"})
        self.assertTrue(all(s.weight == (.25 if s.label_source == "teacher" else 1) for s in mixed.train))
        loss = value_loss(torch.tensor([1.,1.]), torch.tensor([0.,1.]), weights=torch.tensor([.25,1.]))
        self.assertAlmostEqual(float(loss), .2, places=6)

    def test_truncation_interruption_and_anomalies_never_create_labels(self):
        original = ModelRegistry(self.cfg.registry).resolve()["sha256"]
        for mode in ("truncated", "interrupted", "timeout", "crash", "illegal"):
            config = replace(self.cfg, pool=str(self.root / (mode + '.sqlite3')),
                max_games=1, max_empty_games=1,
                selfplay={**self.cfg.selfplay, "max_plies":1 if mode == "truncated" else 225})
            ResultEngine.mode = "black_win" if mode == "truncated" else mode
            with patch.object(sp, "MatchEngine", ResultEngine):
                result = TrainingTask(config).run()
            self.assertEqual(result["status"], "completed" if mode == "truncated" else
                             "stopped" if mode == "interrupted" else "failed", result["reason"])
            with SamplePool(config.pool) as pool:
                self.assertEqual(pool.stats()["pool_samples"], 0)
                self.assertEqual(pool.stats()["staged_games"], 0)
            record = json.loads(next((Path(result["directory"]) / "games").glob('*.json')).read_text())
            self.assertIsNone(record["result_value"])
            self.assertIsNone(record["final_result"])
            self.assertEqual(ModelRegistry(config.registry).resolve()["sha256"], original)
            if mode == "truncated":
                events = [json.loads(line) for line in (Path(result["directory"])/"events.jsonl").read_text().splitlines()]
                self.assertTrue(any(e["event"] == "data_incomplete" for e in events))
                self.assertFalse(any(e["event"] == "data_commit" for e in events))

    def test_requires_existing_incumbent(self):
        result = TrainingTask(replace(self.cfg, registry=str(self.root / "no-registry"))).run()
        self.assertEqual(result["status"], "failed")
        self.assertIn("establish v0 first", result["reason"])

    def test_filtering_keeps_the_complete_game_result(self):
        from executor import ExecutorClient
        def referee(*args, **kw):
            client = ExecutorClient(*args, **kw)
            client.vcf = lambda **budget: {"vcf": {"status": "proven", "fixture": True}}
            return client
        self.cfg = replace(self.cfg, selfplay={**self.cfg.selfplay, "filter_vcf": True})
        task, source = self.source()
        with SamplePool(self.cfg.pool) as pool, patch.object(sp, "MatchEngine", ResultEngine), \
                patch.object(sp, "ExecutorClient", referee):
            result = source.produce(pool, 0)
            record = json.loads(Path(result["game_record"]["path"]).read_text())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["final_result"], "black_win")
            self.assertTrue(record["moves"] and record["filtered_plies"])
            self.assertEqual(result["inserted"], 0)
            self.assertEqual(pool.stats()["pool_samples"], 0)

    def test_real_loop_with_teacher_modules_and_processes_blocked(self):
        build_pool(self.cfg.pool, validation_games=1)
        config = replace(self.cfg, max_games=15,
            selfplay={**self.cfg.selfplay,"opening_plies":2,"epsilon":.05})
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(asdict(config)))
        script = self.root / "blocked_teacher.py"
        script.write_text('''import importlib.abc,json,pathlib,subprocess,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in ('rapfi_teacher','teacher_label','collect'):
   raise AssertionError('forbidden teacher import: '+fullname)
sys.meta_path.insert(0,Block())
spawned=[]
class GuardPopen(subprocess.Popen):
 def __init__(self,args,*a,**k):
  text=str(args).lower()
  assert 'pbrain-rapfi' not in text, text
  spawned.append(str(args))
  super().__init__(args,*a,**k)
subprocess.Popen=GuardPopen
sys.path.insert(0,sys.argv[1])
from training_task import TaskConfig,TrainingTask
config=TaskConfig(**json.loads(pathlib.Path(sys.argv[2]).read_text()))
result=TrainingTask(config).run()
pathlib.Path(sys.argv[3]).write_text(json.dumps({'result':result,'spawned':spawned}))
assert result['status']=='completed',result['reason']
assert result['rounds'] and result['rounds'][0]['evaluation'], json.dumps(result)
assert result['rounds'][0]['data_identity']['label_sources']==['selfplay_result']
assert result['rounds'][0]['data_identity']['train_games']>=2
assert result['resources']['teacher_threads']==0
from dataclasses import replace
transition=TrainingTask(replace(config,mode='transition')).run()
assert transition['status']=='completed',transition['reason']
assert transition['rounds'] and transition['rounds'][0]['evaluation'],json.dumps(transition)
assert set(transition['rounds'][0]['data_identity']['label_sources'])=={'teacher','selfplay_result'}
assert transition['rounds'][0]['data_identity']['source_weights']=={'teacher':.25,'selfplay_result':1.0}
pathlib.Path(sys.argv[3]).write_text(json.dumps({'result':result,'transition':transition,'spawned':spawned}))
''')
        output = self.root / "blocked-result.json"
        process = subprocess.run([sys.executable,str(script),str(ROOT/'tools'),str(config_path),str(output)],
                                 capture_output=True,text=True,timeout=90)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = json.loads(output.read_text())["result"]
        transition = json.loads(output.read_text())["transition"]
        self.assertEqual(transition["mode"], "transition")
        self.assertIsNone(result["rounds"][0]["evaluation"]["publication"])
        self.assertEqual(result["current"]["version"], "v0")
        records = [json.loads(p.read_text()) for p in (Path(result["directory"])/"games").glob('*.json')]
        complete = [r for r in records if r["status"] == "completed"]
        self.assertGreaterEqual(len(complete), 2)
        self.assertTrue(all(r["teacher_calls"] == 0 for r in records))
        self.assertTrue(any(m["kind"] == "opening" for r in complete for m in r["moves"]))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
