"""Ticket 19 stage A: read-only parent/child score and local-ranking diagnosis.

Uses ticket-18 frozen identities, fixed training-side game sampling, actual
C++ handcrafted labels, and unchanged D/S checkpoints. Never trains, tunes,
calibrates, generates self-play, or starts matches.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import binpack
import distill18
from distill18_fidelity import model_predictions
import export
import move_correlation as mc
import targets
import teacher_labels as tl
import validation_freeze as vf
import weights_format as wf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import file_identity, source_identity, write_json

SCHEMA = 1
SEED = 20261119


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sparse(board):
    if len(board) != 225 or any(c not in ".xo" for c in board):
        raise ValueError("malformed board")
    return [i for i, c in enumerate(board) if c == "x"] + [225+i for i, c in enumerate(board) if c == "o"]


def position_key(board, stm):
    return vf.canonical_position(sparse(board), stm)


def child_boards(case):
    for move in case["moves"]:
        if case["board"][move] != ".":
            raise ValueError("occupied candidate")
        board = list(case["board"])
        board[move] = "x" if case["stm"] == 0 else "o"
        yield "".join(board)


def evaluate(probe, boards, stms, weights=None, backend="avx2", timeout=120):
    if len(boards) != len(stms) or not boards:
        raise ValueError("empty or mismatched probe batch")
    command = [str(Path(probe).resolve())]
    command += ["handcrafted"] if weights is None else ["--weights", str(Path(weights).resolve()), "--backend", backend, "board"]
    result = subprocess.run(command, input="".join(f"{s} {b}\n" for s, b in zip(stms, boards)),
                            text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise ValueError(f"probe failed: {result.stderr.strip()}")
    values = [int(v) for v in result.stdout.splitlines()]
    if len(values) != len(boards):
        raise ValueError("probe response count mismatch")
    return values


def select_training_sources(manifest, store, count, seed):
    """Exactly balanced stm, at most one parent per (source hash, game).

    Games are shuffled first, then one requested-head record is drawn from
    each selected game. Sampling never inspects teacher values/predictions.
    """
    if count < 2 or count % 2:
        raise ValueError("--train-parents must be an even integer >= 2")
    games = defaultdict(lambda: [[], []])
    for entry in manifest["files"]:
        path = Path(entry["path"])
        resolved = str(path.resolve())
        _, valid, cache = store.for_source(path)
        allowed = set(entry["games"])
        with binpack.BinpackReader(path) as reader:
            mapping = reader.game_index_map()
            for i, game in enumerate(mapping):
                if game in allowed and valid[i]:
                    games[(cache["source"]["sha256"], game)][reader[i].stm].append((resolved, i))
    rng = random.Random(seed)
    available = sorted(games)
    rng.shuffle(available)
    selected = []
    for slot in range(count):
        head = slot % 2
        found = next((i for i, g in enumerate(available) if games[g][head]), None)
        if found is None:
            raise ValueError("not enough independent games for balanced sample")
        game = available.pop(found)
        path, index = rng.choice(games[game][head])
        selected.append({"source_sha256": game[0], "source_path": path, "game_id": game[1],
                         "record_id": index, "stm": head})
    return selected


def make_cases(sources, manifest, store, probe, timeout):
    allowed = {str(Path(e["path"]).resolve()).lower(): set(e["games"]) for e in manifest["files"]}
    readers, mappings, cases = {}, {}, []
    try:
        for selection in sources:
            path = str(Path(selection["source_path"]).resolve())
            if path.lower() not in allowed:
                raise ValueError("sample source outside training manifest")
            if path not in readers:
                readers[path] = binpack.BinpackReader(path)
                mappings[path] = readers[path].game_index_map()
            reader = readers[path]
            values, mask, cache = store.for_source(path)
            record = reader[selection["record_id"]]
            game = mappings[path][record.index]
            if (selection["source_sha256"] != cache["source"]["sha256"] or game != selection["game_id"]
                    or game not in allowed[path.lower()] or record.stm != selection["stm"] or not mask[record.index]):
                raise ValueError("sample membership/identity mismatch")
            row = tl.probe_rows(probe, "candidates", [record], timeout)[0]
            if row["eval"] != values[record.index] or row["result"] != 0:
                raise ValueError("teacher cache/probe mismatch or terminal parent")
            moves = [r["move"] for r in row["candidates"]]
            if moves != sorted(set(moves)) or record.move not in moves:
                raise ValueError("invalid candidates or missing recorded move")
            results = [r["result"] for r in row["candidates"]]
            bucket = "terminal-tactical" if any(results) else "single-candidate" if len(moves) < 2 else "primary"
            cases.append({"key": f"{cache['source']['sha256']}:{record.index}", "split": "train_sample",
                          **selection, "game_key": f"{cache['source']['sha256']}:{game}", "ply": record.ply,
                          "teacher_move": record.move, "source_search_score": record.score,
                          "board": tl.board_text(record), "teacher_parent": values[record.index],
                          "moves": moves, "child_results": results, "bucket": bucket})
    finally:
        for reader in readers.values():
            reader.close()
    return cases


def validation_cases(directory, frozen, store):
    metric_path = tl.verify_identity(frozen["move_metric_contract"])
    metric = read(metric_path)
    cases_path = tl.verify_identity(metric["cases"])
    original = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    source = tl.verify_identity(frozen["data"])
    values, mask, cache = store.for_source(source)
    cases = []
    with binpack.BinpackReader(source) as reader:
        mapping = reader.game_index_map()
        ids = vf.frozen_validation_record_ids(mapping, frozen)
        if ids != [c["record_id"] for c in original] or ids != frozen["split_identity"]["validation_record_ids"]:
            raise ValueError("validation record membership changed")
        for raw in original:
            r = reader[raw["record_id"]]
            if not mask[r.index] or raw["board"] != tl.board_text(r) or raw["stm"] != r.stm or raw["teacher_move"] != r.move or raw["game_id"] != mapping[r.index]:
                raise ValueError("validation case/source mismatch")
            cases.append({**raw, "key": f"{cache['source']['sha256']}:{r.index}", "split": "validation",
                          "source_sha256": cache["source"]["sha256"], "source_path": str(source),
                          "game_key": f"{cache['source']['sha256']}:{mapping[r.index]}",
                          "teacher_parent": values[r.index], "source_search_score": r.score})
    if len(cases) != metric["total_positions"] or sum(c["bucket"] == "primary" for c in cases) != metric["primary_positions"]:
        raise ValueError("validation primary coverage changed")
    return cases


def extended_overlap(train_cases, val_cases):
    val_parents = {position_key(c["board"], c["stm"]) for c in val_cases}
    val_children = {position_key(b, 1-c["stm"]) for c in val_cases if c["bucket"] == "primary" for b in child_boards(c)}
    totals = Counter()
    excluded_games = set()
    for case in train_cases:
        parent_key = position_key(case["board"], case["stm"])
        counts = {"parent_vs_parent": int(parent_key in val_parents), "parent_vs_child": int(parent_key in val_children),
                  "child_vs_parent": 0, "child_vs_child": 0}
        if case["bucket"] == "primary":
            for board in child_boards(case):
                key = position_key(board, 1-case["stm"])
                counts["child_vs_parent"] += key in val_parents
                counts["child_vs_child"] += key in val_children
        case["extended_overlap"] = counts
        case["eligible_for_followup_training"] = not any(counts.values())
        if not case["eligible_for_followup_training"]:
            excluded_games.add(case["game_key"])
        totals.update(counts)
    if totals["parent_vs_parent"]:
        raise ValueError("original frozen parent D4 separation violated")
    return {"counts": dict(totals), "excluded_sample_game_keys": sorted(excluded_games),
            "rule": "future fitting excludes complete source games if any sampled parent/primary child matches any validation parent/primary child under colour-preserving D4+stm",
            "diagnostic_only": True, "parent_position_separation_verified": True,
            "extended_position_separation_verified": not excluded_games}


def local_scores(teacher_children, predicted_children, teacher_move_index):
    """All child scores are opponent-stm; negate both for the parent's move."""
    if len(teacher_children) != len(predicted_children) or len(teacher_children) < 2:
        raise ValueError("invalid local candidate scores")
    h = [-v for v in teacher_children]
    e = [-v for v in predicted_children]
    n = len(h)
    mh, me = statistics.mean(h), statistics.mean(e)
    teacher_variance = statistics.mean((v-mh)**2 for v in h)
    centred = [((v-me)-(t-mh)) for v, t in zip(e, h)]
    best_h, best_e = max(h), max(e)
    teacher_best = {i for i, v in enumerate(h) if v == best_h}
    predicted_best = {i for i, v in enumerate(e) if v == best_e}
    hit = len(teacher_best & predicted_best)/len(predicted_best)
    random_hit = len(teacher_best)/n
    pairs, wrong, gap_counts, gap_wrong = 0, 0.0, Counter(), Counter()
    for i in range(n):
        for j in range(i):
            dh = h[i]-h[j]
            if dh == 0:
                continue
            de = e[i]-e[j]
            error = .5 if de == 0 else float(dh*de < 0)
            gap = abs(dh)
            bucket = "(0,128]" if gap <= 128 else "(128,1024]" if gap <= 1024 else "(1024,8192]" if gap <= 8192 else ">8192"
            pairs += 1
            wrong += error
            gap_counts[bucket] += 1
            gap_wrong[bucket] += error
    tied = [i for i, v in enumerate(e) if v == e[teacher_move_index]]
    better = sum(v > e[teacher_move_index] for v in e)
    return {"candidates": n, "child_mae": statistics.mean(abs(v-t) for v, t in zip(e, h)),
            "offset_error": me-mh, "centred_mse": statistics.mean(v*v for v in centred),
            "centred_mae": statistics.mean(abs(v) for v in centred), "teacher_variance": teacher_variance,
            "centred_mse_ratio": statistics.mean(v*v for v in centred)/teacher_variance if teacher_variance > 0 else None,
            "teacher_best_hit": hit, "teacher_best_random": random_hit,
            "teacher_best_excess": (hit-random_hit)/(1-random_hit) if random_hit < 1 else None,
            "teacher_best_count": len(teacher_best), "predicted_best_count": len(predicted_best),
            "regret": best_h-statistics.mean(h[i] for i in predicted_best),
            "unequal_teacher_pairs": pairs, "pair_disagreement": wrong/pairs if pairs else None,
            "pair_gap_buckets": {k: {"pairs": gap_counts[k], "disagreement": gap_wrong[k]/gap_counts[k]} for k in gap_counts},
            "search_move_top1_credit": 1/len(tied) if better == 0 else 0}


def score_summary(rows, constants, normalizer):
    if not rows:
        return {"records": 0}
    labels = [r["teacher"] for r in rows]
    result = {}
    teacher_sd = statistics.pstdev(labels)
    for name in rows[0]["predictions"]:
        predictions = [r["predictions"][name] for r in rows]
        result[name] = targets.metrics(predictions, labels, normalizer, constants)
        result[name].update(signed_bias=statistics.mean(p-h for p, h in zip(predictions, labels)),
                            teacher_std=teacher_sd, prediction_std=statistics.pstdev(predictions),
                            spread_ratio=statistics.pstdev(predictions)/teacher_sd if teacher_sd else None)
    return result


def local_summary(rows):
    if not rows:
        return {"parents": 0}
    result = {}
    for name in rows[0]["local"]:
        values = [r["local"][name] for r in rows]
        informative = [v for v in values if v["teacher_variance"] > 0]
        pairs = [v for v in values if v["pair_disagreement"] is not None]
        item = {"parents": len(values), "informative_parents": len(informative),
                "constant_teacher_parents": len(values)-len(informative),
                "search_move_top1_tie_adjusted": statistics.mean(v["search_move_top1_credit"] for v in values),
                "macro_child_mae": statistics.mean(v["child_mae"] for v in values),
                "macro_offset_absolute": statistics.mean(abs(v["offset_error"]) for v in values),
                "centred_rmse": math.sqrt(statistics.mean(v["centred_mse"] for v in values)),
                "pooled_centred_mse_ratio": sum(v["centred_mse"] for v in informative)/sum(v["teacher_variance"] for v in informative) if informative else None,
                "macro_pair_disagreement": statistics.mean(v["pair_disagreement"] for v in pairs) if pairs else None}
        for key in ("centred_mse_ratio", "teacher_best_hit", "teacher_best_random", "teacher_best_excess", "regret"):
            item[key] = statistics.mean(v[key] for v in informative) if informative else None
        item["pair_gap_buckets"] = {}
        for bucket in ("(0,128]", "(128,1024]", "(1024,8192]", ">8192"):
            usable = [v["pair_gap_buckets"][bucket] for v in values if bucket in v["pair_gap_buckets"]]
            item["pair_gap_buckets"][bucket] = {"parents": len(usable), "pairs": sum(v["pairs"] for v in usable),
                "macro_disagreement": statistics.mean(v["disagreement"] for v in usable) if usable else None}
        result[name] = item
    return result


def label_bucket(value):
    a = abs(value)
    return "<=128" if a <= 128 else "(128,1024]" if a <= 1024 else "(1024,8192]" if a <= 8192 else ">8192"


def strata(rows, aggregate):
    result = {}
    for dimension, getter in (("stm", lambda r: str(r["stm"])), ("ply_decade", lambda r: str(r["ply"]//10)),
                              ("teacher_abs", lambda r: label_bucket(r["teacher"]))):
        groups = defaultdict(list)
        for row in rows:
            groups[getter(row)].append(row)
        result[dimension] = {key: aggregate(group) for key, group in sorted(groups.items())}
    return result


def write_rows(path, rows):
    pending = path.with_suffix(path.suffix+".pending")
    pending.write_text("".join(json.dumps(r, separators=(",", ":"), allow_nan=False)+"\n" for r in rows), encoding="utf-8")
    pending.replace(path)


def analyse(args):
    started = time.monotonic()
    base, out = args.experiment_dir.resolve(), args.out_dir.resolve()
    if out.exists():
        raise ValueError("refusing to overwrite diagnosis directory; use a new --out-dir with the original --sample-manifest")
    plan, _ = distill18.verify_plan(base)
    summary = read(base/"summary.json")
    if not summary.get("experiment_complete") or summary["plan"]["sha256"] != file_identity(base/"experiment_plan.json")["sha256"]:
        raise ValueError("ticket-18 comparison is incomplete or belongs to another plan")
    frozen = vf.load_frozen(base/"frozen_validation.json")
    train_manifest = read(base/"train_manifest.json")
    d_contract = targets.load(base/"target_d.json")
    store = tl.LabelStore(tl.verify_identity(d_contract["teacher_manifest"]))
    if targets.pool_identity(train_manifest["files"], store) != d_contract["train_pool_sha256"]:
        raise ValueError("training pool changed")
    probe = tl.verify_identity(plan["probe"])
    tl.verify_identity(store.manifest["eval_source"])
    if store.manifest["probe"]["sha256"] != plan["probe"]["sha256"]:
        raise ValueError("teacher/probe identity mismatch")
    out.mkdir(parents=True)
    report = {"schema": SCHEMA, "complete": False, "diagnostic_only": True, "training_performed": False,
              "source": source_identity(), "script": file_identity(Path(__file__)), "inputs": {
                  "plan18": file_identity(base/"experiment_plan.json"), "summary18": file_identity(base/"summary.json"),
                  "frozen": file_identity(base/"frozen_validation.json"), "train_manifest": file_identity(base/"train_manifest.json"),
                  "teacher_manifest": d_contract["teacher_manifest"], "probe": plan["probe"]}}
    write_json(out/"report.json", report)
    try:
        if args.sample_manifest:
            sample = read(args.sample_manifest)
            if (sample.get("schema") != SCHEMA or sample["train_manifest_sha256"] != report["inputs"]["train_manifest"]["sha256"]
                    or sample["seed"] != args.seed or sample["requested_parents"] != args.train_parents):
                raise ValueError("sample manifest contract mismatch")
            sources = sample["train_sources"]
            # Manifest replay must equal the predeclared deterministic draw.
            if sources != select_training_sources(train_manifest, store, args.train_parents, args.seed):
                raise ValueError("sample manifest differs from deterministic draw")
        else:
            sources = select_training_sources(train_manifest, store, args.train_parents, args.seed)
        sample = {"schema": SCHEMA, "seed": args.seed, "requested_parents": args.train_parents,
                  "train_manifest_sha256": report["inputs"]["train_manifest"]["sha256"],
                  "selection": "shuffled source-hash/game; one record per game; alternating stm; no prediction/label-value selection",
                  "train_sources": sources, "validation_record_ids": frozen["split_identity"]["validation_record_ids"]}
        write_json(out/"sample_manifest.json", sample)
        train_cases = make_cases(sources, train_manifest, store, probe, args.timeout)
        val_cases = validation_cases(base, frozen, store)
        overlap = extended_overlap(train_cases, val_cases)
        cases = train_cases+val_cases
        write_rows(out/"cases.jsonl", cases)
        report.update(sample_manifest=file_identity(out/"sample_manifest.json"), cases=file_identity(out/"cases.jsonl"),
                      extended_d4=overlap, output_gain=d_contract["output_gain"],
                      loss_normalizer=d_contract["loss_normalizer"], baseline_definition="D target constants fitted on all valid training parents, applied unchanged to BOTH arms and to children; not a child-fitted optimum")
        write_json(out/"report.json", report)
        primary = [c for c in cases if c["bucket"] == "primary"]
        if not primary:
            raise ValueError("no primary cases")
        boards = [c["board"] for c in cases]
        stms = [c["stm"] for c in cases]
        child_meta = []
        for case in primary:
            for move, board in zip(case["moves"], child_boards(case)):
                boards.append(board)
                stms.append(1-case["stm"])
                child_meta.append((case, move))
        teacher = evaluate(probe, boards, stms, timeout=args.timeout)
        if teacher[:len(cases)] != [c["teacher_parent"] for c in cases]:
            raise ValueError("teacher parent cache differs from fresh C++ labels")
        samples = [(sparse(b), s, h, 0) for b, s, h in zip(boards, stms, teacher)]
        import torch
        torch.set_num_threads(args.threads)
        predictions, model_ids, parity = {}, {}, {}
        for arm in ("d", "s"):
            evidence = read(tl.verify_identity(summary["arms"][arm]["fidelity"]))
            if (evidence["candidate"]["sha256"] != summary["arms"][arm]["candidate"]["sha256"]
                    or evidence["frozen_validation"]["sha256"] != report["inputs"]["frozen"]["sha256"]):
                raise ValueError("ticket-18 summary/fidelity identity mismatch")
            checkpoint = tl.verify_identity(evidence["checkpoint"])
            weights_path = tl.verify_identity(evidence["candidate"])
            model, payload = export.load_model(checkpoint)
            target_id = file_identity(base/f"target_{arm}.json")
            if payload["target_contract_identity"]["sha256"] != target_id["sha256"] or payload["frozen_validation_identity"]["sha256"] != report["inputs"]["frozen"]["sha256"]:
                raise ValueError("checkpoint target/freeze identity mismatch")
            weights, meta = wf.read_weights(weights_path)
            raw = model.quantized_weights()
            if meta["version"] != 2 or meta["output_gain"] != d_contract["output_gain"] or any(list(raw[k]) != list(weights[k]) for k in ("W0","b0","W1","b1","W2","b2")):
                raise ValueError("candidate does not exactly preserve the QAT checkpoint")
            final = evaluate(probe, boards, stms, weights_path, "avx2", args.timeout)
            scalar = evaluate(probe, boards, stms, weights_path, "scalar", args.timeout)
            ste = model_predictions(model, samples, True)
            master = model_predictions(model, samples, False)
            if final != scalar or final != ste:
                raise ValueError("new parent/child C++ AVX2/scalar/STE parity failed")
            if not all(math.isfinite(v) for v in master):
                raise ValueError("non-finite master float output")
            predictions[f"{arm}_integer"] = final
            predictions[f"{arm}_master_float"] = master
            parity[arm] = {"exact": True, "positions": len(samples), "static_limit_hits": sum(abs(v) >= wf.STATIC_LIMIT for v in final),
                           "master_float_vs_integer": export.error_stats(master, final)}
            model_ids[arm] = {"checkpoint": evidence["checkpoint"], "candidate": evidence["candidate"], "fidelity18": summary["arms"][arm]["fidelity"]}
        parents = [{"key": c["key"], "split": c["split"], "game_key": c["game_key"], "stm": c["stm"],
                    "ply": c["ply"], "bucket": c["bucket"], "teacher": teacher[i],
                    "predictions": {n: v[i] for n, v in predictions.items()}} for i, c in enumerate(cases)]
        children = [{"key": f"{c['key']}:{m}", "parent_key": c["key"], "split": c["split"], "game_key": c["game_key"],
                     "stm": 1-c["stm"], "ply": c["ply"]+1, "teacher": teacher[len(cases)+i],
                     "predictions": {n: v[len(cases)+i] for n, v in predictions.items()}} for i, (c, m) in enumerate(child_meta)]
        offsets, cursor = {}, 0
        for case in primary:
            offsets[case["key"]] = (cursor, cursor+len(case["moves"]))
            cursor += len(case["moves"])
        local = []
        for case in primary:
            start, end = offsets[case["key"]]
            h = [r["teacher"] for r in children[start:end]]
            values = {name: local_scores(h, [r["predictions"][name] for r in children[start:end]], case["moves"].index(case["teacher_move"])) for name in predictions}
            local.append({"key": case["key"], "split": case["split"], "game_key": case["game_key"],
                          "stm": case["stm"], "ply": case["ply"], "teacher": case["teacher_parent"],
                          "eligible_for_followup_training": case.get("eligible_for_followup_training", False), "local": values})
        for name, rows in (("parents", parents), ("children", children), ("local", local)):
            write_rows(out/f"{name}.jsonl", rows)
        constants, normalizer = d_contract["constants"], d_contract["loss_normalizer"]
        aggregate = lambda rows: score_summary(rows, constants, normalizer)
        splits = {}
        clean_keys = {c["key"] for c in train_cases if c["eligible_for_followup_training"]}
        for split in ("train_sample", "validation", "train_followup_clean"):
            if split == "train_followup_clean":
                pr = [r for r in parents if r["key"] in clean_keys]
                cr = [r for r in children if r["parent_key"] in clean_keys]
                lr = [r for r in local if r["key"] in clean_keys]
            else:
                pr = [r for r in parents if r["split"] == split]
                cr = [r for r in children if r["split"] == split]
                lr = [r for r in local if r["split"] == split]
            pp = [r for r in pr if r["bucket"] == "primary"]
            splits[split] = {"parent_counts": dict(Counter(r["bucket"] for r in pr)),
                             "parents": aggregate(pr), "paired_primary_parents": aggregate(pp), "children": aggregate(cr),
                             "local": local_summary(lr), "parent_strata": strata(pr, aggregate),
                             "child_strata": strata(cr, aggregate), "local_strata": strata(lr, local_summary)}
        for arm in ("d", "s"):
            old = read(summary["arms"][arm]["fidelity"]["path"])
            current = splits["validation"]["local"][f"{arm}_integer"]["search_move_top1_tie_adjusted"]
            if not math.isclose(current, old["child_rankings"]["final_integer"]["top1_tie_adjusted"], rel_tol=1e-13, abs_tol=1e-15):
                raise ValueError("validation search Top-1 no longer reproduces ticket 18")
        report.update(complete=True, models=model_ids, parity=parity, splits=splits,
                      artifacts={name: file_identity(out/f"{name}.jsonl") for name in ("parents", "children", "local")},
                      elapsed_seconds=time.monotonic()-started,
                      interpretation={"teacher": "C++ handcrafted static evaluator; child label belongs to NEXT stm",
                          "s_arm": "S was trained on search scores; error against H here is transfer diagnosis, not its own training loss",
                          "master_float": "continuous side path of QAT master weights, NOT an independently trained floating model",
                          "causality": "Parent/child differences are descriptive and require matched strata and stage B/C controls; no unique root-cause claim",
                          "gate": "diagnostic only; original ticket-18 final integer search-move Top-1 thresholds unchanged"})
        write_json(out/"report.json", report)
        print(json.dumps({"complete": True, "out": str(out), "train_parents": len(train_cases), "validation_parents": len(val_cases),
                          "primary_parents": len(primary), "children": len(children), "extended_d4": overlap,
                          "diagnostic_only": True, "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))
        return report
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic()-started)
        write_json(out/"report.json", report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=ROOT/"data/m5/distill18")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-parents", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample-manifest", type=Path, help="replay an interrupted run's frozen selection into a NEW output directory")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.train_parents < 2 or args.train_parents % 2 or args.threads < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("even train-parents >=2, positive threads/timeout required")
    analyse(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
