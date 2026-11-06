#!/usr/bin/env python3
"""End-to-end tests for ticket 07 (datagen worker + binpack read end).

Usage::

  .venv\\Scripts\\python.exe engine\\tests\\datagen_tests.py ^
      build\\engine\\Release\\datagen.exe trainer\\binpack.py tools\\datagen.py

A real C++ worker generates tiny shards; the Python read end checks the magic,
64-byte layout, WDL backfill, game boundaries and shard merging; the C++ worker
then replays sampled records to prove the stored search scores are reproducible.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)
        print(f"FAIL: {message}")


def run(argv: list[str], *, expect: int | None = 0) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [str(a) for a in argv], cwd=str(ROOT),
        capture_output=True, text=True, timeout=180,
    )
    if expect is not None:
        check(proc.returncode == expect,
              f"{Path(str(argv[0])).name} {' '.join(str(a) for a in argv[1:])}: "
              f"rc={proc.returncode} (expected {expect})\n"
              f"stdout={proc.stdout[-800:]}\nstderr={proc.stderr[-800:]}")
    return proc


def generate(exe: Path, out: Path, *, games: int = 1, shard: int = 0,
             seed: int = 777, depth: int = 2, max_plies: int = 40,
             opening_temp: float = 1.0, extra: list[str] | None = None
             ) -> subprocess.CompletedProcess:
    argv = [
        exe, "--out", out, "--games", games, "--seed", seed,
        "--shard-id", shard, "--depth", depth, "--max-plies", max_plies,
        "--opening-temp", opening_temp, "--opening-plies", 3,
        *(extra or []),
    ]
    return run(argv)


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 2
    exe = Path(argv[1]).resolve()
    check(exe.exists(), f"datagen exe exists: {exe}")
    sys.path.insert(0, str(Path(argv[2]).resolve().parent))
    import binpack

    tool = Path(argv[3]).resolve()
    python = sys.executable

    with tempfile.TemporaryDirectory(prefix="gomoku_datagen_") as raw_tmp:
        tmp = Path(raw_tmp)

        # 1. C++ write end: two independent shards, deterministic per (seed, shard).
        shard_a = tmp / "shard_0000.binpack"
        shard_b = tmp / "shard_0001.binpack"
        out_a = generate(exe, shard_a, games=2, shard=0, seed=777)
        out_b = generate(exe, shard_b, games=1, shard=1, seed=888)
        check("STATS" in out_a.stdout, "worker prints a STATS line")
        check("records=" in out_a.stdout, "STATS carries record count")

        header_a = binpack.read_header(shard_a)
        header_b = binpack.read_header(shard_b)
        check(header_a.version == 1 and header_a.record_size == 64,
              "header version / record size")
        check(header_a.search_depth == 2 and header_a.shard_id == 0,
              "header stores search depth and shard id")
        check(header_a.base_seed == 777 and header_b.base_seed == 888,
              "each worker keeps its own RNG seed")
        check(header_a.opening_temp_x100 == 100, "header stores opening temperature")
        check(shard_a.stat().st_size == 64 + 64 * header_a.record_count,
              "file size == 64-byte header + 64-byte records")

        stats_a = binpack.validate_file(shard_a)
        check(not stats_a["errors"], f"shard A validates: {stats_a['errors'][:3]}")
        check(stats_a["scanned"] == header_a.record_count, "all A records scanned")
        check(stats_a["games"] == header_a.game_count, "A game count matches header")
        check(stats_a["wdl_unknown"] == 0, "no unfilled WDL placeholders")
        check(not binpack.validate_file(shard_b)["errors"], "shard B validates")

        with binpack.BinpackReader(shard_a) as reader:
            check(len(reader) == header_a.record_count, "mmap record count")
            first = reader[0]
            check(first.game_start, "first record of first game has game-start flag")
            check(first.ply >= 3, "random-opening positions are not recorded")
            check(first.stm == (first.ply & 1), "side to move matches ply parity")
            check(len(first.feature_indices()) == first.ply,
                  "450-dim feature indices cover every stone exactly once")
            check(all(0 <= i < 450 for i in first.feature_indices()),
                  "feature indices stay in the 450-dim range")
            opening = [r for r in reader.iter_records() if r.ply == 3]
            check(bool(opening), "some records start right after the 3-ply opening")
            for record in opening:
                cells = list(record.black_cells()) + list(record.white_cells())
                check(all(4 <= (c % 15) <= 10 and 4 <= (c // 15) <= 10
                          for c in cells),
                      "all opening stones lie in the centre 7x7")
                check(len(cells) == 3, "opening position has exactly 3 stones")
            games = list(reader.iter_games())
            check(len(games) == header_a.game_count, "iter_games groups by game-start flag")
            check(all(g[-1].game_end for g in games), "every game ends with game-end flag")
            check(all(r.wdl in (0, 1, 2) for g in games for r in g),
                  "every WDL is resolved")

        # 2. Magic / version validation.
        bad_magic = tmp / "bad_magic.binpack"
        shutil.copyfile(shard_a, bad_magic)
        with open(bad_magic, "r+b") as handle:
            handle.seek(0)
            handle.write(b"X")
        try:
            binpack.BinpackReader(bad_magic).close()
            check(False, "corrupted magic must be rejected")
        except binpack.BinpackError as exc:
            check("magic" in str(exc), f"magic error is descriptive: {exc}")
        check(run([python, tool, "validate", "--input", bad_magic], expect=3).returncode == 3,
              "validate CLI rejects bad magic")

        # 3. Shard merge keeps a single valid header.
        merged = tmp / "train.binpack"
        info = binpack.merge_files([shard_a, shard_b], merged)
        merged_header = binpack.read_header(merged)
        check(merged_header.record_count == header_a.record_count + header_b.record_count,
              "merged record count is the sum")
        check(merged_header.game_count == header_a.game_count + header_b.game_count,
              "merged game count is the sum")
        check(merged_header.shard_id == binpack.MERGED_SHARD_ID,
              "merged file uses the merged shard id")
        check(merged.stat().st_size == 64 + 64 * merged_header.record_count,
              "merged file size is aligned")
        merged_stats = binpack.validate_file(merged)
        check(not merged_stats["errors"], f"merged file validates: {merged_stats['errors'][:3]}")
        check(run([python, tool, "validate", "--input", merged], expect=0).returncode == 0,
              "validate CLI accepts merged file")
        merged_cli = tmp / "merged_cli.binpack"
        run([python, tool, "merge", "--inputs", str(tmp / "shard_*.binpack"),
             "--output", merged_cli], expect=0)
        check(binpack.read_header(merged_cli).record_count == merged_header.record_count,
              "merge CLI expands shard globs and writes a valid header")

        # 4. Score replay: C++ re-searches sampled boards from the two bit planes.
        verified = run([python, tool, "verify", "--exe", exe,
                        "--input", merged, "--samples", 8])
        check("mismatches=0" in verified.stdout,
              f"score replay matches stored scores: {verified.stdout.strip()}")
        check("errors=0" in verified.stdout, "replay found no structural errors")

        # A tampered score must be caught by the replay check.
        tampered = tmp / "tampered.binpack"
        shutil.copyfile(merged, tampered)
        with binpack.BinpackReader(tampered) as reader:
            original = reader[0].score
        replacement = 12345 if original != 12345 else 5432
        with open(tampered, "r+b") as handle:
            handle.seek(binpack.HEADER_SIZE + binpack.SCORE_OFFSET)
            handle.write(replacement.to_bytes(2, "little", signed=True))
        bad_verify = run([python, tool, "verify", "--exe", exe,
                          "--input", tampered, "--samples", 8], expect=None)
        check(bad_verify.returncode != 0 and "mismatches=0" not in bad_verify.stdout,
              "tampered score is detected by replay")

        # 5. Multi-process orchestration: shards stay independent.
        multi = tmp / "multi"
        manifest_path = multi / "manifest.json"
        run([python, tool, "run", "--exe", exe,
             "--workers", 2, "--games-per-worker", 1, "--depth", 2,
             "--max-plies", 20, "--seed", 999, "--out-dir", multi,
             "--target", 1000, "--budget-hours", 0.001])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = manifest["shards"]
        check(len(shards) == 2, "two workers were launched")
        check(all(s["returncode"] == 0 for s in shards), "both workers succeeded")
        check(len({s["path"] for s in shards}) == 2, "workers wrote distinct shard files")
        check(all(not binpack.validate_file(s["path"])["errors"] for s in shards),
              "both parallel shards are valid")
        check(manifest["totals"]["records"] == sum(s["records"] for s in shards),
              "manifest aggregates shard record counts")
        check(manifest["totals"]["positions_per_second"] > 0,
              "throughput report contains a positive positions/second")
        check(manifest["totals"]["elapsed_seconds"] >=
              max(s["stats"]["seconds"] for s in shards),
              "wall clock covers the full worker runtime, not just spawn time")
        check(manifest["totals"]["eta_hours"] is not None,
              "throughput report estimates time to the target")
        check(run([python, tool, "report", "--manifest", manifest_path],
                  expect=0).returncode == 0, "report CLI prints the manifest")

        # 6. Ply cap boundary: a game cut short at the cap is a draw, records
        #    still get their result backfilled and game flags set.
        cap = tmp / "cap.binpack"
        generate(exe, cap, games=1, shard=2, seed=555, depth=2, max_plies=4)
        with binpack.BinpackReader(cap) as reader:
            check(len(reader) == 1, "one recorded position before the ply cap")
            record = reader[0]
            check(record.ply == 3, "record sits at ply 3")
            check(record.wdl == binpack.WDL_DRAW, "ply-cap game is a draw")
            check(record.game_start and record.game_end,
                  "single-record game has both boundary flags")
        check(not binpack.validate_file(cap)["errors"], "ply-cap shard validates")

        # 7. Same seed + greedy opening reproduces byte-identical records.
        repro_a = tmp / "repro_a.binpack"
        repro_b = tmp / "repro_b.binpack"
        generate(exe, repro_a, games=1, shard=4, seed=4242, depth=2,
                 max_plies=12, opening_temp=0.0)
        generate(exe, repro_b, games=1, shard=4, seed=4242, depth=2,
                 max_plies=12, opening_temp=0.0)
        body_a = repro_a.read_bytes()[binpack.HEADER_SIZE:]
        body_b = repro_b.read_bytes()[binpack.HEADER_SIZE:]
        check(body_a == body_b and bool(body_a),
              "identical seed/config reproduces identical records")

        # 8. NNUE self-play: the worker loads a GMNNUE01 file and self-plays
        #    with NNUE evaluation; replay with the SAME weights reproduces the
        #    stored scores, while a handcrafted replay reports mismatches.
        import weights_format  # noqa: E402  (trainer/ is on sys.path by design)
        nnue_weights = tmp / "test_weights.nnue"
        weights_format.write_weights(nnue_weights, {
            "W0": [0] * weights_format.W0_LEN,
            "b0": [0] * weights_format.B0_LEN,
            "W1": [0] * weights_format.W1_LEN,
            "b1": [0] * weights_format.B1_LEN,
            "W2": [0] * weights_format.W2_LEN,
            "b2": [0] * weights_format.B2_LEN,
        })
        nnue_data = tmp / "nnue.binpack"
        nnue_out = generate(exe, nnue_data, games=1, shard=5, seed=1212,
                            depth=2, max_plies=20,
                            extra=["--nnue", str(nnue_weights)])
        check("STATS" in nnue_out.stdout, "NNUE worker prints STATS")
        check(not binpack.validate_file(nnue_data)["errors"],
              "NNUE self-play shard validates")
        nnue_verify = run([python, tool, "verify", "--exe", exe,
                           "--input", nnue_data, "--samples", 8,
                           "--nnue", nnue_weights])
        check("mismatches=0" in nnue_verify.stdout,
              f"NNUE score replay matches stored scores: {nnue_verify.stdout.strip()}")
        # A missing --nnue on NNUE data must be reported as mismatches, never
        # silently accepted as a handcrafted replay.
        nnue_verify_bad = run([python, tool, "verify", "--exe", exe,
                               "--input", nnue_data, "--samples", 8], expect=None)
        check("mismatches=0" not in nnue_verify_bad.stdout,
              "handcrafted replay of NNUE data reports mismatches")

    if FAILURES:
        print(f"\n{len(FAILURES)} datagen test(s) failed")
        return 1
    print("datagen_tests: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
