#!/usr/bin/env python3
"""Batch engine-vs-engine match runner (ticket 06).

The runner is the referee: it owns the authoritative 15x15 board, syncs the
full position into each engine with Gomocup ``BOARD`` before every move, and
validates / times every reply. Two engine builds can therefore be compared
under identical search settings, with colours alternating and randomised
openings. Engines never need to agree on incremental state.

Every game's full move sequence (including the random opening) is written to a
JSON file so results can be replayed and reviewed later.

Example (PowerShell, using the project venv)::

  .venv\\Scripts\\python.exe tools/match_runner.py `
      --engine-a "build/engine/Release/gomoku_nnue.exe --depth 2" `
      --engine-b "build/engine/Release/gomoku_nnue.exe --random" `
      --games 10 --rule 2 --opening-plies 2 --seed 7 `
      --output data/match_results.json --min-win-rate 0.65

``--min-win-rate`` turns the run into a strength gate (exit code 2 on failure),
which is what history-evaluation iterations use. The gate compares the score rate
``(wins + 0.5 * draws) / games``.
"""

import argparse
import contextlib
from decimal import Decimal
import json
import math
import queue
import random
import re
import shlex
import subprocess
import sys
import threading
import time
import shutil
from dataclasses import dataclass
from pathlib import Path

from run_support import file_identity, source_identity, write_json

SIZE = 15
CELLS = SIZE * SIZE
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))
MOVE_RE = re.compile(r"(\d+),(\d+)")
# With <= 8 random plies there are <= 4 black stones. A Renju foul
# (overline / 4-4 / 3-3) needs at least 5 black stones, so every randomly
# generated opening is legal without duplicating the engine's foul rules.
MAX_OPENING_PLIES = 8


def gate_result(wins: int, draws: int, games: int, threshold: float,
                strict: bool = False) -> dict:
    """Evaluate original counts, not the rounded display score."""
    points = Decimal(2 * wins + draws)
    required = Decimal(str(threshold)) * (2 * games)
    return {"threshold": threshold, "strict": strict,
            "passed": points > required if strict else points >= required}


class EngineFailure(Exception):
    """An engine failed to produce a legal move in time."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


@dataclass(frozen=True)
class EngineSpec:
    label: str
    command: str
    argv: list[str]


@dataclass(frozen=True)
class MatchConfig:
    engine_a: EngineSpec
    engine_b: EngineSpec
    games: int
    rule: int
    opening_plies: int
    opening_region: int
    seed: int
    move_timeout: float
    timeout_turn: int
    max_plies: int
    output: Path
    min_win_rate: float | None
    paired_openings: bool = False
    openings: Path | None = None
    freeze_openings: Path | None = None
    strict_gate: bool = False
    verify_backend: bool = False


class InvalidMatch(Exception):
    """Invalid experiment configuration/backend; never a valid strength result."""


def opening_manifest(cfg: MatchConfig) -> dict:
    count = cfg.games // 2 if cfg.paired_openings else cfg.games
    if cfg.openings:
        manifest = json.loads(cfg.openings.read_text(encoding="utf-8"))
        if (manifest.get("seed") != cfg.seed or
                manifest.get("plies") != cfg.opening_plies or
                manifest.get("region") != cfg.opening_region):
            raise InvalidMatch("frozen opening seed/plies/region mismatch")
    else:
        positions, seen, attempt = [], set(), 0
        unique = math.perm(cfg.opening_region ** 2, cfg.opening_plies) >= count
        while len(positions) < count:
            position = random_opening(random.Random(cfg.seed + attempt * 1_000_003),
                                      cfg.opening_plies, cfg.opening_region)
            attempt += 1
            key = tuple(position)
            if unique and key in seen:
                continue
            seen.add(key)
            positions.append(position)
        manifest = {"schema": 1, "seed": cfg.seed, "plies": cfg.opening_plies,
                    "region": cfg.opening_region, "positions": positions}
    positions = manifest.get("positions", [])
    if len(positions) != count:
        raise InvalidMatch("frozen opening count mismatch")
    lo = (SIZE - cfg.opening_region) // 2
    for position in positions:
        if (len(position) != cfg.opening_plies or
                any(not isinstance(p, (list, tuple)) or len(p) != 2 or
                    any(type(v) is not int or not lo <= v < lo + cfg.opening_region
                        for v in p) for p in position) or
                len({tuple(p) for p in position}) != len(position)):
            raise InvalidMatch("malformed frozen opening")
    return manifest


def engine_identity(spec: EngineSpec) -> dict:
    path = shutil.which(spec.argv[0]) or spec.argv[0]
    result = {"command": spec.command, "exe": file_identity(path)}
    if "--nnue" in spec.argv:
        result["weights"] = file_identity(spec.argv[spec.argv.index("--nnue") + 1])
    return result


def check_backend(engine, spec: EngineSpec) -> str:
    engine.send("ABOUT")
    about = engine.recv_line()
    expected = "nnue" if "--nnue" in spec.argv else "handcrafted"
    if f'eval="{expected}"' not in about:
        raise InvalidMatch(f"{spec.label}: expected {expected}, ABOUT={about!r}")
    if expected == "nnue" and 'nnue_backend="avx2"' not in about:
        raise InvalidMatch(f"{spec.label}: expected AVX2, ABOUT={about!r}")
    return about


def split_command(command: str) -> list[str]:
    """Split a Windows-friendly command string, keeping backslashes intact."""
    tokens = shlex.split(command, posix=False)
    argv = []
    for token in tokens:
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1]
        if token:
            argv.append(token)
    if not argv:
        raise ValueError("empty engine command")
    return argv


class Engine:
    """One engine process; a reader thread makes every read timeout-able."""

    def __init__(self, spec: EngineSpec, move_timeout: float):
        self.spec = spec
        self.move_timeout = move_timeout
        self.p = subprocess.Popen(
            spec.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        try:
            while True:
                line = self.p.stdout.readline()
                if not line:
                    break
                self._lines.put(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
        finally:
            self._lines.put(None)

    def send_lines(self, lines: list[str]) -> None:
        if self.p.poll() is not None:
            raise EngineFailure("crash", f"process exited with {self.p.returncode}")
        try:
            self.p.stdin.write("".join(line + "\n" for line in lines))
            self.p.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise EngineFailure("crash", f"stdin write failed: {exc}") from exc

    def send(self, line: str) -> None:
        self.send_lines([line])

    def recv_line(self) -> str:
        try:
            line = self._lines.get(timeout=self.move_timeout)
        except queue.Empty as exc:
            raise EngineFailure(
                "timeout", f"no reply within {self.move_timeout:.2f}s"
            ) from exc
        if line is None:
            raise EngineFailure("crash", f"stdout closed (exit {self.p.poll()})")
        return line

    def recv_move(self) -> tuple[int, int]:
        noise: list[str] = []
        while True:
            line = self.recv_line()
            match = MOVE_RE.fullmatch(line)
            if match:
                return int(match.group(1)), int(match.group(2))
            if line.startswith("ERROR") or line == "UNKNOWN":
                raise EngineFailure("bad-reply", f"engine replied {line!r}")
            noise.append(line)
            if len(noise) >= 8:
                raise EngineFailure("bad-reply", f"no move in 8 lines: {noise!r}")

    def start(self, rule: int, timeout_turn: int) -> None:
        self.send(f"START {SIZE}")
        line = self.recv_line()
        if line != "OK":
            raise EngineFailure("protocol-error", f"START {SIZE} -> {line!r}")
        lines = [f"INFO rule {rule}"]
        if timeout_turn > 0:
            lines.append(f"INFO timeout_turn {timeout_turn}")
        self.send_lines(lines)

    def sync_board(self, board: list[int]) -> None:
        lines = ["BOARD"]
        for y in range(SIZE):
            for x in range(SIZE):
                color = board[y * SIZE + x]
                if color:
                    lines.append(f"{x},{y},{color}")
        lines.append("DONE")
        self.send_lines(lines)

    def close(self) -> None:
        if self.p.poll() is not None:
            return
        # END closes the engine cleanly; a crashed engine may already have a
        # broken stdin, which is harmless here.
        with contextlib.suppress(EngineFailure, OSError, ValueError):
            self.send("END")
            self.p.stdin.close()
        try:
            self.p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                self.p.kill()
            self.p.wait(timeout=2)


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < SIZE and 0 <= y < SIZE


def move_state(board: list[int], x: int, y: int, color: int, rule: int) -> str:
    """Classify the last move: 'win', 'foul-overline' or 'ok'.

    Mirrors the engine's rule core: five-in-a-row takes priority, black needs
    exactly five under Renju (an overline is a foul), white and freestyle win
    with five or more. Renju 3-3 / 4-4 foul filtering is delegated to the
    engines' own ``isPlayable`` (every build shares the same rule core), so the
    runner does not duplicate that contract in Python.
    """
    lengths = []
    for dx, dy in DIRS:
        length = 1
        for sx, sy in ((dx, dy), (-dx, -dy)):
            nx, ny = x + sx, y + sy
            while in_bounds(nx, ny) and board[ny * SIZE + nx] == color:
                length += 1
                nx += sx
                ny += sy
        lengths.append(length)
    if 5 in lengths:
        return "win"
    longest = max(lengths)
    if longest > 5:
        if rule == 2 and color == 1:
            return "foul-overline"
        return "win"
    return "ok"


def random_opening(rng: random.Random, plies: int, region: int) -> list[tuple[int, int]]:
    """Pick distinct opening stones from the centred region (default 7x7)."""
    lo = (SIZE - region) // 2
    cells = [(x, y) for y in range(lo, lo + region) for x in range(lo, lo + region)]
    rng.shuffle(cells)
    return cells[:plies]


def color_name(color: int) -> str:
    return "black" if color == 1 else "white"


def play_game(cfg: MatchConfig, game_index: int, opening=None) -> dict:
    started = time.monotonic()
    a_is_black = game_index % 2 == 0
    rng = random.Random(cfg.seed + game_index * 1_000_003)
    board = [0] * CELLS
    moves: list[dict] = []
    anomalies: list[dict] = []
    engines: dict[str, Engine] = {}
    about: dict[str, str] = {}
    result: str | None = None
    reason: str | None = None
    engine_plies = 0

    def engine_for(color: int) -> str:
        if color == 1:
            return "A" if a_is_black else "B"
        return "B" if a_is_black else "A"

    def offender_loses(offender: str) -> str:
        return "B-win" if offender == "A" else "A-win"

    def anomaly(kind: str, detail: str, label: str, ply: int) -> None:
        anomalies.append(
            {"kind": kind, "engine": label, "ply": ply, "detail": detail}
        )

    try:
        for label, spec in (("A", cfg.engine_a), ("B", cfg.engine_b)):
            engines[label] = Engine(spec, cfg.move_timeout)
        for label, engine in engines.items():
            try:
                engine.start(cfg.rule, cfg.timeout_turn)
                if cfg.verify_backend:
                    about[label] = check_backend(engine, engine.spec)
            except EngineFailure as failure:
                if cfg.verify_backend:
                    raise InvalidMatch(f"{label}: startup verification failed: {failure}") from failure
                anomaly(failure.kind, failure.detail, label, 0)
                result = offender_loses(label)
                reason = f"{failure.kind} at startup"

        if result is None:
            for ply, (x, y) in enumerate(
                opening if opening is not None else
                random_opening(rng, cfg.opening_plies, cfg.opening_region)
            ):
                color = 1 if ply % 2 == 0 else 2
                board[y * SIZE + x] = color
                moves.append(
                    {"ply": len(moves) + 1, "move": f"{x},{y}", "x": x, "y": y,
                     "color": color_name(color), "engine": None, "opening": True}
                )

            while len(moves) < CELLS and engine_plies < cfg.max_plies:
                color = 1 if len(moves) % 2 == 0 else 2
                label = engine_for(color)
                ply = len(moves) + 1
                try:
                    engines[label].sync_board(board)
                    x, y = engines[label].recv_move()
                except EngineFailure as failure:
                    anomaly(failure.kind, failure.detail, label, ply)
                    result = offender_loses(label)
                    reason = f"{failure.kind}: {failure.detail}"
                    break

                if not in_bounds(x, y):
                    kind, detail = "illegal-move", f"out of bounds {x},{y}"
                elif board[y * SIZE + x] != 0:
                    kind, detail = "illegal-move", f"occupied point {x},{y}"
                else:
                    kind, detail = "", ""
                if kind:
                    anomaly(kind, detail, label, ply)
                    result = offender_loses(label)
                    reason = detail
                    break

                board[y * SIZE + x] = color
                moves.append(
                    {"ply": ply, "move": f"{x},{y}", "x": x, "y": y,
                     "color": color_name(color), "engine": label, "opening": False}
                )
                engine_plies += 1

                state = move_state(board, x, y, color, cfg.rule)
                if state == "win":
                    result = f"{engine_for(color)}-win"
                    reason = "five-in-row"
                    break
                if state == "foul-overline":
                    anomaly("illegal-move", f"Renju black overline at {x},{y}", label, ply)
                    result = offender_loses(label)
                    reason = "Renju black overline (forbidden)"
                    break

            if result is None:
                result = "draw"
                reason = "board full or --max-plies reached"
    finally:
        for engine in engines.values():
            engine.close()

    return {
        "game": game_index + 1,
        "pair": game_index // 2 + 1 if cfg.paired_openings else None,
        "about": about,
        "a_color": "black" if a_is_black else "white",
        "b_color": "white" if a_is_black else "black",
        "result": result,
        "winner": {"A-win": "A", "B-win": "B", "draw": None}[result],
        "reason": reason,
        "plies": len(moves),
        "engine_plies": engine_plies,
        "elapsed_s": round(time.monotonic() - started, 3),
        "moves": moves,
        "anomalies": anomalies,
    }


def run_matches(cfg: MatchConfig) -> dict:
    manifest = opening_manifest(cfg)
    provenance = {"A": engine_identity(cfg.engine_a),
                  "B": engine_identity(cfg.engine_b), "source": source_identity()}
    games = []
    invalid = None
    for index in range(cfg.games):
        opening_index = index // 2 if cfg.paired_openings else index
        try:
            game = play_game(cfg, index, manifest["positions"][opening_index])
        except (InvalidMatch, OSError) as error:
            invalid = str(error)
            break
        games.append(game)
        print(
            f"[{game['game']}/{cfg.games}] A={game['a_color']} B={game['b_color']}"
            f" | {game['result']} ({game['reason']}, {game['plies']} plies,"
            f" {game['elapsed_s']:.2f}s)",
            flush=True,
        )

    wins = sum(g["result"] == "A-win" for g in games)
    losses = sum(g["result"] == "B-win" for g in games)
    draws = len(games) - wins - losses
    anomaly_counts: dict[str, int] = {}
    for game in games:
        for item in game["anomalies"]:
            anomaly_counts[item["kind"]] = anomaly_counts.get(item["kind"], 0) + 1

    report = {
        "schema": 1,
        "complete": len(games) == cfg.games and invalid is None,
        "invalid_reason": invalid,
        "provenance": provenance,
        "openings": {"manifest": manifest,
                     **(file_identity(cfg.openings) if cfg.openings else {})},
        "config": {
            "engine_a": cfg.engine_a.command,
            "engine_b": cfg.engine_b.command,
            "games": cfg.games,
            "rule": cfg.rule,
            "opening_plies": cfg.opening_plies,
            "opening_region": cfg.opening_region,
            "seed": cfg.seed,
            "move_timeout": cfg.move_timeout,
            "timeout_turn": cfg.timeout_turn,
            "max_plies": cfg.max_plies,
            "min_win_rate": cfg.min_win_rate,
            "paired_openings": cfg.paired_openings,
            "verify_backend": cfg.verify_backend,
            "strict_gate": cfg.strict_gate,
        },
        "summary": {
            "games": len(games),
            "a_wins": wins,
            "a_losses": losses,
            "draws": draws,
            "win_rate": round((wins + 0.5 * draws) / len(games), 4) if games else 0,
            "decisive_win_rate": round(wins / len(games), 4) if games else 0,
            "a_as_black": sum(g["a_color"] == "black" for g in games),
            "anomaly_counts": anomaly_counts,
        },
        "games": games,
    }
    if cfg.min_win_rate is not None:
        report["gate"] = gate_result(wins, draws, len(games), cfg.min_win_rate,
                                     cfg.strict_gate)
        report["gate"]["passed"] &= report["complete"]
    return report


def print_report(report: dict, output: Path) -> None:
    summary = report["summary"]
    print()
    print(
        f"Summary: A {summary['a_wins']}W / {summary['draws']}D /"
        f" {summary['a_losses']}L in {summary['games']} games"
        f" | win_rate={summary['win_rate'] * 100:.2f}%"
        f" | decisive={summary['decisive_win_rate'] * 100:.2f}%"
    )
    counts = summary["anomaly_counts"]
    print("Anomalies: " + (", ".join(f"{k}={v}" for k, v in counts.items()) or "none"))
    print(f"Results: {output.resolve()}")


def parse_args(argv: list[str] | None = None) -> MatchConfig:
    parser = argparse.ArgumentParser(
        description="Batch Gomocup engine-vs-engine matches with win-rate gating.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--engine-a", default="gomoku_nnue.exe",
                        help='engine A command, e.g. "build/engine/Release/gomoku_nnue.exe --depth 2"')
    parser.add_argument("--engine-b", default="gomoku_nnue.exe",
                        help='engine B command, e.g. "build/engine/Release/gomoku_nnue.exe --random"')
    parser.add_argument("--games", type=int, default=10, help="number of games (colours alternate)")
    parser.add_argument("--rule", type=int, choices=(1, 2), default=2,
                        help="INFO rule: 1=freestyle, 2=Renju (default)")
    parser.add_argument("--opening-plies", type=int, default=2,
                        help=f"random opening plies, 0..{MAX_OPENING_PLIES} (default 2)")
    parser.add_argument("--opening-region", type=int, default=7,
                        help="odd side length of the centred opening region (default 7)")
    parser.add_argument("--seed", type=int, default=2026, help="opening RNG seed")
    parser.add_argument("--move-timeout", type=float, default=10.0,
                        help="per-move reply timeout in seconds (default 10)")
    parser.add_argument("--timeout-turn", type=int, default=0,
                        help="INFO timeout_turn in ms; 0 means do not send (default)")
    parser.add_argument("--max-plies", type=int, default=225,
                        help="maximum engine moves per game before a draw (default 225)")
    parser.add_argument("--output", default="match_results.json",
                        help="JSON result file (default match_results.json)")
    parser.add_argument("--min-win-rate", type=float, default=None,
                        help="strength gate on score rate; exit 2 when not met")
    parser.add_argument("--paired-openings", action="store_true")
    parser.add_argument("--openings", type=Path, help="frozen opening JSON")
    parser.add_argument("--freeze-openings", type=Path,
                        help="write opening JSON without running engines")
    parser.add_argument("--strict-gate", action="store_true", help="use > instead of >=")
    parser.add_argument("--verify-backend", action="store_true",
                        help="verify ABOUT eval/nnue_backend; a silent "
                             "handcrafted fallback invalidates the report")
    args = parser.parse_args(argv)

    if args.games < 1:
        parser.error("--games must be >= 1")
    if args.paired_openings and args.games % 2:
        parser.error("paired openings require an even game count")
    if not 0 <= args.opening_plies <= MAX_OPENING_PLIES:
        parser.error(f"--opening-plies must be 0..{MAX_OPENING_PLIES}")
    if args.opening_region < 1 or args.opening_region > SIZE or args.opening_region % 2 == 0:
        parser.error("--opening-region must be an odd number in 1..15")
    if args.opening_plies > args.opening_region ** 2:
        parser.error("opening region too small for requested plies")
    if args.move_timeout <= 0:
        parser.error("--move-timeout must be > 0")
    if args.max_plies < 1:
        parser.error("--max-plies must be >= 1")
    if args.min_win_rate is not None and not 0 <= args.min_win_rate <= 1:
        parser.error("--min-win-rate must be in 0..1")
    try:
        argv_a = split_command(args.engine_a)
        argv_b = split_command(args.engine_b)
    except ValueError as exc:
        parser.error(str(exc))

    return MatchConfig(
        engine_a=EngineSpec("A", args.engine_a, argv_a),
        engine_b=EngineSpec("B", args.engine_b, argv_b),
        games=args.games,
        rule=args.rule,
        opening_plies=args.opening_plies,
        opening_region=args.opening_region,
        seed=args.seed,
        move_timeout=args.move_timeout,
        timeout_turn=args.timeout_turn,
        max_plies=args.max_plies,
        output=Path(args.output),
        min_win_rate=args.min_win_rate,
        paired_openings=args.paired_openings,
        openings=args.openings,
        freeze_openings=args.freeze_openings,
        strict_gate=args.strict_gate,
        verify_backend=args.verify_backend,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    try:
        if cfg.freeze_openings:
            if cfg.freeze_openings.exists():
                raise InvalidMatch("refusing to overwrite frozen openings")
            write_json(cfg.freeze_openings, opening_manifest(cfg))
            return 0
        report = run_matches(cfg)
    except (InvalidMatch, OSError, ValueError, IndexError) as error:
        write_json(cfg.output, {"complete": False, "invalid_reason": str(error)})
        print(f"Invalid match: {error}", file=sys.stderr)
        return 3
    write_json(cfg.output, report)
    if not report["complete"]:
        print(f"Invalid match: {report['invalid_reason']}", file=sys.stderr)
        return 3
    print_report(report, cfg.output)

    if cfg.min_win_rate is not None:
        rate = report["summary"]["win_rate"]
        ok = report["gate"]["passed"]
        op = ">" if cfg.strict_gate else ">="
        print(f"Gate: win_rate {rate * 100:.2f}% {op} {cfg.min_win_rate * 100:.2f}% -> "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
