#!/usr/bin/env python3
"""Adapter for the local Rapfi teacher (ticket 04).

Rapfi is the external teacher used to label startup positions for the value
network. This module is the *only* place that speaks to it:

- it verifies the installed executable, Renju weights and config against
  ``tools/rapfi.lock.json`` (release, SHA-256, version) before use;
- it keeps Renju selected (``INFO rule 2``) and one search thread;
- it sends a position with ``BOARD`` and reads the search report, so the label
  is the engine's **search score at a fixed budget** - not a static score and
  not the suggested move;
- it distinguishes a numeric score, a mate score, "no search score" (a
  position the engine wins without reporting), a timeout and an engine error.

The score is reported from the side-to-move perspective (verified against
Rapfi 0.43.01: a position won by the side to move reports ``+M``), and it is
normalized with the value contract recorded in the lock file
(``teacher-v1``: ``tanh(score / 600)``). Mate scores are never normalized.

Usage::

    from rapfi_teacher import discover_teacher, RapfiTeacher

    teacher = RapfiTeacher()
    with teacher:
        report = teacher.label(cells, side="black", budget_ms=800)
        print(report.status, report.raw_eval, report.suggested_move)

Command line::

    python tools/rapfi_teacher.py about
    python tools/rapfi_teacher.py probe H8,H7,I9,G7
"""

import argparse
import hashlib
import json
import math
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
LOCK_PATH = TOOLS_DIR / "rapfi.lock.json"
DEFAULT_INSTALL_DIR = TOOLS_DIR / "rapfi"

MATE_RE = re.compile(r"^([+-])M(\d+)$")
MOVE_RE = re.compile(r"^(\d{1,2}),(\d{1,2})$")
EVAL_RE = re.compile(r"Eval\s+(\S+)")
DEPTH_RE = re.compile(r"Depth\s+(\d+)-(\d+)")
PV_RE = re.compile(r"Eval\s+\S+\s+\|\s+Time\s+\S+\s+\|\s+(.+)$")
NAME_RE = re.compile(r'name="([^"]*)".*version="([^"]*)"')


class TeacherError(RuntimeError):
    """The teacher is missing, misconfigured or its process failed."""


@dataclass(frozen=True)
class TeacherIdentity:
    release: str
    published: str
    asset: str
    asset_url: str
    engine_file: str
    engine_sha256: str
    engine_version: str
    weights: dict[str, dict[str, str]]
    config_file: str
    config_sha256: str
    label_version: str
    normalization: str
    normalization_param: float
    default_budget_ms: int
    default_threads: int
    timeout_margin_ms: int

    @property
    def weights_sha256(self) -> str:
        """One hash for the weight set, so a sample row can carry it."""
        digest = hashlib.sha256()
        for key in sorted(self.weights):
            digest.update(key.encode())
            digest.update(self.weights[key]["sha256"].encode())
        return digest.hexdigest()

    def describe(self) -> str:
        return (f"Rapfi {self.release} ({self.engine_version}), "
                f"renju weights {self.weights['renju_black']['file']} / "
                f"{self.weights['renju_white']['file']}, "
                f"label {self.label_version}")


@dataclass
class TeacherReport:
    status: str  # ok | mate | no_score | timeout | error
    raw_eval: str | None = None
    raw_score: int | None = None
    mate_plies: int | None = None
    perspective: str = "side_to_move"
    suggested_move: tuple[int, int] | None = None
    pv: str | None = None
    depth: int | None = None
    elapsed_ms: int = 0
    budget_ms: int = 0
    threads: int = 1
    message: str = ""
    identity: TeacherIdentity | None = None

    def normalized(self) -> float | None:
        """Value in [-1, 1] for the side to move; None unless numeric score."""
        if self.status != "ok" or self.raw_score is None or self.identity is None:
            return None
        if self.identity.normalization != "tanh":
            raise TeacherError(
                f"unsupported normalization {self.identity.normalization!r}")
        return normalize_tanh(self.raw_score, self.identity.normalization_param)


def normalize_tanh(score: int, param: float) -> float:
    if param <= 0:
        raise TeacherError("normalization parameter must be positive")
    return math.tanh(float(score) / float(param))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path: Path | None = None) -> TeacherIdentity:
    lock_path = Path(path or LOCK_PATH)
    try:
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TeacherError(f"cannot read teacher lock {lock_path}: {exc}") from exc
    try:
        release = raw["release"]
        contract = raw["value_contract"]
        protocol = raw["protocol"]
        return TeacherIdentity(
            release=str(release["tag"]),
            published=str(release.get("published", "")),
            asset=str(release.get("asset", "")),
            asset_url=str(release.get("asset_url", "")),
            engine_file=str(raw["engine"]["file"]),
            engine_sha256=str(raw["engine"]["sha256"]).lower(),
            engine_version=str(raw["engine"].get("version", "")),
            weights={name: {"file": str(item["file"]),
                            "sha256": str(item["sha256"]).lower()}
                     for name, item in raw["weights"].items()},
            config_file=str(raw["config"]["file"]),
            config_sha256=str(raw["config"]["sha256"]).lower(),
            label_version=str(contract["label_version"]),
            normalization=str(contract["normalization"]),
            normalization_param=float(contract["normalization_param"]),
            default_budget_ms=int(raw["budget"]["timeout_turn_ms"]),
            default_threads=int(raw["budget"]["threads"]),
            timeout_margin_ms=int(protocol.get("timeout_margin_ms", 6000)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TeacherError(f"malformed teacher lock {lock_path}: {exc}") from exc


@dataclass(frozen=True)
class InstallCheck:
    ok: bool
    install_dir: Path
    engine_path: Path | None
    problems: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return f"teacher ready at {self.install_dir}"
        return "teacher unavailable: " + "; ".join(self.problems)


def check_install(identity: TeacherIdentity,
                  install_dir: Path | None = None,
                  engine_override: str | None = None) -> InstallCheck:
    """Verifies the installed files against the lock, without running them."""
    base = Path(install_dir or DEFAULT_INSTALL_DIR)
    problems: list[str] = []
    if not base.is_dir():
        return InstallCheck(False, base, None,
                            [f"install directory missing: {base}"])
    engine_path = (Path(engine_override) if engine_override
                   else base / identity.engine_file)
    if not engine_path.is_file():
        problems.append(f"engine missing: {engine_path}")
    else:
        actual = sha256_file(engine_path).lower()
        if actual != identity.engine_sha256:
            problems.append(
                f"engine hash mismatch: {engine_path.name} has {actual}, "
                f"lock expects {identity.engine_sha256}")
    for name, item in identity.weights.items():
        path = base / item["file"]
        if not path.is_file():
            problems.append(f"{name} weights missing: {path}")
            continue
        actual = sha256_file(path).lower()
        if actual != item["sha256"]:
            problems.append(
                f"{name} weights hash mismatch: {path.name} has {actual}, "
                f"lock expects {item['sha256']}")
    config = base / identity.config_file
    if not config.is_file():
        problems.append(f"config missing: {config}")
    else:
        actual = sha256_file(config).lower()
        if actual != identity.config_sha256:
            problems.append(
                f"config hash mismatch: {config.name} has {actual}, "
                f"lock expects {identity.config_sha256}")
    return InstallCheck(not problems, base, engine_path if not problems else None,
                        problems)


def discover_teacher(install_dir: Path | None = None,
                     engine_override: str | None = None) -> InstallCheck:
    identity = load_lock()
    return check_install(identity, install_dir, engine_override)


class _LineReader(threading.Thread):
    def __init__(self, stream, sink: "queue.Queue[str | None]") -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.sink = sink

    def run(self) -> None:
        try:
            for line in self.stream:
                self.sink.put(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
        finally:
            self.sink.put(None)


class RapfiTeacher:
    """One Rapfi process, usable for several positions (batch labelling)."""

    def __init__(self, identity: TeacherIdentity | None = None,
                 install_dir: Path | None = None,
                 engine_override: str | None = None,
                 budget_ms: int | None = None,
                 threads: int | None = None,
                 timeout_margin_ms: int | None = None) -> None:
        self.identity = identity or load_lock()
        self.install_dir = Path(install_dir or DEFAULT_INSTALL_DIR)
        self.engine_override = engine_override
        self.budget_ms = int(budget_ms or self.identity.default_budget_ms)
        self.threads = int(threads or self.identity.default_threads)
        self.margin_ms = int(timeout_margin_ms or self.identity.timeout_margin_ms)
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._stderr: queue.Queue = queue.Queue()
        self._stderr_lines: list[str] = []
        self._dead_reason = ""

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        check = check_install(self.identity, self.install_dir,
                              self.engine_override)
        if not check.ok:
            raise TeacherError(check.summary())
        assert check.engine_path is not None
        try:
            self._proc = subprocess.Popen(
                [str(check.engine_path)],
                cwd=str(self.install_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise TeacherError(f"cannot start Rapfi: {exc}") from exc
        self._lines = queue.Queue()
        self._stderr = queue.Queue()
        self._stderr_lines = []
        _LineReader(self._proc.stdout, self._lines).start()
        _LineReader(self._proc.stderr, self._stderr).start()
        self._dead_reason = ""
        self._write("START 15")
        try:
            self._wait_ok(time.monotonic() + self.margin_ms / 1000.0)
        except TimeoutError as exc:
            self._kill()
            raise TeacherError(f"Rapfi did not finish starting: {exc}") from exc
        self._write("INFO rule 2")
        self._write(f"INFO thread_num {self.threads}")

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    proc.stdin.write("END\n")
                    proc.stdin.flush()
                except (OSError, ValueError):
                    pass
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass

    def __enter__(self) -> "RapfiTeacher":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stderr_text(self) -> str:
        while True:
            try:
                line = self._stderr.get_nowait()
            except queue.Empty:
                break
            if line is None:
                continue
            self._stderr_lines.append(line)
        return "\n".join(self._stderr_lines)

    # -- protocol --------------------------------------------------------

    def _write(self, line: str) -> None:
        if self._proc is None or self._proc.poll() is not None:
            raise TeacherError(self._dead_reason or "Rapfi process is not running")
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise TeacherError(f"cannot write to Rapfi: {exc}") from exc

    def _next_line(self, deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Rapfi did not answer in time")
        try:
            line = self._lines.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("Rapfi did not answer in time") from exc
        if line is None:
            self._dead_reason = (
                "Rapfi exited: " + (self.stderr_text().strip() or "no stderr"))
            raise TeacherError(self._dead_reason)
        return line

    def _wait_ok(self, deadline: float) -> None:
        while True:
            line = self._next_line(deadline)
            if line.strip() == "OK":
                return
            if line.startswith("ERROR"):
                raise TeacherError(f"Rapfi rejected the setup: {line}")

    def about(self) -> dict[str, str]:
        """Reads the engine banner and verifies it against the lock."""
        self.start()
        deadline = time.monotonic() + self.margin_ms / 1000.0
        self._write("ABOUT")
        while True:
            line = self._next_line(deadline).strip()
            match = NAME_RE.search(line)
            if match:
                name, version = match.group(1), match.group(2)
                if version != self.identity.engine_version:
                    raise TeacherError(
                        f"engine version {version!r} does not match the lock "
                        f"{self.identity.engine_version!r}")
                return {"name": name, "version": version}
            if line.startswith("ERROR"):
                raise TeacherError(f"Rapfi rejected ABOUT: {line}")

    def label(self, cells: list[int], *, side: str,
              budget_ms: int | None = None) -> TeacherReport:
        """Asks the teacher to search the position and reports its score."""
        if len(cells) != 225 or any(c not in (0, 1, 2) for c in cells):
            raise TeacherError("cells must be 225 values of 0/1/2")
        if side not in ("black", "white"):
            raise TeacherError("side must be black or white")
        self.start()
        budget = int(budget_ms or self.budget_ms)
        started = time.monotonic()
        deadline = started + budget / 1000.0 + self.margin_ms / 1000.0
        report = TeacherReport(status="error", budget_ms=budget,
                               threads=self.threads, identity=self.identity)
        try:
            self._write(f"INFO timeout_turn {budget}")
            self._write("BOARD")
            for index, value in enumerate(cells):
                if value:
                    x, y = index % 15, index // 15
                    self._write(f"{x},{y},{value}")
            self._write("DONE")
            eval_text: str | None = None
            pv: str | None = None
            depth: int | None = None
            move: tuple[int, int] | None = None
            while True:
                line = self._next_line(deadline).strip()
                if not line:
                    continue
                if line.startswith("ERROR"):
                    report.message = line
                    return report
                move_match = MOVE_RE.match(line)
                if move_match:
                    move = (int(move_match.group(1)), int(move_match.group(2)))
                    break
                eval_match = EVAL_RE.search(line)
                if eval_match:
                    eval_text = eval_match.group(1)
                    depth_match = DEPTH_RE.search(line)
                    if depth_match:
                        depth = int(depth_match.group(2))
                    pv_match = PV_RE.search(line)
                    if pv_match:
                        pv = pv_match.group(1).strip()
            report.elapsed_ms = int((time.monotonic() - started) * 1000)
            report.suggested_move = move
            report.pv = pv
            report.depth = depth
            report.raw_eval = eval_text
            if eval_text is None:
                report.status = "no_score"
                report.message = ("no search score reported "
                                  "(the engine found an immediate result)")
                return report
            mate = MATE_RE.match(eval_text)
            if mate:
                report.status = "mate"
                report.mate_plies = int(mate.group(2))
                if mate.group(1) == "-":
                    report.mate_plies = -report.mate_plies
                return report
            try:
                report.raw_score = int(eval_text)
            except ValueError:
                report.status = "error"
                report.message = f"unexpected eval token {eval_text!r}"
                return report
            report.status = "ok"
            return report
        except TimeoutError as exc:
            report.status = "timeout"
            report.message = str(exc)
            report.elapsed_ms = int((time.monotonic() - started) * 1000)
            self._kill()
            return report
        except TeacherError as exc:
            report.status = "error"
            report.message = str(exc)
            report.elapsed_ms = int((time.monotonic() - started) * 1000)
            self._kill()
            return report

    def _kill(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass


def label_once(cells: list[int], *, side: str, budget_ms: int | None = None,
               identity: TeacherIdentity | None = None,
               install_dir: Path | None = None,
               engine_override: str | None = None) -> TeacherReport:
    """One position with a fresh process; convenient for the GUI."""
    teacher = RapfiTeacher(identity=identity, install_dir=install_dir,
                           engine_override=engine_override,
                           budget_ms=budget_ms)
    with teacher:
        return teacher.label(cells, side=side, budget_ms=budget_ms)


def parse_moves(text: str) -> tuple[list[int], str]:
    """Builds a cell array and side to move from 'H8,H7,...' (black first)."""
    cells = [0] * 225
    side = "black"
    count = 0
    for token in text.replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "pass":
            count += 1
        else:
            x = ord(token[0].upper()) - ord("A")
            y = int(token[1:]) - 1
            color = 1 if count % 2 == 0 else 2
            cells[y * 15 + x] = color
            count += 1
        side = "white" if count % 2 == 1 else "black"
    return cells, side


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    about = sub.add_parser("about", help="verify the install and print its banner")
    about.add_argument("--install-dir", default=None)
    probe = sub.add_parser("probe", help="label one position given as moves")
    probe.add_argument("moves", help="H8,H7,I9,... (black first)")
    probe.add_argument("--budget-ms", type=int, default=None)
    probe.add_argument("--install-dir", default=None)
    args = parser.parse_args(argv)

    try:
        identity = load_lock()
        if args.command == "about":
            check = check_install(identity,
                                  Path(args.install_dir) if args.install_dir else None)
            print(check.summary())
            print(identity.describe())
            if not check.ok:
                return 2
            teacher = RapfiTeacher(identity=identity,
                                   install_dir=Path(args.install_dir)
                                   if args.install_dir else None)
            with teacher:
                print("banner:", teacher.about())
            return 0

        cells, side = parse_moves(args.moves)
        report = label_once(cells, side=side, budget_ms=args.budget_ms,
                            identity=identity,
                            install_dir=Path(args.install_dir)
                            if args.install_dir else None)
        print(json.dumps({
            "status": report.status,
            "raw_eval": report.raw_eval,
            "raw_score": report.raw_score,
            "mate_plies": report.mate_plies,
            "perspective": report.perspective,
            "value": report.normalized(),
            "suggested_move": report.suggested_move,
            "depth": report.depth,
            "elapsed_ms": report.elapsed_ms,
            "message": report.message,
        }, ensure_ascii=False, indent=2))
        return 0 if report.status in ("ok", "mate", "no_score") else 1
    except TeacherError as exc:
        print(f"rapfi_teacher: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
