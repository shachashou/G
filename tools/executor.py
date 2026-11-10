#!/usr/bin/env python3
"""Shared client for the authoritative executor (ticket 02).

The GUI, data collection, tactic-evidence review and match arbitration all
talk to the C++ engine's ``--executor`` JSON-lines mode through this client.
Renju legality is NEVER implemented here: every verdict (complete legal move
set, forbidden reasons, result) comes from the single C++ rules executor.

Usage::

    from executor import ExecutorClient
    with ExecutorClient("build/engine/gomoku_nnue.exe") as client:
        print(client.about()["eval_description"])
        client.new_game("renju")
        client.play(7, 7)
        print(client.legal_moves()["count"])
"""
from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any

__all__ = ["ExecutorClient", "ExecutorError"]


class ExecutorError(RuntimeError):
    """The executor returned an error or its process died."""


class ExecutorClient:
    """Synchronous JSON-lines client over one engine subprocess.

    Requests are serialised through a lock so concurrent callers (e.g. the GUI
    main thread and a background thinking thread) cannot interleave lines.
    ``close()`` can terminate the process while a ``think`` request is in
    flight, which is how the GUI stops a game without leaving workers behind.
    """

    def __init__(self, exe: str | Path, *engine_args: str) -> None:
        self.exe = str(exe)
        self.engine_args = [str(arg) for arg in engine_args]
        self._proc = subprocess.Popen(
            [self.exe, "--executor", *self.engine_args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._closed = False
        self._stderr_lines: list[str] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        try:
            for line in self._proc.stderr:
                self._stderr_lines.append(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass

    def diagnostics(self) -> str:
        """Engine-side diagnostics (e.g. a rejected NNUE weight file)."""
        return "\n".join(self._stderr_lines)

    def call(self, cmd: str, **params: Any) -> dict:
        """Send one request, return the response object."""
        request = {"cmd": cmd, **params}
        with self._lock:
            if self._closed:
                raise ExecutorError("executor client is closed")
            if self._proc.poll() is not None:
                raise ExecutorError(
                    f"executor process exited (code {self._proc.returncode})"
                )
            try:
                self._proc.stdin.write(json.dumps(request) + "\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
            except (OSError, ValueError) as exc:
                raise ExecutorError(
                    f"executor communication failed: {exc}"
                ) from exc
        if not line:
            raise ExecutorError("executor closed the stream without a reply")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExecutorError(f"invalid executor response: {line!r}") from exc
        if not isinstance(response, dict):
            raise ExecutorError(f"executor response is not an object: {line!r}")
        return response

    def about(self) -> dict:
        return self.call("about")

    def new_game(self, rule: str = "renju") -> dict:
        return self.call("new_game", rule=rule)

    def state(self) -> dict:
        return self.call("state")

    def legal_moves(self) -> dict:
        return self.call("legal_moves")

    def move_reason(self, x: int, y: int) -> dict:
        return self.call("move_reason", x=x, y=y)

    def play(self, x: int, y: int) -> dict:
        return self.call("play", x=x, y=y)

    def undo(self) -> dict:
        return self.call("undo")

    def rebuild(self, cells: list[int], side_to_move: str = "black",
                rule: str = "renju") -> dict:
        return self.call("rebuild", cells=cells,
                         side_to_move=side_to_move, rule=rule)

    def replay(self, moves: list[list[int]], rule: str = "renju") -> dict:
        return self.call("replay", moves=moves, rule=rule)

    def candidates(self, radius: int = 1) -> dict:
        return self.call("candidates", radius=radius)

    def think(self, *, depth: int | None = None,
              movetime_ms: int | None = None,
              vcf_policy: str | None = None,
              vcf_nodes: int | None = None,
              vcf_plies: int | None = None) -> dict:
        params: dict[str, int | str] = {}
        if depth is not None:
            params["depth"] = depth
        if movetime_ms is not None:
            params["movetime_ms"] = movetime_ms
        if vcf_policy is not None:
            params["vcf_policy"] = vcf_policy
        if vcf_nodes is not None:
            params["vcf_nodes"] = vcf_nodes
        if vcf_plies is not None:
            params["vcf_plies"] = vcf_plies
        return self.call("think", **params)

    def vcf(self, *, max_nodes: int | None = None,
            max_attack_plies: int | None = None,
            time_ms: int | None = None) -> dict:
        """Ask the engine's upstream VCF solver about the current position.

        The response's ``vcf`` object distinguishes ``proven``, ``disproven``,
        ``budget_exhausted``, ``interrupted``, ``unsupported_rule`` and
        ``invalid_input``; only ``proven`` carries a re-checked variation.
        """
        params: dict[str, int] = {}
        if max_nodes is not None:
            params["max_nodes"] = max_nodes
        if max_attack_plies is not None:
            params["max_attack_plies"] = max_attack_plies
        if time_ms is not None:
            params["time_ms"] = time_ms
        return self.call("vcf", **params)

    def close(self) -> None:
        """Stop the engine process. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        try:
            proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    def __enter__(self) -> "ExecutorClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
