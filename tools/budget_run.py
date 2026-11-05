#!/usr/bin/env python3
"""Run one user-started experiment under a persistent 2-hour wall-clock budget.

Example: python tools/budget_run.py --ledger data/m5/budget.json
  --seconds 120 --out-dir data/m5/run1 -- python trainer/diagnose.py ...

The ledger reserves the whole requested timeout before launch. A wrapper crash
therefore cannot silently restore spent budget. An exclusive lock rejects
concurrent wrappers; a stale lock requires human inspection, never auto-reset.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_support import ROOT, source_identity, write_json

BUDGET_SECONDS = 7200


class ProcessTree:
    """Own children for the entire job, including after the parent exits."""

    def __init__(self, argv, log):
        self.job = None
        self.process = None
        if os.name != "nt":
            self.process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"})
            return
        import ctypes
        from ctypes import wintypes as w

        class BasicLimits(ctypes.Structure):
            _fields_ = [("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64),
                        ("Flags", w.DWORD), ("MinWorkingSet", ctypes.c_size_t),
                        ("MaxWorkingSet", ctypes.c_size_t), ("ActiveProcesses", w.DWORD),
                        ("Affinity", ctypes.c_size_t), ("Priority", w.DWORD),
                        ("Scheduling", w.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("Basic", BasicLimits), ("Io", IoCounters),
                        ("ProcessMemory", ctypes.c_size_t), ("JobMemory", ctypes.c_size_t),
                        ("PeakProcessMemory", ctypes.c_size_t), ("PeakJobMemory", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        kernel.SetInformationJobObject.restype = w.BOOL
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.AssignProcessToJobObject.restype = w.BOOL
        kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        kernel.TerminateJobObject.restype = w.BOOL
        kernel.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int,
                                                      ctypes.c_void_p, w.DWORD, ctypes.c_void_p]
        kernel.QueryInformationJobObject.restype = w.BOOL
        kernel.CloseHandle.argtypes = [w.HANDLE]
        kernel.CloseHandle.restype = w.BOOL
        self.kernel = kernel
        self.job = kernel.CreateJobObjectW(None, None)
        if not self.job:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = ExtendedLimits()
            limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits),
                                                   ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            # Suspend before job assignment: no descendant can escape the job
            # during the gap between CreateProcess and AssignProcessToJobObject.
            self.process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, creationflags=0x4 | 0x08000000,
                env={**os.environ, "PYTHONUNBUFFERED": "1"})
            if not kernel.AssignProcessToJobObject(self.job, int(self.process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            resume = ctypes.WinDLL("ntdll").NtResumeProcess
            resume.argtypes, resume.restype = [w.HANDLE], w.LONG
            if resume(int(self.process._handle)) != 0:
                raise OSError("could not resume budgeted process")
        except BaseException:
            if self.process:
                self.process.kill()
                self.process.wait(timeout=5)
            self.close()
            raise

    def remaining_processes(self):
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes as w

            class Accounting(ctypes.Structure):
                _fields_ = [(name, ctypes.c_int64) for name in
                            ("User", "Kernel", "PeriodUser", "PeriodKernel")] + [
                            (name, w.DWORD) for name in
                            ("Faults", "Total", "Active", "Terminated")]
            info = Accounting()
            if not self.kernel.QueryInformationJobObject(self.job, 1, ctypes.byref(info),
                                                         ctypes.sizeof(info), None):
                raise ctypes.WinError(ctypes.get_last_error())
            return info.Active
        try:
            os.killpg(self.process.pid, 0)
            return 1
        except ProcessLookupError:
            return 0

    def close(self):
        if self.job:
            self.kernel.TerminateJobObject(self.job, 124)
            self.kernel.CloseHandle(self.job)
            self.job = None
        elif os.name != "nt" and self.process:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
        if self.process:
            self.process.wait(timeout=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("provide a command and finite positive --seconds")
    args.ledger = args.ledger.resolve()
    args.out_dir = args.out_dir.resolve()
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    lock = args.ledger.with_suffix(args.ledger.suffix + ".lock")
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        parser.error("budget is locked; inspect running/stale job before removing lock")
    with handle:
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            ledger = (json.loads(args.ledger.read_text(encoding="utf-8"))
                      if args.ledger.exists() else
                      {"budget_seconds": BUDGET_SECONDS, "used_seconds": 0.0, "runs": []})
            used = ledger["used_seconds"]
            if (ledger["budget_seconds"] != BUDGET_SECONDS or
                    not isinstance(used, (float, int)) or not math.isfinite(used) or used < 0):
                parser.error("invalid budget ledger")
            if args.seconds > BUDGET_SECONDS - used:
                parser.error(f"requested {args.seconds}s exceeds remaining {BUDGET_SECONDS - used:.3f}s")
            if args.out_dir.exists():
                parser.error("refusing to overwrite existing run directory")
            args.out_dir.mkdir(parents=True)
            run = {"command": command, "cwd": str(ROOT), "pid": os.getpid(),
                   "limit_seconds": args.seconds, "status": "running", "complete": False,
                   "out_dir": str(args.out_dir), "source": source_identity()}
            ledger["runs"].append(run)
            ledger["used_seconds"] = used + args.seconds  # crash-safe reservation
            write_json(args.ledger, ledger)
            write_json(args.out_dir / "run.json", run)
            started, tree, exit_code = time.monotonic(), None, 1
            try:
                with (args.out_dir / "console.log").open("w", encoding="utf-8") as log:
                    tree = ProcessTree(command, log)
                    run["child_pid"] = tree.process.pid
                    write_json(args.out_dir / "run.json", run)
                    exit_code = tree.process.wait(timeout=max(.001, args.seconds - (time.monotonic() - started)))
                    run["status"] = "completed" if exit_code == 0 else "failed"
                    run["complete"] = exit_code == 0
                    # Process-exit signalling can precede Windows job accounting
                    # updates (and the venv launcher briefly outlives Python).
                    grace = min(time.monotonic() + .2, started + args.seconds)
                    remaining = tree.remaining_processes()
                    while remaining and time.monotonic() < grace:
                        time.sleep(.01)
                        remaining = tree.remaining_processes()
                    run["surviving_descendants"] = remaining
                    if exit_code == 0 and run["surviving_descendants"]:
                        run["status"], run["complete"], exit_code = "incomplete_descendants", False, 3
            except subprocess.TimeoutExpired:
                run["status"], exit_code = "timeout", 124
            except KeyboardInterrupt:
                run["status"], exit_code = "interrupted", 130
            except Exception as error:
                run["status"], run["error"] = "failed", str(error)
                exit_code = 1
            finally:
                run["complete"] = run["status"] == "completed" and exit_code == 0
                if tree:
                    tree.close()
                elapsed = time.monotonic() - started
                run.update(elapsed_seconds=elapsed, exit_code=exit_code)
                ledger["used_seconds"] = used + elapsed
                write_json(args.ledger, ledger)
                write_json(args.out_dir / "run.json", run)
            print(f"{run['status']}: {args.out_dir / 'run.json'}")
            print(f"Budget used {ledger['used_seconds']:.3f}/{BUDGET_SECONDS}s")
            return exit_code if 0 <= exit_code <= 255 else 1
        finally:
            handle.close()
            lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
