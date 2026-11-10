"""Atomic report writes remain safe with Windows readers and concurrent writers."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import write_json


def windows_reader(path):
    """Return a closer for a real reader that does not share DELETE."""
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
        wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return lambda: kernel.CloseHandle(handle)


class ReportWrites(unittest.TestCase):
    def test_concurrent_writers_publish_whole_documents(self):
        with tempfile.TemporaryDirectory(prefix="report-writers-") as raw:
            path = Path(raw) / "state.json"
            write_json(path, {"writer": -1, "sequence": 0})
            barrier = threading.Barrier(4)
            errors = []
            def writer(index):
                try:
                    barrier.wait()
                    for sequence in range(20):
                        write_json(path, {"writer": index, "sequence": sequence,
                                         "data": [index] * 100})
                except Exception as exc:
                    errors.append(exc)
            threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertFalse(errors, errors)
            result = json.loads(path.read_text())
            self.assertEqual(result["sequence"], 19)
            self.assertEqual(result["data"], [result["writer"]] * 100)
            self.assertEqual(list(Path(raw).iterdir()), [path])

    @unittest.skipUnless(os.name == "nt", "Windows sharing semantics")
    def test_brief_reader_does_not_abort_state_update(self):
        with tempfile.TemporaryDirectory(prefix="state-writer-") as raw:
            path = Path(raw) / "state.json"
            write_json(path, {"step": 0})
            # A genuine Windows reader allowing read/write but not DELETE.
            close = windows_reader(path)
            released = threading.Event()
            def release():
                time.sleep(.15)
                close()
                released.set()
            reader = threading.Thread(target=release)
            reader.start()
            try:
                write_json(path, {"step": 120})
                self.assertEqual(json.loads(path.read_text())["step"], 120)
                self.assertTrue(released.is_set())
                self.assertEqual(list(Path(raw).iterdir()), [path])
            finally:
                reader.join()

    @unittest.skipUnless(os.name == "nt", "Windows sharing semantics")
    def test_permanent_lock_is_bounded_and_preserves_previous_document(self):
        with tempfile.TemporaryDirectory(prefix="state-permanent-lock-") as raw:
            path = Path(raw) / "state.json"
            write_json(path, {"step": 100})
            close = windows_reader(path)
            started = time.monotonic()
            try:
                with self.assertRaises(PermissionError):
                    write_json(path, {"step": 120})
            finally:
                close()
            self.assertLess(time.monotonic() - started, 4)
            self.assertEqual(json.loads(path.read_text()), {"step": 100})
            self.assertEqual(list(Path(raw).iterdir()), [path])

    def test_invalid_payload_does_not_touch_previous_document(self):
        with tempfile.TemporaryDirectory(prefix="state-invalid-") as raw:
            path = Path(raw) / "state.json"
            write_json(path, {"step": 100})
            with self.assertRaises(ValueError):
                write_json(path, {"loss": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"step": 100})
            self.assertEqual(list(Path(raw).iterdir()), [path])


if __name__ == "__main__":
    unittest.main(verbosity=2)
