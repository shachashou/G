"""C++ handcrafted labels, indexed by the ORIGINAL binpack record number."""
from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
import subprocess
import sys

import binpack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_support import file_identity, source_identity, write_json


def verify_identity(identity: dict) -> Path:
    path = Path(identity["path"])
    actual = file_identity(path)
    if actual["sha256"] != identity["sha256"] or actual["bytes"] != identity["bytes"]:
        raise ValueError(f"identity mismatch: {path}")
    return path


def board_text(record) -> str:
    black, white = binpack.unpack_plane(record.black), binpack.unpack_plane(record.white)
    if any(b and w for b, w in zip(black, white)):
        raise ValueError(f"overlapping occupancy at record {record.index}")
    if sum(black) + sum(white) != record.ply:
        raise ValueError(f"ply mismatch at record {record.index}")
    return "".join("x" if b else "o" if w else "." for b, w in zip(black, white))


def probe_rows(probe: Path, mode: str, records, timeout: float = 120) -> list[dict]:
    records = list(records)
    payload = "".join(f"{r.index} {r.stm} {board_text(r)}\n" for r in records)
    proc = subprocess.run([str(probe.resolve()), mode, "--rule", "renju", "--radius", "1"],
                          input=payload, text=True, capture_output=True, timeout=timeout)
    if proc.returncode:
        raise ValueError(f"probe {mode} failed: {proc.stderr.strip()}")
    rows = [json.loads(line) for line in proc.stdout.splitlines()]
    if len(rows) != len(records) or [r.get("record_id") for r in rows] != [r.index for r in records]:
        raise ValueError("probe response length/order mismatch")
    for row in rows:
        if type(row.get("eval")) is not int or not -(1 << 31) <= row["eval"] < (1 << 31):
            raise ValueError("probe returned invalid int32 eval")
        if type(row.get("result")) is not int or row["result"] not in range(4):
            raise ValueError("probe returned invalid result")
    return rows


class LabelStore:
    def __init__(self, manifest_path: Path | str):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != 1 or self.manifest.get("complete") is not True:
            raise ValueError("incomplete/unsupported teacher manifest")
        if self.manifest.get("teacher_definition") != "cpp-evaluator-stm-v1":
            raise ValueError("unknown teacher definition")
        self.entries = {}
        for entry in self.manifest["files"]:
            if entry["dtype"] != "<i4":
                raise ValueError("teacher label dtype must be little-endian int32")
            source = verify_identity(entry["source"])
            raw = verify_identity(entry["labels"]).read_bytes()
            mask = verify_identity(entry["mask"]).read_bytes()
            count = entry["records"]
            if len(raw) != 4 * count or len(mask) != count or any(v not in (0, 1) for v in mask):
                raise ValueError("teacher label/mask length or value mismatch")
            with binpack.BinpackReader(source) as reader:
                if len(reader) != count:
                    raise ValueError("teacher source record count mismatch")
            values = array("i")
            if values.itemsize != 4:
                raise ValueError("platform int32 array unsupported")
            values.frombytes(raw)
            if sys.byteorder != "little":
                values.byteswap()
            key = str(source.resolve()).lower()
            if key in self.entries:
                raise ValueError("duplicate teacher source")
            self.entries[key] = (values, mask, entry)

    def for_source(self, path: Path | str):
        key = str(Path(path).resolve()).lower()
        if key not in self.entries:
            raise ValueError(f"missing teacher labels for {path}")
        return self.entries[key]


def build(data_paths, probe: Path, out_dir: Path, timeout=120):
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "complete": False, "teacher_definition": "cpp-evaluator-stm-v1",
                "perspective": "side-to-move", "units": "engine-score",
                "probe": file_identity(probe), "eval_source": file_identity(ROOT / "engine/src/eval.cpp"),
                "source": source_identity(), "batch_records": 2048, "timeout": timeout, "files": []}
    write_json(out_dir / "teacher_manifest.json", manifest)
    for number, path in enumerate(dict.fromkeys(Path(p).resolve() for p in data_paths)):
        values, mask = array("i"), bytearray()
        with binpack.BinpackReader(path) as reader:
            for start in range(0, len(reader), 2048):
                for row in probe_rows(probe, "labels", (reader[i] for i in range(start, min(start+2048, len(reader)))), timeout):
                    values.append(row["eval"])
                    mask.append(int(row["result"] == 0))
        if sys.byteorder != "little":
            values.byteswap()
        labels_path, mask_path = out_dir / f"labels_{number}.i32", out_dir / f"valid_{number}.u8"
        for target, content in ((labels_path, values.tobytes()), (mask_path, mask)):
            pending = target.with_suffix(target.suffix + ".pending")
            pending.write_bytes(content)
            pending.replace(target)
        manifest["files"].append({"source": file_identity(path), "records": len(mask), "dtype": "<i4",
                                  "labels": file_identity(labels_path), "mask": file_identity(mask_path),
                                  "valid_records": sum(mask)})
        write_json(out_dir / "teacher_manifest.json", manifest)
    manifest["complete"] = True
    write_json(out_dir / "teacher_manifest.json", manifest)
    return out_dir / "teacher_manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build"])
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    print(build(args.data, args.probe, args.out_dir, args.timeout))


if __name__ == "__main__":
    main()
