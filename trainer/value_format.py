#!/usr/bin/env python3
"""New value-network deployment format + dependency-free integer reference (ticket 06).

Architecture (fixed, self-describing):

    450 absolute-colour features -> accumulator 1024 -> 64 -> 32 -> 1
    single output; value = tanh(z) in BLACK-absolute space [-1, 1]

``teacher-v1`` labels are side-to-move values, so training converts them with
the side sign before they reach this contract:
``target_black = value_stm * (+1 if black to move else -1)``.

Integer scheme (fixed-point, one shift per layer):

    S = 2^ea  (ea stored in the header, default 9)
    W int16 = round(w * S), b int32 = round(b * S)
    acc     = b0 + sum W0[feature]                  (int64)
    a0      = clamp(acc, 0, ACT_MAX)
    z1      = round_div(sum a0*W1, S) + b1; a1 = clamp(z1, 0, ACT_MAX)
    z2      = round_div(sum a1*W2, S) + b2; a2 = clamp(z2, 0, ACT_MAX)
    z       = round_div(sum a2*W3, S) + b3          (fixed point, /S = real z)
    value   = tanh(z / S)

File layout: 64-byte header, then a JSON metadata block, then the int16/int32
body. The header's FNV-1a checksum covers metadata + body. The metadata must
declare the same architecture, rule, feature spec and value contract as the
header; anything else is rejected instead of being reinterpreted.

Usage::

    from value_format import pack_value_weights, unpack_value_weights, infer_int

    blob = pack_value_weights(weights, metadata=meta)
    weights, meta = unpack_value_weights(blob)
    fixed_point_z = infer_int(weights, sparse_indices)
"""
from __future__ import annotations

import json
import math
import struct
from array import array
from collections.abc import Iterable, Sequence
from pathlib import Path

try:  # optional fast path for fidelity measurement; the format is pure stdlib
    import numpy as _np  # type: ignore
except ImportError:  # pragma: no cover
    _np = None  # type: ignore

MAGIC = b"GMVALUE1"
VERSION = 1
HEADER_SIZE = 64

N_FEATURES = 450
N_ACC = 1024
N_H1 = 64
N_H2 = 32
N_OUT = 1

RULE_RENJU = 2
FEATURE_SPEC = 1  # absolute-colour 450 (features.py)
VALUE_CONTRACT = 1  # black-absolute value = tanh(z), teacher-v1 compatible

EA_DEFAULT = 9
ACT_MAX_DEFAULT = 8.0  # real activation ceiling

INT16_MIN, INT16_MAX = -32768, 32767
INT32_MIN, INT32_MAX = -2147483648, 2147483647

HEADER_STRUCT = struct.Struct("<8s14I")
assert HEADER_STRUCT.size == 64

BODY_SPEC = (
    ("W0", N_FEATURES * N_ACC, "h"),
    ("b0", N_ACC, "i"),
    ("W1", N_ACC * N_H1, "h"),
    ("b1", N_H1, "i"),
    ("W2", N_H1 * N_H2, "h"),
    ("b2", N_H2, "i"),
    ("W3", N_H2 * N_OUT, "h"),
    ("b3", N_OUT, "i"),
)

# Frozen before measuring: the exporter must record these and fail if the
# measured fidelity is worse. Changing them is a versioned contract change.
VALUE_TOLERANCE = {
    "mean_abs": 0.010,
    "p95_abs": 0.030,
    "max_abs": 0.100,
}
INTEGER_TOLERANCE = {
    "mean_abs": 0.002,
    "max_abs": 0.010,
}

_WEIGHT_KEYS = tuple(name for name, _, _ in BODY_SPEC)
_MATRIX_KEYS = {"W0", "W1", "W2", "W3"}
_BIAS_KEYS = {"b0", "b1", "b2", "b3"}


class ValueFormatError(Exception):
    """Malformed or incompatible value-model file."""


def fnv1a(data: bytes) -> int:
    value = 0x811C9DC5
    for byte in data:
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


def round_div(n: int, d: int) -> int:
    """Round n/d half away from zero (d > 0)."""
    if d <= 0:
        raise ValueError("divisor must be positive")
    sign = -1 if n < 0 else 1
    return sign * ((2 * abs(n) + d) // (2 * d))


def _array(values: Iterable[int], typecode: str, name: str, expected: int,
           lo: int, hi: int) -> array:
    numbers = [int(v) for v in values]
    if len(numbers) != expected:
        raise ValueFormatError(
            f"{name}: expected {expected} values, got {len(numbers)}")
    numbers = [max(lo, min(hi, v)) for v in numbers]
    return array(typecode, numbers)


def _little_endian(arr: array) -> bytes:
    out = array(arr.typecode, arr)
    if struct.pack("=H", 1) == b"\x01\x00":
        return out.tobytes()
    out.byteswap()
    return out.tobytes()


def _from_little_endian(raw: bytes, typecode: str) -> array:
    result = array(typecode)
    result.frombytes(raw)
    if struct.pack("=H", 1) != b"\x01\x00":
        result.byteswap()
    return result


def body_length() -> int:
    return sum(count * array(typecode).itemsize
               for _, count, typecode in BODY_SPEC)


def _validate_metadata(metadata: dict, ea: int) -> dict:
    required = ("model_id", "arch", "rule", "feature_spec", "value_contract",
                "label_versions", "training")
    for key in required:
        if key not in metadata:
            raise ValueFormatError(f"metadata is missing {key!r}")
    if list(metadata["arch"]) != [N_FEATURES, N_ACC, N_H1, N_H2, N_OUT]:
        raise ValueFormatError(
            f"metadata architecture {metadata['arch']} is not the value net")
    if metadata["rule"] != "renju":
        raise ValueFormatError(f"unsupported rule {metadata['rule']!r}")
    if metadata["feature_spec"] != "absolute-color-450":
        raise ValueFormatError(
            f"unsupported feature spec {metadata['feature_spec']!r}")
    if metadata["value_contract"] != "black-absolute-tanh-v1":
        raise ValueFormatError(
            f"unsupported value contract {metadata['value_contract']!r}")
    if not isinstance(metadata["label_versions"], list):
        raise ValueFormatError("label_versions must be a list")
    if not isinstance(metadata["training"], dict):
        raise ValueFormatError("training must be an object")
    metadata["ea"] = ea
    return metadata


def quantize_parameters(weights: dict, *, ea: int = EA_DEFAULT,
                        act_max: float = ACT_MAX_DEFAULT) -> dict:
    """Float parameters -> integer weights, biases and activation ceiling."""
    scale = 1 << ea
    out = {}
    for name, count, typecode in BODY_SPEC:
        values = weights[name]
        if len(values) != count:
            raise ValueFormatError(
                f"{name}: expected {count} parameters, got {len(values)}")
        limit = INT16_MAX if typecode == "h" else INT32_MAX
        lo = INT16_MIN if typecode == "h" else INT32_MIN
        out[name] = [max(lo, min(limit, int(round(float(v) * scale))))
                     for v in values]
    out["ea"] = ea
    out["act_max_int"] = int(round(act_max * scale))
    return out


def pack_value_weights(weights: dict, metadata: dict, *,
                       ea: int = EA_DEFAULT,
                       act_max: float = ACT_MAX_DEFAULT) -> bytes:
    """Packs float or integer parameters into the GMVALUE1 file format."""
    if any(name not in weights for name, _, _ in BODY_SPEC):
        raise ValueFormatError("weights are missing layers")
    if weights.get("ea") == ea and "act_max_int" in weights:
        quantized = weights
    else:
        quantized = quantize_parameters(weights, ea=ea, act_max=act_max)
    meta = _validate_metadata(dict(metadata), ea)
    meta["act_max_int"] = quantized["act_max_int"]
    meta_text = json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
    body = b"".join(_little_endian(array(
        "h" if typecode == "h" else "i",
        [int(v) for v in quantized[name]])) for name, _, typecode in BODY_SPEC)
    checksum = fnv1a(meta_text + body)
    header = HEADER_STRUCT.pack(
        MAGIC, VERSION, HEADER_SIZE, N_FEATURES, N_ACC, N_H1, N_H2, N_OUT,
        RULE_RENJU, FEATURE_SPEC, VALUE_CONTRACT, ea,
        quantized["act_max_int"], checksum, len(meta_text))
    return header + meta_text + body


def write_value_weights(path: str | Path, weights: dict, metadata: dict, *,
                        ea: int = EA_DEFAULT,
                        act_max: float = ACT_MAX_DEFAULT) -> dict:
    blob = pack_value_weights(weights, metadata, ea=ea, act_max=act_max)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    return {"path": str(out), "bytes": len(blob),
            "sha256": __import__("hashlib").sha256(blob).hexdigest()}


def unpack_value_weights(blob: bytes) -> tuple[dict, dict]:
    """Decodes and validates a GMVALUE1 file; rejects anything else."""
    if len(blob) < HEADER_SIZE:
        raise ValueFormatError("truncated value-model header")
    fields = HEADER_STRUCT.unpack_from(blob, 0)
    (magic, version, header_size, n_features, n_acc, n_h1, n_h2, n_out,
     rule, feature_spec, value_contract, ea, act_max_int, checksum,
     meta_len) = fields
    if magic != MAGIC:
        raise ValueFormatError(
            f"bad value-model magic {magic!r}, expected {MAGIC!r}")
    if version != VERSION:
        raise ValueFormatError(f"unsupported value-model version {version}")
    if header_size != HEADER_SIZE:
        raise ValueFormatError(f"bad header size {header_size}")
    if (n_features, n_acc, n_h1, n_h2, n_out) != (
            N_FEATURES, N_ACC, N_H1, N_H2, N_OUT):
        raise ValueFormatError("unsupported value-model dimensions")
    if rule != RULE_RENJU or feature_spec != FEATURE_SPEC \
            or value_contract != VALUE_CONTRACT:
        raise ValueFormatError(
            "value-model rule/features/contract mismatch")
    if not 1 <= ea <= 16:
        raise ValueFormatError(f"invalid fixed-point exponent {ea}")
    if not 1 <= act_max_int <= (1 << 30):
        raise ValueFormatError("invalid activation ceiling")
    meta_start = HEADER_SIZE
    meta_end = meta_start + meta_len
    body_end = meta_end + body_length()
    if meta_len <= 0 or body_end != len(blob):
        raise ValueFormatError(
            f"file length {len(blob)} does not match header (expected {body_end})")
    meta_text = blob[meta_start:meta_end]
    body = blob[meta_end:]
    if fnv1a(meta_text + body) != checksum:
        raise ValueFormatError("value-model checksum mismatch")
    try:
        metadata = json.loads(meta_text.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueFormatError(f"metadata is not valid JSON: {exc}") from exc
    metadata = _validate_metadata(metadata, ea)
    if metadata.get("act_max_int") != act_max_int:
        raise ValueFormatError("metadata/header activation ceiling mismatch")
    weights: dict = {}
    cursor = 0
    for name, count, typecode in BODY_SPEC:
        size = count * array(typecode).itemsize
        weights[name] = _from_little_endian(body[cursor:cursor + size], typecode)
        cursor += size
    weights["ea"] = ea
    weights["act_max_int"] = act_max_int
    return weights, metadata


def read_value_weights(path: str | Path) -> tuple[dict, dict]:
    return unpack_value_weights(Path(path).read_bytes())


def infer_int(weights: dict, features: Sequence[int]) -> int:
    """Integer forward; returns z in fixed point, where real z = z / 2^ea."""
    if _np is not None:
        return _infer_numpy(weights, features)
    ea = int(weights["ea"])
    scale = 1 << ea
    act_max = int(weights["act_max_int"])
    w0, w1, w2, w3 = (weights["W0"], weights["W1"],
                      weights["W2"], weights["W3"])
    acc = list(weights["b0"])
    for index in features:
        if not 0 <= index < N_FEATURES:
            raise ValueError(f"feature index out of range: {index}")
        base = index * N_ACC
        row = w0[base:base + N_ACC]
        for j in range(N_ACC):
            acc[j] += row[j]
    a0 = [0 if v < 0 else (act_max if v > act_max else v) for v in acc]

    z1 = [0] * N_H1
    for j, value in enumerate(a0):
        if value == 0:
            continue
        base = j * N_H1
        row = w1[base:base + N_H1]
        for k in range(N_H1):
            z1[k] += value * row[k]
    a1 = []
    for k in range(N_H1):
        v = round_div(z1[k], scale) + int(weights["b1"][k])
        a1.append(0 if v < 0 else (act_max if v > act_max else v))

    z2 = [0] * N_H2
    for k, value in enumerate(a1):
        if value == 0:
            continue
        base = k * N_H2
        row = w2[base:base + N_H2]
        for m in range(N_H2):
            z2[m] += value * row[m]
    a2 = []
    for m in range(N_H2):
        v = round_div(z2[m], scale) + int(weights["b2"][m])
        a2.append(0 if v < 0 else (act_max if v > act_max else v))

    z = sum(a2[m] * int(w3[m * N_OUT]) for m in range(N_H2))
    return round_div(z, scale) + int(weights["b3"][0])


def _infer_numpy(weights: dict, features: Sequence[int]) -> int:
    ea = int(weights["ea"])
    scale = 1 << ea
    act_max = int(weights["act_max_int"])
    w0 = _np.asarray(weights["W0"], dtype=_np.int64).reshape(N_FEATURES, N_ACC)
    w1 = _np.asarray(weights["W1"], dtype=_np.int64).reshape(N_ACC, N_H1)
    w2 = _np.asarray(weights["W2"], dtype=_np.int64).reshape(N_H1, N_H2)
    w3 = _np.asarray(weights["W3"], dtype=_np.int64).reshape(N_H2, N_OUT)
    indices = [int(i) for i in features]
    acc = _np.asarray(weights["b0"], dtype=_np.int64).copy()
    if indices:
        acc += w0[indices].sum(axis=0)
    a0 = _np.clip(acc, 0, act_max)

    def layer(a, matrix, bias):
        dot = a @ matrix
        z = _np.sign(dot) * ((2 * _np.abs(dot) + scale) // (2 * scale))
        z = z + _np.asarray(bias, dtype=_np.int64)
        return _np.clip(z, 0, act_max)

    a1 = layer(a0, w1, weights["b1"])
    a2 = layer(a1, w2, weights["b2"])
    z = int((a2 @ w3)[0])
    return round_div(z, scale) + int(weights["b3"][0])


def value_from_int_z(z_int: int, weights: dict) -> float:
    """Shared final mapping: value = tanh(z / 2^ea)."""
    return math.tanh(z_int / float(1 << int(weights["ea"])))


def summarize(weights: dict) -> dict:
    return {name: {"min": min(values), "max": max(values)}
            for name, values in weights.items()
            if name in _WEIGHT_KEYS}
