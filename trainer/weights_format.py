"""Quantised NNUE weight file format + dependency-free integer reference forward.

This is the contract the C++ engine loads in ticket 09. Everything here is
standard library so the format can be checked and the integer forward can be
replayed without torch/numpy installed.

Architecture (absolute-colour features, no perspective flip)::

    sparse features (450) -> accumulator 1024 (int16) -> ClippedReLU(0, QA)
      -> L1 15 -> ClippedReLU(0, QB) -> output head (2, selected by side to move)

Integer forward (ALL arithmetic is exact integer arithmetic)::

    acc[j] = clamp_i16(b0[j] + sum(W0[i, j] for i in sparse_indices))
    a[j]   = clamp(acc[j], 0, QA)
    z[k]   = sum_j a[j] * W1[j, k]
    zq[k]  = round_div(z[k], QA) + b1[k]
    a2[k]  = clamp(zq[k], 0, QB)
    y      = sum_k a2[k] * W2[h, k]          h = stm (0 black, 1 white)
    eval   = round_div(y, QB) + b2[h]        "engine score" units

``round_div(n, d)`` is round-half-away-from-zero:
``sign(n) * ((2*|n| + d) // (2*d))``. C++ must not use plain integer division
(it truncates toward zero for negatives). ``clamp_i16`` saturates; C++ must
accumulate into int32 and saturate before storing into an int16 accumulator.

File layout (little-endian): 64-byte header, then the weight body.

    offset  size  field
    0       8     magic b"GMNNUE01"
    8       4     version = 1
    12      4     header_size = 64
    16      4     n_features = 450
    20      4     n_acc = 1024
    24      4     n_l1 = 15
    28      4     n_heads = 2
    32      4     QA (accumulator ClippedReLU upper bound)
    36      4     QB (L1 ClippedReLU upper bound)
    40      4     score_scale (tanh denominator)
    44      4     weight_clamp (abs limit of int8 weights)
    48      4     FNV-1a 32-bit hash of the weight body
    52      12    reserved zeros

    body (offset 64), all row-major:
      W0 int8  [450, 1024]   b0 int16 [1024]
      W1 int8  [1024, 15]    b1 int16 [15]
      W2 int8  [2, 15]       b2 int16 [2]
"""
from __future__ import annotations

import os
import struct
import sys
from array import array
from collections.abc import Iterable, Sequence
from pathlib import Path

MAGIC = b"GMNNUE01"
VERSION = 1
MAGIC_V2 = b"GMNNUE02"
STATIC_LIMIT = 900000
OUTPUT_GAINS = tuple(1 << i for i in range(11))
HEADER_SIZE = 64

N_FEATURES = 450
N_ACC = 1024
N_L1 = 15
N_HEADS = 2

QA = 127
QB = 64
SCORE_SCALE = 400
WEIGHT_CLAMP = 64
INT16_MIN, INT16_MAX = -32768, 32767

W0_LEN = N_FEATURES * N_ACC
B0_LEN = N_ACC
W1_LEN = N_ACC * N_L1
B1_LEN = N_L1
W2_LEN = N_HEADS * N_L1
B2_LEN = N_HEADS
BODY_LEN = W0_LEN + 2 * B0_LEN + W1_LEN + 2 * B1_LEN + W2_LEN + 2 * B2_LEN
FILE_LEN = HEADER_SIZE + BODY_LEN

HEADER_STRUCT = struct.Struct("<8s11I")
assert HEADER_STRUCT.size == 52

try:  # optional fast path for the integer reference forward
    import numpy as _np  # type: ignore
except ImportError:  # pragma: no cover - exercised on bare installs
    _np = None  # type: ignore


class WeightFormatError(Exception):
    """Malformed or unsupported NNUE weight file."""


def fnv1a(data: bytes) -> int:
    """FNV-1a 32-bit; C++ reference: h = 2166136261; for b: h = (h ^ b) * 16777619u."""
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


def clamp_i16(value: int) -> int:
    return max(INT16_MIN, min(value, INT16_MAX))


def _as_array(values: Iterable[int], typecode: str, name: str, expected: int,
              lo: int, hi: int) -> array:
    numbers = [int(v) for v in values]
    if len(numbers) != expected:
        raise WeightFormatError(f"{name}: expected {expected} values, got {len(numbers)}")
    for value in numbers:
        if not lo <= value <= hi:
            raise WeightFormatError(f"{name}: value {value} outside [{lo}, {hi}]")
    return array(typecode, numbers)


def normalise_weights(weights: dict) -> dict:
    """Validate and convert a weight dict into array('b') / array('h')."""
    return {
        "W0": _as_array(weights["W0"], "b", "W0", W0_LEN, -WEIGHT_CLAMP, WEIGHT_CLAMP),
        "b0": _as_array(weights["b0"], "h", "b0", B0_LEN, INT16_MIN, INT16_MAX),
        "W1": _as_array(weights["W1"], "b", "W1", W1_LEN, -WEIGHT_CLAMP, WEIGHT_CLAMP),
        "b1": _as_array(weights["b1"], "h", "b1", B1_LEN, INT16_MIN, INT16_MAX),
        "W2": _as_array(weights["W2"], "b", "W2", W2_LEN, -WEIGHT_CLAMP, WEIGHT_CLAMP),
        "b2": _as_array(weights["b2"], "h", "b2", B2_LEN, INT16_MIN, INT16_MAX),
    }


def _to_little_endian(arr: array) -> bytes:
    if sys.byteorder == "big":
        arr = array(arr.typecode, arr)
        arr.byteswap()
    return arr.tobytes()


def _from_little_endian(raw: bytes, typecode: str) -> array:
    result = array(typecode)
    result.frombytes(raw)
    if sys.byteorder == "big":
        result.byteswap()
    return result


def body_bytes(weights: dict) -> bytes:
    w = normalise_weights(weights)
    return b"".join(_to_little_endian(w[key])
                    for key in ("W0", "b0", "W1", "b1", "W2", "b2"))


def pack_weights(weights: dict, *, qa: int = QA, qb: int = QB,
                 score_scale: int = SCORE_SCALE,
                 weight_clamp: int = WEIGHT_CLAMP,
                 version: int | None = None, output_gain: int | None = None) -> bytes:
    version = weights.get("format_version", 1) if version is None else version
    output_gain = weights.get("output_gain", 1) if output_gain is None else output_gain
    if version not in (1, 2) or output_gain not in OUTPUT_GAINS or (version == 1 and output_gain != 1):
        raise WeightFormatError("invalid format version/output gain")
    body = body_bytes(weights)
    if len(body) != BODY_LEN:
        raise WeightFormatError(f"body length {len(body)} != {BODY_LEN}")
    header = HEADER_STRUCT.pack(
        MAGIC if version == 1 else MAGIC_V2, version, HEADER_SIZE, N_FEATURES, N_ACC, N_L1, N_HEADS,
        qa, qb, score_scale, weight_clamp, fnv1a(body),
    )
    reserved = b"\0" * 12 if version == 1 else struct.pack("<I", output_gain) + b"\0" * 8
    return header + reserved + body


def write_weights(path: str | os.PathLike[str], weights: dict, **kwargs) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pack_weights(weights, **kwargs))


def unpack_weights(blob: bytes) -> tuple[dict, dict]:
    """Return (weights, metadata); raises on any format/version/hash mismatch."""
    if len(blob) < HEADER_SIZE:
        raise WeightFormatError("truncated NNUE weight header")
    fields = HEADER_STRUCT.unpack_from(blob, 0)
    magic = fields[0]
    (version, header_size, n_features, n_acc, n_l1, n_heads,
     qa, qb, score_scale, weight_clamp, checksum) = fields[1:]
    if magic not in (MAGIC, MAGIC_V2):
        raise WeightFormatError(f"bad NNUE weight magic {magic!r}, expected {MAGIC!r}")
    if version not in (1, 2) or magic != (MAGIC if version == 1 else MAGIC_V2):
        raise WeightFormatError(f"unsupported weight version {version}")
    if header_size != HEADER_SIZE:
        raise WeightFormatError(f"bad header size {header_size}")
    if (n_features, n_acc, n_l1, n_heads) != (N_FEATURES, N_ACC, N_L1, N_HEADS):
        raise WeightFormatError("unsupported network dimensions")
    if not 0 < qa <= 0x7fffffff or not 0 < qb <= 0x7fffffff or score_scale <= 0 or not 0 < weight_clamp <= 127:
        raise WeightFormatError("invalid quantisation constants in header")
    gain = 1 if version == 1 else struct.unpack_from("<I", blob, 52)[0]
    if gain not in OUTPUT_GAINS:
        raise WeightFormatError("invalid output gain")
    reserved_start = 52 if version == 1 else 56
    if any(blob[reserved_start:HEADER_SIZE]):
        raise WeightFormatError("non-zero reserved header bytes")

    body = blob[HEADER_SIZE:]
    if len(body) != BODY_LEN:
        raise WeightFormatError(f"body length {len(body)} != {BODY_LEN}")
    if fnv1a(body) != checksum:
        raise WeightFormatError("NNUE weight checksum mismatch")

    cursor = 0
    weights = {}
    for key, length, typecode in (
        ("W0", W0_LEN, "b"), ("b0", B0_LEN * 2, "h"),
        ("W1", W1_LEN, "b"), ("b1", B1_LEN * 2, "h"),
        ("W2", W2_LEN, "b"), ("b2", B2_LEN * 2, "h"),
    ):
        weights[key] = _from_little_endian(body[cursor:cursor + length], typecode)
        cursor += length
    normalise_weights(weights)  # range check after decoding
    if any(abs(v) > weight_clamp for k in ("W0", "W1", "W2") for v in weights[k]):
        raise WeightFormatError("weight exceeds header weight_clamp")
    weights.update(format_version=version, output_gain=gain)
    meta = {
        "version": version, "output_gain": gain, "qa": qa, "qb": qb,
        "score_scale": score_scale, "weight_clamp": weight_clamp,
        "checksum": checksum, "file_size": len(blob),
    }
    return weights, meta


def read_weights(path: str | os.PathLike[str]) -> tuple[dict, dict]:
    return unpack_weights(Path(path).read_bytes())


def infer_int(weights: dict, sparse_indices: Sequence[int], stm: int,
              *, qa: int = QA, qb: int = QB) -> int:
    """Dependency-free integer forward. Optional numpy fast path if available."""
    if stm not in (0, 1):
        raise ValueError(f"stm must be 0 (black) or 1 (white), got {stm}")
    indices = list(sparse_indices)
    for index in indices:
        if not 0 <= index < N_FEATURES:
            raise ValueError(f"feature index out of range: {index}")
    if _np is not None:
        return _infer_numpy(weights, indices, stm, qa, qb)
    return _infer_python(weights, indices, stm, qa, qb)


def _infer_numpy(weights: dict, indices: Sequence[int], stm: int,
                 qa: int, qb: int) -> int:
    w0 = _np.asarray(weights["W0"], dtype=_np.int64).reshape(N_FEATURES, N_ACC)
    b0 = _np.asarray(weights["b0"], dtype=_np.int64)
    w1 = _np.asarray(weights["W1"], dtype=_np.int64).reshape(N_ACC, N_L1)
    b1 = _np.asarray(weights["b1"], dtype=_np.int64)
    w2 = _np.asarray(weights["W2"], dtype=_np.int64).reshape(N_HEADS, N_L1)
    b2 = _np.asarray(weights["b2"], dtype=_np.int64)

    acc = b0.copy()
    if indices:
        acc += w0[list(indices)].sum(axis=0)
    acc = _np.clip(acc, INT16_MIN, INT16_MAX)
    a = _np.clip(acc, 0, qa)
    z = a @ w1
    sign = _np.where(z >= 0, 1, -1)
    zq = sign * ((2 * _np.abs(z) + qa) // (2 * qa)) + b1
    a2 = _np.clip(zq, 0, qb)
    y = int(a2 @ w2[stm])
    return output_score(y, int(b2[stm]), qb, weights)


def _infer_python(weights: dict, indices: Sequence[int], stm: int,
                  qa: int, qb: int) -> int:
    w0, b0 = weights["W0"], weights["b0"]
    w1, b1 = weights["W1"], weights["b1"]
    w2, b2 = weights["W2"], weights["b2"]

    acc = list(b0)
    for index in indices:
        base = index * N_ACC
        for j in range(N_ACC):
            acc[j] += w0[base + j]
    a = [max(0, min(v, qa)) for v in (clamp_i16(v) for v in acc)]

    z = [0] * N_L1
    for j, av in enumerate(a):
        if av == 0:
            continue
        base = j * N_L1
        for k in range(N_L1):
            z[k] += av * w1[base + k]
    zq = [round_div(z[k], qa) + b1[k] for k in range(N_L1)]
    a2 = [max(0, min(v, qb)) for v in zq]

    base = stm * N_L1
    y = sum(a2[k] * w2[base + k] for k in range(N_L1))
    return output_score(y, int(b2[stm]), qb, weights)


def output_score(y: int, bias: int, qb: int, weights: dict) -> int:
    if weights.get("format_version", 1) == 1:
        return round_div(y, qb) + bias
    gain = weights["output_gain"]
    value = round_div(gain * y, qb) + gain * bias
    return max(-STATIC_LIMIT, min(STATIC_LIMIT, value))


def weights_summary(weights: dict) -> dict:
    return {
        key: {"length": len(weights[key]), "min": min(weights[key]),
              "max": max(weights[key])}
        for key in ("W0", "b0", "W1", "b1", "W2", "b2")
    }
