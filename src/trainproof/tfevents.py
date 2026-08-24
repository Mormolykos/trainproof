"""TensorBoard event-file reader - pure Python, zero dependencies.

PyTorch Lightning, Fish Speech, Coqui and most custom loops write their scalars
to `events.out.tfevents.*` and nowhere else. Without this module those runs are
invisible to trainproof no matter how broken they are.

Nothing here imports tensorflow, tensorboard, or protobuf. A linter that made
you install a 600 MB ML framework to read a log file would be a worse tool. The
two formats involved are small and stable, so they are decoded by hand:

  TFRecord framing   uint64 length | uint32 crc | payload | uint32 crc
  Event protobuf     wall_time (1, double), step (2, varint), summary (5, msg)
  Summary.Value      tag (1, string), simple_value (2, float), tensor (8, msg)

CRC footers are skipped, not verified: crc32c is not in the stdlib and adding a
dependency to checksum a file we are only reading would be a bad trade. A
truncated final record - the normal state of a killed run - is dropped rather
than raising, because a killed run is exactly the run a user most wants judged.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path

# protobuf wire types
_VARINT, _FIXED64, _LEN, _FIXED32 = 0, 1, 2, 5


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise IndexError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf: bytes, start: int = 0, end: int | None = None) -> Iterator[tuple[int, int, int | bytes]]:
    """Yield (field_number, wire_type, value) over one protobuf message.

    The value type depends on the wire type and always has: a varint yields an
    int, every other wire type yields bytes. Callers must branch on `wire`
    before using `val`, which is why that union is written down rather than
    inferred from whichever branch happens to come first.
    """
    end = len(buf) if end is None else end
    pos = start
    val: int | bytes
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 0x07
        if wire == _VARINT:
            val, pos = _read_varint(buf, pos)
        elif wire == _FIXED64:
            val = buf[pos:pos + 8]
            pos += 8
        elif wire == _LEN:
            ln, pos = _read_varint(buf, pos)
            val = buf[pos:pos + ln]
            pos += ln
        elif wire == _FIXED32:
            val = buf[pos:pos + 4]
            pos += 4
        else:  # groups - removed from proto3, never emitted by TensorBoard
            raise ValueError(f"unsupported wire type {wire}")
        yield field, wire, val


def _tensor_scalar(buf: bytes) -> float | None:
    """Extract a single float from a TensorProto.

    Lightning logs scalars as rank-0 tensors rather than simple_value, so a
    reader that only handles simple_value sees an empty run.
    """
    # The `isinstance` guards restate, at the point of use, the invariant
    # `_iter_fields` documents: a varint yields an int and every other wire
    # type yields bytes. They are also what lets a type checker see it. A field
    # that contradicts its own wire type is skipped rather than raised on,
    # which is this module's standing policy for a malformed log - see the
    # module docstring on truncated records.
    for field, wire, val in _iter_fields(buf):
        if field == 5 and wire == _LEN and isinstance(val, bytes) and len(val) >= 4:   # packed float_val
            return struct.unpack("<f", val[:4])[0]
        if field == 5 and wire == _FIXED32 and isinstance(val, bytes):                 # single float_val
            return struct.unpack("<f", val)[0]
        if field == 4 and wire == _LEN and isinstance(val, bytes) and len(val) >= 4:   # tensor_content
            return struct.unpack("<f", val[:4])[0]
        if field == 6 and wire == _LEN and isinstance(val, bytes) and len(val) >= 8:   # double_val
            return struct.unpack("<d", val[:8])[0]
    return None


def _parse_event(buf: bytes) -> tuple[float, int, list[tuple[str, float]]]:
    wall_time, step, scalars = 0.0, 0, []
    for field, wire, val in _iter_fields(buf):
        if field == 1 and wire == _FIXED64 and isinstance(val, bytes):
            wall_time = struct.unpack("<d", val)[0]
        elif field == 2 and wire == _VARINT and isinstance(val, int):
            step = val
        elif field == 5 and wire == _LEN and isinstance(val, bytes):   # Summary
            for sf, sw, sv in _iter_fields(val):
                if sf != 1 or sw != _LEN or not isinstance(sv, bytes):
                    continue
                tag, value = None, None
                for vf, vw, vv in _iter_fields(sv):             # Summary.Value
                    if vf == 1 and vw == _LEN and isinstance(vv, bytes):
                        tag = vv.decode("utf-8", "replace")
                    elif vf == 2 and vw == _FIXED32 and isinstance(vv, bytes):
                        value = struct.unpack("<f", vv)[0]
                    elif vf == 8 and vw == _LEN and isinstance(vv, bytes):
                        value = _tensor_scalar(vv)
                if tag is not None and value is not None:
                    scalars.append((tag, value))
    return wall_time, step, scalars


def read_scalars(path: str | Path) -> dict[str, list[tuple[int, float, float]]]:
    """Read one event file into {tag: [(step, value, wall_time), ...]}."""
    data = Path(path).read_bytes()
    out: dict[str, list[tuple[int, float, float]]] = {}
    pos, n = 0, len(data)
    while pos + 12 <= n:
        length = struct.unpack("<Q", data[pos:pos + 8])[0]
        body = pos + 12
        tail = body + length + 4
        if tail > n:
            break                                    # truncated: killed run
        try:
            wall_time, step, scalars = _parse_event(data[body:body + length])
        except (IndexError, ValueError, struct.error):
            pos = tail                               # skip the bad record only
            continue
        for tag, value in scalars:
            out.setdefault(tag, []).append((step, value, wall_time))
        pos = tail
    return out


# --------------------------------------------------------------------------
# tag -> canonical column
# --------------------------------------------------------------------------
def _section(prefix: str) -> str | None:
    """Classify a tag's section as eval or train.

    Frameworks do not agree on section names: Lightning writes 'train'/'val',
    Coqui writes 'TrainIterStats'/'EvalStats'. Matching on substrings covers
    both. Eval is tested first because 'eval' contains 'val'.
    """
    p = prefix.lower()
    if "eval" in p or "valid" in p or p.startswith("val") or "test" in p:
        return "val_"
    if "train" in p or p in ("tr", "fit"):
        return ""
    return None


def normalize_tag(tag: str) -> str:
    """Map a TensorBoard tag onto a name the canonical alias table knows.

    'train/loss' -> 'loss'        'val/loss'              -> 'val_loss'
    'lr-AdamW/pg1' -> 'lr'        'TrainIterStats/loss'   -> 'loss'
                                  'EvalStats/avg_loss'    -> 'val_loss'
    """
    t = tag.strip().lower()
    if t.startswith("lr-") or t.startswith("lr/") or t in ("lr", "learning_rate"):
        return "lr"
    if "/" in t:
        prefix, rest = t.split("/", 1)
        rest = rest.replace("/", "_")
        # 'avg_loss' is the same quantity as 'loss', just aggregated.
        if rest.startswith("avg_"):
            rest = rest[4:]
        section = _section(prefix)
        if section is not None:
            return section + rest
        return t.replace("/", "_")
    return t


def scalars_to_records(scalars: dict[str, list[tuple[int, float, float]]],
                       mapping_overrides: dict[str, str] | None = None
                       ) -> tuple[list[dict[str, float]], dict[str, str]]:
    """Fold {tag: [(step, value, wall)]} into trainproof's per-step records."""
    from .adapters import _resolve_key

    by_step: dict[int, dict[str, float]] = {}
    used_mapping: dict[str, str] = {}

    # Several tags can claim one column: Coqui logs both TrainIterStats/loss
    # (per step) and TrainEpochStats/avg_loss (per epoch). Resolve by density -
    # the denser series is the one with something to say - and break ties
    # alphabetically so the choice is deterministic across runs.
    claims: dict[str, list[str]] = {}
    for tag in scalars:
        canon = _resolve_key(normalize_tag(tag), mapping_overrides)
        if canon is None or canon == "step":
            continue
        claims.setdefault(canon, []).append(tag)

    for canon, tags in claims.items():
        winner = sorted(tags, key=lambda t: (-len(scalars[t]), t))[0]
        used_mapping[canon] = winner
        for step, value, wall in scalars[winner]:
            row = by_step.setdefault(step, {"step": float(step)})
            row[canon] = float(value)
            row.setdefault("time", float(wall))

    records = [by_step[s] for s in sorted(by_step)]
    return records, used_mapping


def parse_tfevents(path: str | Path, mapping_overrides: dict[str, str] | None = None
                   ) -> tuple[list[dict[str, float]], dict[str, str]]:
    """Full path: event file (or a directory containing them) -> records."""
    p = Path(path)
    files = (sorted(f for f in p.glob("**/*tfevents*") if _looks_like_event_file(f))
             if p.is_dir() else [p])
    merged: dict[str, list[tuple[int, float, float]]] = {}
    for f in files:
        for tag, pts in read_scalars(f).items():
            merged.setdefault(tag, []).extend(pts)
    for tag in merged:
        merged[tag].sort(key=lambda x: x[0])
    return scalars_to_records(merged, mapping_overrides)


# A source file discussing tfevents is not a tfevents file. Substring matching
# alone would hand this module's own test file to the binary parser.
_NOT_EVENT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".csv",
                       ".yaml", ".yml", ".log", ".rst", ".toml"}


def _looks_like_event_file(p: Path) -> bool:
    return "tfevents" in p.name and p.suffix.lower() not in _NOT_EVENT_SUFFIXES


def is_tfevents(path: str | Path) -> bool:
    p = Path(path)
    if p.is_dir():
        return any(_looks_like_event_file(f) for f in p.glob("**/*tfevents*"))
    return _looks_like_event_file(p)
