"""Pure-Python TFRecord and tf.train.Example parsing."""

from __future__ import annotations

import gzip
import io
import struct
from pathlib import Path
from typing import Any, AbstractSet, Iterator

# gzip magic bytes (RFC 1952)
_GZIP_MAGIC = b"\x1f\x8b"


def _is_gzip(path: Path) -> bool:
    """Return True if *path* begins with the gzip magic header."""
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == _GZIP_MAGIC
    except OSError:
        return False


def _iter_raw_tfrecord_records(reader: io.RawIOBase | io.BufferedIOBase, path: Path) -> Iterator[bytes]:
    """Yield serialized records from an already-open binary stream (raw TFRecord)."""
    while True:
        header = reader.read(12)
        if not header:
            return
        if len(header) < 12:
            raise EOFError(f"Truncated TFRecord header in {path}")
        length, _masked_length_crc = struct.unpack("<QI", header)
        payload = reader.read(length)
        _masked_data_crc = reader.read(4)
        if len(payload) != length:
            raise EOFError(f"Unexpected EOF while reading TFRecord payload from {path}")
        yield payload


def iter_gzip_tfrecord_records(path: Path) -> Iterator[bytes]:
    """Yield serialized records from a gzip-compressed TFRecord shard."""

    with gzip.open(path, "rb") as reader:
        yield from _iter_raw_tfrecord_records(reader, path)


def iter_tfrecord_records(path: Path) -> Iterator[bytes]:
    """Yield serialized records from a TFRecord shard.

    Auto-detects whether the shard is gzip-compressed (checks for the
    0x1f 0x8b magic header) and falls back to uncompressed reading when the
    magic bytes are absent.  This lets the same code transparently handle:

    * The original GCS download (``*.tfrecord.gz`` — gzip-compressed).
    * HuggingFace mirror shards (no file extension — may or may not be gzip).
    """
    if _is_gzip(path):
        with gzip.open(path, "rb") as reader:
            yield from _iter_raw_tfrecord_records(reader, path)
    else:
        with open(path, "rb") as reader:
            yield from _iter_raw_tfrecord_records(reader, path)


def _read_varint(buffer: bytes, position: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = buffer[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, position
        shift += 7


def _skip_field(wire_type: int, buffer: bytes, position: int) -> int:
    if wire_type == 0:
        _value, position = _read_varint(buffer, position)
        return position
    if wire_type == 1:
        return position + 8
    if wire_type == 2:
        length, position = _read_varint(buffer, position)
        return position + length
    if wire_type == 5:
        return position + 4
    raise ValueError(f"Unsupported protobuf wire type: {wire_type}")


def _parse_bytes_list_message(buffer: bytes) -> list[bytes]:
    position = 0
    values: list[bytes] = []
    while position < len(buffer):
        key, position = _read_varint(buffer, position)
        field_number, wire_type = key >> 3, key & 0x07
        if field_number == 1 and wire_type == 2:
            length, position = _read_varint(buffer, position)
            values.append(buffer[position : position + length])
            position += length
        else:
            position = _skip_field(wire_type, buffer, position)
    return values


def _parse_int64_list_message(buffer: bytes) -> list[int]:
    position = 0
    values: list[int] = []
    while position < len(buffer):
        key, position = _read_varint(buffer, position)
        field_number, wire_type = key >> 3, key & 0x07
        if field_number == 1 and wire_type == 0:
            value, position = _read_varint(buffer, position)
            values.append(value)
        elif field_number == 1 and wire_type == 2:
            length, position = _read_varint(buffer, position)
            end = position + length
            while position < end:
                value, position = _read_varint(buffer, position)
                values.append(value)
        else:
            position = _skip_field(wire_type, buffer, position)
    return values


def _parse_float_list_message(buffer: bytes) -> list[float]:
    position = 0
    values: list[float] = []
    while position < len(buffer):
        key, position = _read_varint(buffer, position)
        field_number, wire_type = key >> 3, key & 0x07
        if field_number == 1 and wire_type == 5:
            values.append(struct.unpack("<f", buffer[position : position + 4])[0])
            position += 4
        elif field_number == 1 and wire_type == 2:
            length, position = _read_varint(buffer, position)
            payload = buffer[position : position + length]
            position += length
            if length % 4 != 0:
                raise ValueError("Packed float list length must be divisible by 4.")
            values.extend(struct.unpack("<" + ("f" * (length // 4)), payload))
        else:
            position = _skip_field(wire_type, buffer, position)
    return values


def _parse_feature_message(buffer: bytes) -> tuple[str, list[Any]] | None:
    position = 0
    value: tuple[str, list[Any]] | None = None
    while position < len(buffer):
        key, position = _read_varint(buffer, position)
        field_number, wire_type = key >> 3, key & 0x07
        if wire_type != 2:
            position = _skip_field(wire_type, buffer, position)
            continue
        length, position = _read_varint(buffer, position)
        payload = buffer[position : position + length]
        position += length
        if field_number == 1:
            value = ("bytes_list", _parse_bytes_list_message(payload))
        elif field_number == 2:
            value = ("float_list", _parse_float_list_message(payload))
        elif field_number == 3:
            value = ("int64_list", _parse_int64_list_message(payload))
    return value


def _parse_features_message(
    buffer: bytes,
    *,
    feature_names: AbstractSet[str] | None = None,
) -> dict[str, tuple[str, list[Any]]]:
    position = 0
    feature_map: dict[str, tuple[str, list[Any]]] = {}
    remaining = set(feature_names) if feature_names is not None else None
    while position < len(buffer):
        key, position = _read_varint(buffer, position)
        field_number, wire_type = key >> 3, key & 0x07
        if field_number != 1 or wire_type != 2:
            position = _skip_field(wire_type, buffer, position)
            continue

        entry_length, position = _read_varint(buffer, position)
        entry_end = position + entry_length
        entry_position = position
        name: str | None = None
        value: tuple[str, list[Any]] | None = None
        deferred_value_bounds: tuple[int, int] | None = None
        include_value = feature_names is None

        while entry_position < entry_end:
            entry_key, entry_position = _read_varint(buffer, entry_position)
            entry_field_number, entry_wire_type = entry_key >> 3, entry_key & 0x07
            if entry_field_number == 1 and entry_wire_type == 2:
                item_length, entry_position = _read_varint(buffer, entry_position)
                name = buffer[entry_position : entry_position + item_length].decode("utf-8")
                entry_position += item_length
                include_value = feature_names is None or name in feature_names
                if include_value and deferred_value_bounds is not None:
                    value_start, value_end = deferred_value_bounds
                    value = _parse_feature_message(buffer[value_start:value_end])
            elif entry_field_number == 2 and entry_wire_type == 2:
                item_length, entry_position = _read_varint(buffer, entry_position)
                value_start = entry_position
                value_end = entry_position + item_length
                if include_value:
                    value = _parse_feature_message(buffer[value_start:value_end])
                elif name is None:
                    deferred_value_bounds = (value_start, value_end)
                entry_position = value_end
            else:
                entry_position = _skip_field(entry_wire_type, buffer, entry_position)

        if name is not None and value is not None:
            feature_map[name] = value
            if remaining is not None:
                remaining.discard(name)
                if not remaining:
                    return feature_map
        position = entry_end

    return feature_map


def parse_tf_example(
    serialized_example: bytes,
    *,
    feature_names: AbstractSet[str] | None = None,
) -> dict[str, tuple[str, list[Any]]]:
    """Parse a serialized tf.train.Example into a Python feature map."""

    position = 0
    while position < len(serialized_example):
        key, position = _read_varint(serialized_example, position)
        field_number, wire_type = key >> 3, key & 0x07
        if field_number == 1 and wire_type == 2:
            length, position = _read_varint(serialized_example, position)
            payload = serialized_example[position : position + length]
            return _parse_features_message(payload, feature_names=feature_names)
        position = _skip_field(wire_type, serialized_example, position)
    return {}


def decode_tf_example_features(
    feature_map: dict[str, tuple[str, list[Any]]],
) -> dict[str, list[Any]]:
    """Discard tf.train.Feature wrappers and return plain Python lists."""

    return {name: values for name, (_kind, values) in feature_map.items()}
