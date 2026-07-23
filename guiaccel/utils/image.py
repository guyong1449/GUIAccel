"""Lightweight PNG helpers."""

from __future__ import annotations

import struct
from pathlib import Path

from guiaccel.types import ScreenshotAsset

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _parse_png_dimensions(header: bytes) -> tuple[int, int]:
    if len(header) < 24 or not header.startswith(PNG_SIGNATURE):
        raise ValueError("Expected a PNG image header.")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    return _parse_png_dimensions(header)


def read_png_size_from_bytes(png_bytes: bytes) -> tuple[int, int]:
    return _parse_png_dimensions(png_bytes[:24])


def screenshot_from_path(path: Path) -> ScreenshotAsset:
    width, height = read_png_size(path)
    return ScreenshotAsset(path=path, width=width, height=height, source="path")


def screenshot_from_bytes(
    png_bytes: bytes,
    *,
    path: Path | None = None,
    width: int | None = None,
    height: int | None = None,
) -> ScreenshotAsset:
    resolved_width, resolved_height = (width, height)
    if resolved_width is None or resolved_height is None:
        resolved_width, resolved_height = read_png_size_from_bytes(png_bytes)
    return ScreenshotAsset(
        path=path,
        png_bytes=png_bytes,
        width=resolved_width,
        height=resolved_height,
        source="bytes" if path is None else "cache",
    )
