"""2x2 frame compositing and perceptual hashing."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

log = logging.getLogger(__name__)

_LABELS = ("1", "2", "3", "4")


def compose_grid(
    frames: Sequence[Path | Image.Image],
    dest: Path,
    *,
    width: int = 1280,
    height: int = 720,
    quality: int = 78,
    annotate: bool = True,
) -> Path:
    """Stitch up to four frames into a single 2x2 JPEG.

    One grid image costs roughly a quarter of the vision tokens of four
    separate images while still giving the model a 12 second motion window.
    Missing frames render as black cells so cell position always maps to the
    same relative timestamp.
    """
    if not frames:
        raise ValueError("compose_grid requires at least one frame")

    cell_w, cell_h = width // 2, height // 2
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    positions = ((0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h))

    for index, position in enumerate(positions):
        if index >= len(frames):
            break
        source = frames[index]
        try:
            img = Image.open(source) if isinstance(source, Path) else source
            with img:
                cell = _fit(img.convert("RGB"), cell_w, cell_h)
        except Exception as exc:
            log.warning("skipping unreadable frame %s: %s", source, exc)
            continue
        canvas.paste(cell, position)

    if annotate:
        _draw_labels(canvas, cell_w, cell_h)

    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, format="JPEG", quality=quality, optimize=True)
    return dest


def _fit(img: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    """Letterbox an image into a cell without distorting aspect ratio."""
    scaled = img.copy()
    scaled.thumbnail((cell_w, cell_h), Image.Resampling.BILINEAR)
    cell = Image.new("RGB", (cell_w, cell_h), (0, 0, 0))
    cell.paste(scaled, ((cell_w - scaled.width) // 2, (cell_h - scaled.height) // 2))
    return cell


def _draw_labels(canvas: Image.Image, cell_w: int, cell_h: int) -> None:
    """Number the cells so the model can reason about frame ordering."""
    draw = ImageDraw.Draw(canvas)
    origins = ((0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h))
    for label, (x, y) in zip(_LABELS, origins):
        draw.rectangle([x + 4, y + 4, x + 26, y + 26], fill=(0, 0, 0))
        draw.text((x + 11, y + 9), label, fill=(255, 255, 255))
    draw.line([(cell_w, 0), (cell_w, canvas.height)], fill=(255, 255, 255), width=2)
    draw.line([(0, cell_h), (canvas.width, cell_h)], fill=(255, 255, 255), width=2)


def dhash(source: Path | Image.Image, *, size: int = 8) -> int:
    """Difference hash, tolerant of compression artefacts and bitrate dips.

    Comparing adjacent pixel gradients rather than absolute luminance means
    blocky IRL camera frames do not register as new scenes.
    """
    img = Image.open(source) if isinstance(source, Path) else source
    with img:
        small = img.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
        pixels = list(small.getdata())

    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits <<= 1
            if pixels[base + col] > pixels[base + col + 1]:
                bits |= 1
    return bits


def _gray_code(value: int) -> int:
    """Adjacent levels differ by exactly one bit, so quantiser boundaries cost 1."""
    return value ^ (value >> 1)


def colour_signature(source: Path | Image.Image, *, grid: int = 2, levels: int = 5) -> int:
    """Coarse per-quadrant mean colour, quantised and Gray coded.

    A difference hash is blind to flat frames: every gradient is zero whatever
    the colour, so a BRB card, a black loading screen, and a full-screen menu
    all collapse to the same value. This adds absolute colour back.
    """
    img = Image.open(source) if isinstance(source, Path) else source
    with img:
        small = img.convert("RGB").resize((grid, grid), Image.Resampling.BILINEAR)
        pixels = list(small.getdata())

    bits = 0
    shift = 8 - levels
    for pixel in pixels:
        for channel in pixel:
            bits = (bits << levels) | _gray_code(channel >> shift)
    return bits


def frame_signature(source: Path | Image.Image) -> int:
    """Combined structure and colour fingerprint used for grid deduplication."""
    if isinstance(source, Path):
        with Image.open(source) as image:
            loaded = image.convert("RGB")
            return (dhash(loaded.copy()) << 60) | colour_signature(loaded.copy())
    return (dhash(source.copy()) << 60) | colour_signature(source.copy())


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def is_near_duplicate(left: int, right: int, *, max_distance: int = 4) -> bool:
    return hamming_distance(left, right) <= max_distance


def encode_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
