from __future__ import annotations

import pytest
from PIL import Image

from kick_vod_analyser.sampling.grid import (
    colour_signature,
    compose_grid,
    dhash,
    encode_base64,
    frame_signature,
    hamming_distance,
    is_near_duplicate,
)


def cell_colour(image: Image.Image, quadrant: int) -> tuple[int, int, int]:
    """Sample the centre of one quadrant, away from labels and dividers."""
    half_w, half_h = image.width // 2, image.height // 2
    origins = ((0, 0), (half_w, 0), (0, half_h), (half_w, half_h))
    x, y = origins[quadrant]
    return image.getpixel((x + half_w // 2, y + half_h // 2))


def approx_colour(actual, expected, tolerance: int = 40) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(actual, expected))


class TestComposeGrid:
    def test_produces_the_requested_dimensions(self, tmp_path, solid_frame):
        frames = [solid_frame((255, 0, 0), f"f{i}.jpg") for i in range(4)]
        out = compose_grid(frames, tmp_path / "grid.jpg", width=1280, height=720)
        with Image.open(out) as image:
            assert image.size == (1280, 720)
            assert image.format == "JPEG"

    def test_places_frames_in_temporal_reading_order(self, tmp_path, solid_frame):
        colours = [(220, 20, 20), (20, 220, 20), (20, 20, 220), (220, 220, 20)]
        frames = [solid_frame(c, f"f{i}.jpg") for i, c in enumerate(colours)]
        out = compose_grid(frames, tmp_path / "grid.jpg", width=640, height=360, annotate=False)
        with Image.open(out) as image:
            for index, expected in enumerate(colours):
                assert approx_colour(cell_colour(image, index), expected), index

    def test_missing_frames_leave_black_cells(self, tmp_path, solid_frame):
        frames = [solid_frame((220, 20, 20), "f0.jpg"), solid_frame((20, 220, 20), "f1.jpg")]
        out = compose_grid(frames, tmp_path / "grid.jpg", width=640, height=360, annotate=False)
        with Image.open(out) as image:
            assert approx_colour(cell_colour(image, 2), (0, 0, 0))
            assert approx_colour(cell_colour(image, 3), (0, 0, 0))

    def test_extra_frames_beyond_four_are_ignored(self, tmp_path, solid_frame):
        frames = [solid_frame((10 * i, 10 * i, 10 * i), f"f{i}.jpg") for i in range(8)]
        out = compose_grid(frames, tmp_path / "grid.jpg", width=640, height=360)
        assert out.exists()

    def test_unreadable_frame_does_not_abort_the_grid(self, tmp_path, solid_frame):
        broken = tmp_path / "broken.jpg"
        broken.write_bytes(b"not an image")
        frames = [solid_frame((220, 20, 20), "f0.jpg"), broken]
        out = compose_grid(frames, tmp_path / "grid.jpg", width=640, height=360, annotate=False)
        with Image.open(out) as image:
            assert approx_colour(cell_colour(image, 0), (220, 20, 20))

    def test_preserves_aspect_ratio_by_letterboxing(self, tmp_path, solid_frame):
        wide = solid_frame((220, 20, 20), "wide.jpg", size=(400, 100))
        out = compose_grid([wide], tmp_path / "grid.jpg", width=640, height=360, annotate=False)
        with Image.open(out) as image:
            assert approx_colour(image.getpixel((160, 10)), (0, 0, 0))
            assert approx_colour(cell_colour(image, 0), (220, 20, 20))

    def test_rejects_an_empty_frame_list(self, tmp_path):
        with pytest.raises(ValueError):
            compose_grid([], tmp_path / "grid.jpg")

    def test_creates_missing_parent_directories(self, tmp_path, solid_frame):
        frames = [solid_frame((5, 5, 5), "f0.jpg")]
        out = compose_grid(frames, tmp_path / "a" / "b" / "grid.jpg")
        assert out.exists()

    def test_lower_quality_yields_a_smaller_file(self, tmp_path, solid_frame):
        frames = [solid_frame((i * 30, 200 - i * 20, i * 10), f"f{i}.jpg") for i in range(4)]
        big = compose_grid(frames, tmp_path / "hi.jpg", quality=95)
        small = compose_grid(frames, tmp_path / "lo.jpg", quality=30)
        assert small.stat().st_size < big.stat().st_size


class TestPerceptualHash:
    def test_identical_images_hash_the_same(self, tmp_path, solid_frame):
        a = solid_frame((120, 80, 40), "a.jpg")
        b = solid_frame((120, 80, 40), "b.jpg")
        assert dhash(a) == dhash(b)

    def test_tolerates_recompression(self, tmp_path):
        """A bitrate dip must not read as a new scene.

        The source mimics a real frame: smooth gradients plus a few solid HUD
        blocks, which is the content profile JPEG artefacts actually degrade.
        """
        source = Image.new("RGB", (320, 180))
        for x in range(320):
            for y in range(180):
                source.putpixel((x, y), (x * 255 // 320, y * 255 // 180, 90))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(source)
        draw.rectangle([20, 20, 120, 70], fill=(240, 240, 240))
        draw.rectangle([200, 110, 300, 160], fill=(10, 10, 10))

        high, low = tmp_path / "high.jpg", tmp_path / "low.jpg"
        source.save(high, quality=95)
        source.save(low, quality=15)
        assert is_near_duplicate(dhash(high), dhash(low), max_distance=6)

    def test_distinct_content_is_not_a_near_duplicate(self, tmp_path):
        left = Image.new("RGB", (64, 64), (0, 0, 0))
        for x in range(0, 64, 2):
            for y in range(64):
                left.putpixel((x, y), (255, 255, 255))
        right = Image.new("RGB", (64, 64), (0, 0, 0))
        for y in range(0, 64, 2):
            for x in range(64):
                right.putpixel((x, y), (255, 255, 255))
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        left.save(a)
        right.save(b)
        assert not is_near_duplicate(dhash(a), dhash(b), max_distance=4)

    def test_hamming_distance_counts_differing_bits(self):
        assert hamming_distance(0b1010, 0b1010) == 0
        assert hamming_distance(0b1010, 0b0101) == 4

    def test_hash_fits_the_requested_bit_width(self, solid_frame):
        assert dhash(solid_frame((10, 20, 30), "a.jpg"), size=8) < 2**64

    def test_accepts_an_open_image(self, solid_frame):
        path = solid_frame((10, 20, 30), "a.jpg")
        with Image.open(path) as image:
            assert dhash(image.copy()) == dhash(path)


class TestFrameSignature:
    def test_separates_flat_frames_that_dhash_alone_cannot(self, solid_frame):
        red = solid_frame((200, 10, 10), "r.jpg")
        green = solid_frame((10, 200, 10), "g.jpg")
        assert dhash(red) == dhash(green), "premise: a difference hash is blind here"
        assert not is_near_duplicate(
            frame_signature(red), frame_signature(green), max_distance=6
        )

    def test_separates_a_brb_card_from_a_black_screen(self, solid_frame):
        black = solid_frame((0, 0, 0), "black.jpg")
        card = solid_frame((30, 30, 90), "card.jpg")
        assert not is_near_duplicate(
            frame_signature(black), frame_signature(card), max_distance=6
        )

    def test_identical_frames_match_exactly(self, solid_frame):
        a = solid_frame((77, 88, 99), "a.jpg")
        b = solid_frame((77, 88, 99), "b.jpg")
        assert frame_signature(a) == frame_signature(b)

    def test_survives_recompression_within_the_default_threshold(self, tmp_path):
        from PIL import ImageDraw

        source = Image.new("RGB", (320, 180))
        for x in range(320):
            for y in range(180):
                source.putpixel((x, y), (x * 255 // 320, y * 255 // 180, 90))
        draw = ImageDraw.Draw(source)
        draw.rectangle([20, 20, 120, 70], fill=(240, 240, 240))
        high, low = tmp_path / "high.jpg", tmp_path / "low.jpg"
        source.save(high, quality=95)
        source.save(low, quality=15)
        assert is_near_duplicate(frame_signature(high), frame_signature(low), max_distance=6)

    def test_colour_component_uses_gray_coding(self):
        """Neighbouring quantiser levels must differ by a single bit."""
        left = Image.new("RGB", (8, 8), (100, 100, 100))
        right = Image.new("RGB", (8, 8), (104, 100, 100))
        distance = hamming_distance(colour_signature(left), colour_signature(right))
        assert distance <= 4


class TestEncodeBase64:
    def test_roundtrips_the_file_bytes(self, tmp_path, solid_frame):
        import base64

        path = solid_frame((1, 2, 3), "a.jpg")
        assert base64.b64decode(encode_base64(path)) == path.read_bytes()
