from __future__ import annotations


from kick_vod_analyser.sampling import renditions as renditions_module
from kick_vod_analyser.sampling.renditions import (
    Rendition,
    closest_to_height,
    parse_master_playlist,
    plan_streams,
    smallest,
)

MASTER = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-STREAM-INF:BANDWIDTH=9307804,RESOLUTION=1920x1080,CODECS="avc1.640028",NAME="1080p"
1080p30/playlist.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=3506882,RESOLUTION=1280x720,CODECS="avc1.4d401f",NAME="720p"
720p30/playlist.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=630000,RESOLUTION=640x360,NAME="360p"
360p30/playlist.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=230000,RESOLUTION=284x160,NAME="160p"
160p30/playlist.m3u8
"""


class TestParseMasterPlaylist:
    def test_extracts_every_variant(self):
        assert len(parse_master_playlist(MASTER)) == 4

    def test_reads_bandwidth_and_resolution(self):
        first = parse_master_playlist(MASTER)[0]
        assert first.bandwidth == 9_307_804
        assert (first.width, first.height) == (1920, 1080)
        assert first.name == "1080p"

    def test_resolves_relative_urls_against_the_base(self):
        base = "https://cdn.example/path/master.m3u8"
        assert parse_master_playlist(MASTER, base)[0].url == (
            "https://cdn.example/path/1080p30/playlist.m3u8"
        )

    def test_absolute_urls_are_left_alone(self):
        text = (
            "#EXTM3U\n"
            "#EXT-X-STREAM-INF:BANDWIDTH=100,RESOLUTION=100x100\n"
            "https://other.example/x.m3u8\n"
        )
        assert parse_master_playlist(text, "https://cdn.example/m.m3u8")[0].url == (
            "https://other.example/x.m3u8"
        )

    def test_a_variant_without_a_resolution_still_parses(self):
        text = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=500000\naudio.m3u8\n"
        rendition = parse_master_playlist(text)[0]
        assert rendition.height == 0 and rendition.bandwidth == 500_000

    def test_a_media_playlist_yields_no_variants(self):
        media = "#EXTM3U\n#EXT-X-TARGETDURATION:6\n#EXTINF:6.0,\nseg1.ts\n"
        assert parse_master_playlist(media) == []

    def test_blank_lines_between_tag_and_uri_are_tolerated(self):
        text = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=100,RESOLUTION=10x10\n\n\nlow.m3u8\n"
        assert parse_master_playlist(text)[0].url == "low.m3u8"

    def test_a_tag_without_a_following_uri_is_skipped(self):
        text = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=100\n"
        assert parse_master_playlist(text) == []

    def test_empty_input_is_safe(self):
        assert parse_master_playlist("") == []


class TestSelection:
    def test_smallest_prefers_the_lowest_resolution(self):
        assert smallest(parse_master_playlist(MASTER)).height == 160

    def test_smallest_of_nothing_is_none(self):
        assert smallest([]) is None

    def test_closest_to_height_prefers_an_exact_match(self):
        assert closest_to_height(parse_master_playlist(MASTER), 720).height == 720

    def test_closest_to_height_rounds_up_when_it_can(self):
        assert closest_to_height(parse_master_playlist(MASTER), 400).height == 720

    def test_closest_to_height_falls_back_below_the_target(self):
        low_only = [Rendition(url="a", bandwidth=1, height=360, width=640)]
        assert closest_to_height(low_only, 1080).height == 360

    def test_ties_break_on_bandwidth(self):
        options = [
            Rendition(url="a", bandwidth=5_000_000, height=720, width=1280),
            Rendition(url="b", bandwidth=2_000_000, height=720, width=1280),
        ]
        assert closest_to_height(options, 720).url == "b"

    def test_label_falls_back_when_a_resolution_is_missing(self):
        assert Rendition(url="a", bandwidth=500_000).label == "500kbps"
        assert Rendition(url="a", bandwidth=500_000, name="audio").label == "audio"


class TestPlanStreams:
    def test_splits_detection_from_extraction(self, monkeypatch):
        monkeypatch.setattr(
            renditions_module,
            "fetch_master_playlist",
            lambda url, timeout=30.0: parse_master_playlist(MASTER, "https://cdn.example/m.m3u8"),
        )
        plan = plan_streams("https://cdn.example/m.m3u8", extract_height=720)

        assert plan.is_split
        assert plan.detect_rendition.height == 160
        assert plan.extract_rendition.height == 720

    def test_a_single_rendition_uses_the_source_for_both(self, monkeypatch):
        single = [Rendition(url="https://cdn/only.m3u8", bandwidth=1, height=720, width=1280)]
        monkeypatch.setattr(
            renditions_module, "fetch_master_playlist", lambda url, timeout=30.0: single
        )
        plan = plan_streams("https://cdn/master.m3u8")

        assert not plan.is_split
        assert plan.detect_url == "https://cdn/master.m3u8"

    def test_an_unreadable_playlist_falls_back_to_the_source(self, monkeypatch):
        monkeypatch.setattr(
            renditions_module, "fetch_master_playlist", lambda url, timeout=30.0: []
        )
        plan = plan_streams("https://cdn/master.m3u8")

        assert plan.detect_url == plan.extract_url == "https://cdn/master.m3u8"

    def test_a_non_playlist_source_is_used_directly(self, tmp_path):
        plan = plan_streams(str(tmp_path / "local.mp4"))
        assert plan.detect_url == plan.extract_url == str(tmp_path / "local.mp4")

    def test_a_local_file_never_triggers_a_request(self, monkeypatch, tmp_path):
        import httpx

        calls = []
        monkeypatch.setattr(httpx, "get", lambda *a, **k: calls.append(a))
        plan_streams(str(tmp_path / "local.mp4"))
        assert calls == []


class TestFetchMasterPlaylist:
    def test_a_non_playlist_url_short_circuits(self):
        assert renditions_module.fetch_master_playlist("https://cdn/video.mp4") == []

    def test_a_transport_error_returns_empty(self, monkeypatch):
        import httpx

        def boom(*args, **kwargs):
            raise httpx.ConnectError("no route")

        monkeypatch.setattr(httpx, "get", boom)
        assert renditions_module.fetch_master_playlist("https://cdn/master.m3u8") == []

    def test_a_non_200_response_returns_empty(self, monkeypatch):
        import httpx

        class Response:
            status_code = 404
            text = ""
            url = "https://cdn/master.m3u8"

        monkeypatch.setattr(httpx, "get", lambda *a, **k: Response())
        assert renditions_module.fetch_master_playlist("https://cdn/master.m3u8") == []

    def test_a_successful_response_is_parsed(self, monkeypatch):
        import httpx

        class Response:
            status_code = 200
            text = MASTER
            url = "https://cdn.example/m.m3u8"

        monkeypatch.setattr(httpx, "get", lambda *a, **k: Response())
        renditions = renditions_module.fetch_master_playlist("https://cdn.example/m.m3u8")
        assert len(renditions) == 4
        assert renditions[0].url.startswith("https://cdn.example/")

    def test_query_strings_do_not_defeat_the_extension_check(self, monkeypatch):
        import httpx

        class Response:
            status_code = 200
            text = MASTER
            url = "https://cdn.example/m.m3u8"

        monkeypatch.setattr(httpx, "get", lambda *a, **k: Response())
        assert renditions_module.fetch_master_playlist("https://cdn.example/m.m3u8?token=abc")
