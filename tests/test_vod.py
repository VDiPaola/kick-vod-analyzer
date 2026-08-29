from __future__ import annotations

import json

import pytest

from kick_vod_analyser.ingest import vod as vod_module
from kick_vod_analyser.ingest.vod import (
    VodResolutionError,
    dig,
    normalise_duration,
    parse_epoch,
    parse_vod_url,
    resolve_vod,
)

UUID = "1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d"


class TestParseVodUrl:
    @pytest.mark.parametrize(
        "url,expected_slug",
        [
            (f"https://kick.com/xqc/videos/{UUID}", "xqc"),
            (f"https://kick.com/video/{UUID}", None),
            (f"https://kick.com/some_streamer-1/videos/{UUID}?t=60", "some_streamer-1"),
            (f"http://kick.com/xqc/videos/{UUID}", "xqc"),
        ],
    )
    def test_extracts_uuid_and_slug(self, url, expected_slug):
        uuid, slug = parse_vod_url(url)
        assert uuid == UUID
        assert slug == expected_slug

    def test_uppercase_uuid_is_accepted(self):
        uuid, _ = parse_vod_url(f"https://kick.com/video/{UUID.upper()}")
        assert uuid.lower() == UUID

    def test_a_url_without_a_uuid_yields_none(self):
        assert parse_vod_url("https://kick.com/xqc") == (None, "xqc")

    def test_a_non_kick_url_yields_no_slug(self):
        assert parse_vod_url(f"https://example.com/{UUID}") == (UUID, None)


class TestNormaliseDuration:
    @pytest.mark.parametrize(
        "raw,expected", [(36_000_000, 36000.0), (36000, 36000.0), (None, 0.0), (0, 0.0)]
    )
    def test_detects_milliseconds(self, raw, expected):
        assert normalise_duration(raw) == pytest.approx(expected)


class TestParseEpoch:
    @pytest.mark.parametrize(
        "value",
        ["2023-11-14T22:13:20+00:00", "2023-11-14T22:13:20Z", "2023-11-14 22:13:20"],
    )
    def test_accepts_the_shapes_kick_returns(self, value):
        assert parse_epoch(value) == pytest.approx(1_700_000_000.0, abs=0.001)

    def test_naive_timestamps_are_treated_as_utc(self):
        assert parse_epoch("2023-11-14 22:13:20") == parse_epoch("2023-11-14T22:13:20+00:00")

    @pytest.mark.parametrize("value", [None, "", "not a date", 0])
    def test_unparseable_values_yield_none(self, value):
        assert parse_epoch(value) is None


class TestDig:
    def test_returns_the_first_populated_path(self):
        payload = {"a": {"b": None}, "c": {"d": 7}}
        assert dig(payload, ("a", "b"), ("c", "d")) == 7

    def test_missing_paths_yield_none(self):
        assert dig({"a": 1}, ("x", "y")) is None

    def test_treats_empty_string_and_zero_as_missing(self):
        assert dig({"a": "", "b": 0, "c": 5}, ("a",), ("b",), ("c",)) == 5


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.closed = False

    def get(self, url, **kwargs):
        return self.response

    def close(self):
        self.closed = True


VIDEO_PAYLOAD = {
    "id": 999,
    "source": "https://cdn.kick.com/master.m3u8",
    "livestream": {
        "duration": 36_000_000,
        "session_title": "10 hour marathon",
        "start_time": "2023-11-14 22:13:20",
        "channel": {"id": 77, "slug": "teststreamer"},
    },
}


class TestResolveViaApi:
    def test_maps_the_kick_payload_onto_vod_info(self, monkeypatch):
        client = FakeClient(FakeResponse(VIDEO_PAYLOAD))
        monkeypatch.setattr(vod_module, "build_client", lambda timeout: client)

        info = vod_module.resolve_via_api(f"https://kick.com/teststreamer/videos/{UUID}")

        assert info.vod_id == UUID
        assert info.channel_slug == "teststreamer"
        assert info.channel_id == 77
        assert info.duration_seconds == pytest.approx(36000.0)
        assert info.playback_url == "https://cdn.kick.com/master.m3u8"
        assert info.started_at_epoch is not None
        assert client.closed

    def test_a_url_without_a_uuid_is_rejected(self):
        with pytest.raises(VodResolutionError, match="no video UUID"):
            vod_module.resolve_via_api("https://kick.com/xqc")

    def test_a_cloudflare_block_raises(self, monkeypatch):
        monkeypatch.setattr(
            vod_module, "build_client", lambda timeout: FakeClient(FakeResponse({}, 403))
        )
        with pytest.raises(VodResolutionError, match="403"):
            vod_module.resolve_via_api(f"https://kick.com/video/{UUID}")

    def test_a_payload_without_a_duration_raises(self, monkeypatch):
        monkeypatch.setattr(
            vod_module,
            "build_client",
            lambda timeout: FakeClient(FakeResponse({"source": "x", "livestream": {}})),
        )
        monkeypatch.setattr(vod_module, "probe_duration", lambda source, **k: 0.0)
        with pytest.raises(VodResolutionError, match="duration"):
            vod_module.resolve_via_api(f"https://kick.com/video/{UUID}")

    def test_a_live_vod_falls_back_to_probing_the_playlist(self, monkeypatch):
        """Kick reports duration 0 until a stream ends; the playlist still knows."""
        payload = dict(VIDEO_PAYLOAD)
        payload["livestream"] = dict(VIDEO_PAYLOAD["livestream"], duration=0)
        monkeypatch.setattr(
            vod_module, "build_client", lambda timeout: FakeClient(FakeResponse(payload))
        )
        monkeypatch.setattr(vod_module, "probe_duration", lambda source, **k: 1234.0)

        info = vod_module.resolve_via_api(f"https://kick.com/video/{UUID}")
        assert info.duration_seconds == pytest.approx(1234.0)

    def test_probing_is_skipped_when_the_api_duration_is_usable(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            vod_module, "build_client", lambda timeout: FakeClient(FakeResponse(VIDEO_PAYLOAD))
        )
        monkeypatch.setattr(vod_module, "probe_duration", lambda source, **k: calls.append(source))

        vod_module.resolve_via_api(f"https://kick.com/video/{UUID}")
        assert calls == []


class TestProbeDuration:
    def test_an_unreadable_source_returns_zero_rather_than_raising(self, tmp_path):
        assert vod_module.probe_duration(str(tmp_path / "absent.m3u8")) == 0.0

    def test_the_slug_in_the_url_wins_over_the_payload(self, monkeypatch):
        monkeypatch.setattr(
            vod_module, "build_client", lambda timeout: FakeClient(FakeResponse(VIDEO_PAYLOAD))
        )
        info = vod_module.resolve_via_api(f"https://kick.com/othername/videos/{UUID}")
        assert info.channel_slug == "othername"


class TestResolveVod:
    def test_falls_back_to_ytdlp_when_the_api_fails(self, monkeypatch):
        monkeypatch.setattr(
            vod_module,
            "resolve_via_api",
            lambda url, timeout=30.0: (_ for _ in ()).throw(VodResolutionError("blocked")),
        )
        from kick_vod_analyser.models import VodInfo

        fallback = VodInfo(
            vod_id=UUID,
            url="u",
            channel_slug="teststreamer",
            duration_seconds=100.0,
            playback_url="https://cdn/x.m3u8",
        )
        monkeypatch.setattr(vod_module, "resolve_via_ytdlp", lambda url: fallback)

        assert resolve_vod(f"https://kick.com/video/{UUID}") is fallback

    def test_a_result_without_a_playback_url_is_rejected(self, monkeypatch):
        from kick_vod_analyser.models import VodInfo

        useless = VodInfo(vod_id=UUID, url="u", channel_slug="s", duration_seconds=10.0)
        monkeypatch.setattr(vod_module, "resolve_via_api", lambda url, timeout=30.0: useless)
        monkeypatch.setattr(vod_module, "resolve_via_ytdlp", lambda url: useless)

        with pytest.raises(VodResolutionError, match="no playback URL"):
            resolve_vod(f"https://kick.com/video/{UUID}")

    def test_both_strategies_failing_reports_both_reasons(self, monkeypatch):
        def boom_api(url, timeout=30.0):
            raise VodResolutionError("api down")

        def boom_ytdlp(url):
            raise VodResolutionError("ytdlp down")

        monkeypatch.setattr(vod_module, "resolve_via_api", boom_api)
        monkeypatch.setattr(vod_module, "resolve_via_ytdlp", boom_ytdlp)

        with pytest.raises(VodResolutionError) as excinfo:
            resolve_vod(f"https://kick.com/video/{UUID}")
        assert "api down" in str(excinfo.value)
        assert "ytdlp down" in str(excinfo.value)

    def test_prefer_ytdlp_tries_it_first(self, monkeypatch):
        order = []

        def api(url, timeout=30.0):
            order.append("api")
            raise VodResolutionError("no")

        def ytdlp(url):
            order.append("ytdlp")
            raise VodResolutionError("no")

        monkeypatch.setattr(vod_module, "resolve_via_api", api)
        monkeypatch.setattr(vod_module, "resolve_via_ytdlp", ytdlp)

        with pytest.raises(VodResolutionError):
            resolve_vod("https://kick.com/video/x", prefer="ytdlp")
        assert order == ["ytdlp", "api"]


class TestResolveViaYtdlp:
    def test_maps_ytdlp_json_onto_vod_info(self, monkeypatch):
        payload = {
            "id": UUID,
            "duration": 3600.0,
            "title": "marathon",
            "uploader_id": "teststreamer",
            "timestamp": 1_700_000_000,
            "formats": [{"url": "https://cdn/low.m3u8"}, {"url": "https://cdn/high.m3u8"}],
        }

        class Completed:
            returncode = 0
            stdout = json.dumps(payload)
            stderr = ""

        monkeypatch.setattr(vod_module.subprocess, "run", lambda *a, **k: Completed())
        monkeypatch.setattr(vod_module.shutil, "which", lambda name: "yt-dlp")

        info = vod_module.resolve_via_ytdlp(f"https://kick.com/video/{UUID}")

        assert info.duration_seconds == pytest.approx(3600.0)
        assert info.playback_url == "https://cdn/high.m3u8"
        assert info.channel_slug == "teststreamer"

    def test_a_non_zero_exit_raises(self, monkeypatch):
        class Failed:
            returncode = 1
            stdout = ""
            stderr = "ERROR: unsupported URL"

        monkeypatch.setattr(vod_module.subprocess, "run", lambda *a, **k: Failed())
        monkeypatch.setattr(vod_module.shutil, "which", lambda name: "yt-dlp")

        with pytest.raises(VodResolutionError, match="unsupported URL"):
            vod_module.resolve_via_ytdlp("https://kick.com/video/x")
