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

UUID = "1a2b3c4d-5e6f-4a8b-9c0d-1e2f3a4b5c6d"


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
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
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


UUID7 = "01a039e2-ec00-71a0-8ebf-4d986fe440d0"
UUID7_EPOCH = 1_787_677_568.0
VIDEO_UUID = "1e7fa39e-09ad-47c5-9f25-f08eedafa16d"

CHANNEL_LISTING = [
    {"created_at": "2026-08-27 18:30:58", "video": {"uuid": "3e6a9d94-b3d8-45f9-bcf7-0d6c291b2da3"}},
    {"created_at": "2026-08-25 17:06:12", "video": {"uuid": VIDEO_UUID}},
    {"created_at": "2026-08-24 16:32:46", "video": {"uuid": "84e0dcbf-97b5-4e39-9535-54a07f062547"}},
]


class TestUuidHelpers:
    def test_detects_the_url_only_version(self):
        assert vod_module.uuid_version(UUID7) == 7
        assert vod_module.uuid_version(UUID) == 4

    @pytest.mark.parametrize("value", ["", "not-a-uuid", None, 12345])
    def test_a_non_uuid_has_no_version(self, value):
        assert vod_module.uuid_version(value) is None

    def test_decodes_the_embedded_timestamp(self):
        assert vod_module.uuid7_epoch(UUID7) == pytest.approx(UUID7_EPOCH, abs=1.0)

    def test_a_version_4_id_carries_no_timestamp(self):
        assert vod_module.uuid7_epoch(UUID) is None


class TestHostValidation:
    @pytest.mark.parametrize(
        "url",
        [
            f"https://kick.com/xqc/videos/{UUID}",
            f"http://kick.com/video/{UUID}",
            f"https://www.kick.com/xqc/videos/{UUID}",
            f"kick.com/xqc/videos/{UUID}",
        ],
    )
    def test_accepts_kick_hosts(self, url):
        assert vod_module.is_kick_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            f"https://example.com/videos/{UUID}",
            f"https://kick.com.evil.test/videos/{UUID}",
            f"https://notkick.com/xqc/videos/{UUID}",
            "",
        ],
    )
    def test_rejects_everything_else(self, url):
        assert not vod_module.is_kick_url(url)

    def test_a_non_kick_url_is_reported_as_such(self):
        with pytest.raises(VodResolutionError, match="not a Kick URL"):
            vod_module.resolve_vod(f"https://example.com/videos/{UUID7}")


class TestFindVideoUuidByTime:
    def _client(self):
        return FakeClient(FakeResponse(CHANNEL_LISTING))

    def test_matches_the_closest_listing_entry(self):
        found = vod_module.find_video_uuid_by_time(
            "nickwhite", UUID7_EPOCH, client=self._client()
        )
        assert found == VIDEO_UUID

    def test_queries_the_channel_listing_endpoint(self):
        client = self._client()
        vod_module.find_video_uuid_by_time("nickwhite", UUID7_EPOCH, client=client)
        assert "channels/nickwhite/videos" in client.urls[0]

    def test_a_caller_supplied_client_is_left_open(self):
        client = self._client()
        vod_module.find_video_uuid_by_time("nickwhite", UUID7_EPOCH, client=client)
        assert not client.closed

    def test_a_timestamp_outside_the_tolerance_does_not_match(self):
        assert (
            vod_module.find_video_uuid_by_time(
                "nickwhite", UUID7_EPOCH + 86_400, client=self._client()
            )
            is None
        )

    def test_the_tolerance_is_configurable(self):
        found = vod_module.find_video_uuid_by_time(
            "nickwhite", UUID7_EPOCH + 3600, tolerance=7200.0, client=self._client()
        )
        assert found == VIDEO_UUID

    def test_a_failed_listing_returns_none(self):
        client = FakeClient(FakeResponse({}, status_code=403))
        assert vod_module.find_video_uuid_by_time("x", UUID7_EPOCH, client=client) is None

    def test_an_empty_listing_returns_none(self):
        client = FakeClient(FakeResponse([]))
        assert vod_module.find_video_uuid_by_time("x", UUID7_EPOCH, client=client) is None

    def test_rows_without_a_uuid_are_skipped(self):
        client = FakeClient(FakeResponse([{"created_at": "2026-08-25 17:06:12"}]))
        assert vod_module.find_video_uuid_by_time("x", UUID7_EPOCH, client=client) is None

    def test_a_transport_error_returns_none(self):
        class Boom:
            closed = False

            def get(self, url, **kwargs):
                raise ConnectionError("network down")

            def close(self):
                self.closed = True

        assert vod_module.find_video_uuid_by_time("x", UUID7_EPOCH, client=Boom()) is None


class TestResolveVideoUuid:
    def test_a_version_4_id_passes_through_without_a_request(self):
        client = FakeClient(FakeResponse(CHANNEL_LISTING))
        assert (
            vod_module.resolve_video_uuid(f"https://kick.com/xqc/videos/{UUID}", client=client)
            == UUID
        )
        assert client.urls == []

    def test_a_url_only_id_is_mapped_through_the_channel(self):
        client = FakeClient(FakeResponse(CHANNEL_LISTING))
        resolved = vod_module.resolve_video_uuid(
            f"https://kick.com/nickwhite/videos/{UUID7}", client=client
        )
        assert resolved == VIDEO_UUID

    def test_a_url_only_id_without_a_channel_is_rejected(self):
        with pytest.raises(VodResolutionError, match="URL-only video id"):
            vod_module.resolve_video_uuid(f"https://kick.com/video/{UUID7}")

    def test_an_unmatchable_id_names_the_channel(self):
        client = FakeClient(FakeResponse([]))
        with pytest.raises(VodResolutionError, match="could not map URL id"):
            vod_module.resolve_video_uuid(
                f"https://kick.com/nickwhite/videos/{UUID7}", client=client
            )

    def test_a_url_without_any_id_is_rejected(self):
        with pytest.raises(VodResolutionError, match="no video id"):
            vod_module.resolve_video_uuid("https://kick.com/nickwhite")


class TestCanonicalVodUrl:
    def test_a_version_4_url_is_unchanged(self, monkeypatch):
        url = f"https://kick.com/xqc/videos/{UUID}"
        assert vod_module.canonical_vod_url(url) == url

    def test_a_url_only_id_is_rewritten_for_the_fallback(self, monkeypatch):
        monkeypatch.setattr(
            vod_module, "resolve_video_uuid", lambda url, timeout=30.0: VIDEO_UUID
        )
        rewritten = vod_module.canonical_vod_url(f"https://kick.com/nickwhite/videos/{UUID7}")
        assert rewritten == f"https://kick.com/nickwhite/videos/{VIDEO_UUID}"

    def test_a_url_without_an_id_is_unchanged(self):
        assert vod_module.canonical_vod_url("https://kick.com/xqc") == "https://kick.com/xqc"


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
        with pytest.raises(VodResolutionError, match="no video id"):
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


    def test_the_slug_in_the_url_wins_over_the_payload(self, monkeypatch):
        monkeypatch.setattr(
            vod_module, "build_client", lambda timeout: FakeClient(FakeResponse(VIDEO_PAYLOAD))
        )
        info = vod_module.resolve_via_api(f"https://kick.com/othername/videos/{UUID}")
        assert info.channel_slug == "othername"


class TestProbeDuration:
    def test_an_unreadable_source_returns_zero_rather_than_raising(self, tmp_path):
        assert vod_module.probe_duration(str(tmp_path / "absent.m3u8")) == 0.0


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
