from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from kick_vod_analyser.ingest.chat import (
    HISTORY_API,
    ChatIndex,
    JsonlChatSource,
    KickReplayChatSource,
    NullChatSource,
    build_chat_source,
    extract_emotes,
    load_records,
    normalise_record,
    plan_chunks,
)
from kick_vod_analyser.models import ChatMessage, VodInfo

EPOCH = 1_700_000_000.0


def msg(offset: float, text: str = "hello", username: str = "viewer") -> ChatMessage:
    return ChatMessage(offset_seconds=offset, username=username, text=text)


class TestChatIndex:
    def test_sorts_on_construction(self):
        index = ChatIndex([msg(300), msg(100), msg(200)])
        assert [m.offset_seconds for m in index.messages] == [100.0, 200.0, 300.0]

    def test_window_is_inclusive_at_both_edges(self):
        index = ChatIndex([msg(55), msg(100), msg(145), msg(146)])
        assert len(index.window(100.0, 45.0)) == 3

    def test_window_outside_any_message_is_empty(self):
        assert ChatIndex([msg(100)]).window(5000.0, 45.0) == []

    def test_empty_index_is_falsy(self):
        assert not ChatIndex()
        assert ChatIndex([msg(1)])

    def test_jsonl_round_trip(self, tmp_path):
        original = ChatIndex([msg(100, "what game is this"), msg(200, "KEKW")])
        path = original.to_jsonl(tmp_path / "chat.jsonl")
        restored = ChatIndex.from_jsonl(path)
        assert [m.text for m in restored.messages] == ["what game is this", "KEKW"]

    def test_window_is_correct_on_a_large_index(self):
        index = ChatIndex([msg(float(t)) for t in range(0, 100_000)])
        window = index.window(50_000.0, 45.0)
        assert len(window) == 91
        assert window[0].offset_seconds == 49_955.0


class TestLoadRecords:
    def test_json_array(self):
        assert len(load_records('[{"a": 1}, {"a": 2}]')) == 2

    def test_json_lines(self):
        assert len(load_records('{"a": 1}\n{"a": 2}\n')) == 2

    def test_object_wrapper_with_a_messages_key(self):
        assert load_records('{"messages": [{"a": 1}]}') == [{"a": 1}]

    def test_object_wrapper_with_a_data_key(self):
        assert load_records('{"data": [{"a": 1}]}') == [{"a": 1}]

    def test_single_object_without_a_wrapper_key(self):
        assert load_records('{"a": 1}') == [{"a": 1}]

    def test_empty_input(self):
        assert load_records("   ") == []


class TestNormaliseRecord:
    def test_kick_internal_shape(self):
        record = {
            "id": "abc",
            "content": "what game is this",
            "sender": {"username": "viewer1"},
            "created_at": "2023-11-14T22:13:40+00:00",
        }
        message = normalise_record(record, EPOCH)
        assert message.username == "viewer1"
        assert message.text == "what game is this"
        assert message.offset_seconds == pytest.approx(20.0)

    def test_chat_downloader_shape(self):
        record = {
            "message": "KEKW",
            "author": {"name": "viewer2"},
            "time_in_seconds": 1234.5,
        }
        message = normalise_record(record, EPOCH)
        assert message.offset_seconds == pytest.approx(1234.5)
        assert message.username == "viewer2"

    def test_internal_export_shape(self):
        record = {"offset_seconds": 42.0, "username": "viewer3", "text": "hi there"}
        assert normalise_record(record, None).offset_seconds == pytest.approx(42.0)

    @pytest.mark.parametrize(
        "timestamp,expected",
        [
            (EPOCH + 60, 60.0),
            ((EPOCH + 60) * 1000, 60.0),
            ((EPOCH + 60) * 1_000_000, 60.0),
        ],
    )
    def test_epoch_units_are_detected(self, timestamp, expected):
        record = {"content": "hi", "timestamp": timestamp, "username": "v"}
        assert normalise_record(record, EPOCH).offset_seconds == pytest.approx(expected, abs=0.01)

    def test_offsets_never_go_negative(self):
        record = {"content": "hi", "timestamp": EPOCH - 500, "username": "v"}
        assert normalise_record(record, EPOCH).offset_seconds == 0.0

    def test_username_falls_back_when_absent(self):
        assert normalise_record({"content": "hi", "offset_seconds": 1}, None).username == "unknown"

    @pytest.mark.parametrize("record", [{}, {"content": "  "}, {"username": "v"}, "not a dict"])
    def test_unusable_records_are_dropped(self, record):
        assert normalise_record(record, EPOCH) is None

    def test_a_record_without_a_resolvable_offset_is_dropped(self):
        assert normalise_record({"content": "hi", "timestamp": EPOCH}, None) is None

    def test_emotes_are_captured(self):
        record = {
            "content": "hi",
            "offset_seconds": 1,
            "emotes": [{"name": "KEKW"}, {"id": "123"}, "PogU"],
        }
        assert normalise_record(record, None).emotes == ("KEKW", "123", "PogU")

    def test_malformed_emotes_are_ignored(self):
        assert extract_emotes({"emotes": "not a list"}) == []


class TestJsonlChatSource:
    def test_loads_a_jsonl_export(self, tmp_path, vod):
        path = tmp_path / "chat.jsonl"
        path.write_text(
            "\n".join(
                json.dumps({"offset_seconds": t, "username": "v", "text": f"line {t}"})
                for t in (10, 20, 30)
            ),
            encoding="utf-8",
        )
        index = JsonlChatSource(path).fetch(vod)
        assert len(index) == 3

    def test_loads_a_json_array_export(self, tmp_path, vod):
        path = tmp_path / "chat.json"
        path.write_text(
            json.dumps([{"time_in_seconds": 5.0, "message": "hi", "author": {"name": "v"}}]),
            encoding="utf-8",
        )
        assert len(JsonlChatSource(path).fetch(vod)) == 1

    def test_a_missing_file_degrades_to_an_empty_index(self, tmp_path, vod):
        assert len(JsonlChatSource(tmp_path / "absent.jsonl").fetch(vod)) == 0

    def test_unusable_records_are_skipped_not_fatal(self, tmp_path, vod):
        path = tmp_path / "chat.jsonl"
        path.write_text(
            json.dumps({"offset_seconds": 1, "username": "v", "text": "ok"})
            + "\n"
            + json.dumps({"username": "v"}),
            encoding="utf-8",
        )
        assert len(JsonlChatSource(path).fetch(vod)) == 1


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append(kwargs.get("params", {}))
        if not self.pages:
            return FakeResponse({"data": {"messages": []}})
        return self.pages.pop(0)

    def close(self):
        self.closed = True


class HistoryClient:
    """Simulates web.kick.com chat history: 25 messages before `cursor`, newest first."""

    PAGE = 25

    def __init__(self, records, *, fail_status=None, fail_on_call=None, explode_on_call=None):
        self.records = sorted(records, key=lambda r: r["_epoch"], reverse=True)
        self.calls = []
        self.closed = False
        self.fail_status = fail_status
        self.fail_on_call = fail_on_call
        self.explode_on_call = explode_on_call

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        n = len(self.calls)
        if self.explode_on_call == n:
            raise ConnectionError("network down")
        if self.fail_status and (self.fail_on_call is None or self.fail_on_call == n):
            return FakeResponse({"data": {}, "message": "Invalid request"}, self.fail_status)
        cursor = int(kwargs["params"]["cursor"]) / 1_000_000
        page = [r for r in self.records if r["_epoch"] < cursor][: self.PAGE]
        clean = [{k: v for k, v in r.items() if k != "_epoch"} for r in page]
        next_cursor = str(int(page[-1]["_epoch"] * 1_000_000)) if page else None
        return FakeResponse({"data": {"messages": clean, "cursor": next_cursor}})

    def close(self):
        self.closed = True


def kick_record(index, epoch, content="hello"):
    stamp = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "id": f"m{index}",
        "chat_id": 42,
        "content": content,
        "type": "message",
        "sender": {"id": index, "username": f"user{index}"},
        "created_at": stamp,
        "_epoch": float(epoch),
    }


def install(monkeypatch, client):
    from kick_vod_analyser.ingest import chat as chat_module

    monkeypatch.setattr(chat_module, "build_client", lambda timeout: client)


class TestPlanChunks:
    def test_covers_the_window_exactly(self):
        chunks = plan_chunks(0.0, 1500.0, 600.0)
        assert chunks == [(0.0, 600.0), (600.0, 1200.0), (1200.0, 1500.0)]

    def test_empty_window_has_no_chunks(self):
        assert plan_chunks(10.0, 10.0, 600.0) == []


class TestKickReplayChatSource:
    def source(self, **kwargs):
        kwargs.setdefault("workers", 3)
        kwargs.setdefault("chunk_seconds", 600.0)
        kwargs.setdefault("retry_sleep", lambda s: None)
        return KickReplayChatSource(**kwargs)

    def test_collects_every_message_across_chunks_in_order(self, monkeypatch, vod):
        records = [kick_record(i, EPOCH + i * 7, f"line {i}") for i in range(500)]
        client = HistoryClient(records)
        install(monkeypatch, client)

        index = self.source().fetch(vod)

        assert len(index) == 500
        assert [m.offset_seconds for m in index.messages] == [i * 7.0 for i in range(500)]
        assert index.messages[3].username == "user3"
        assert client.closed

    def test_bursts_larger_than_a_page_are_not_truncated(self, monkeypatch, vod):
        records = [kick_record(i, EPOCH + 100 + i * 0.01) for i in range(80)]
        install(monkeypatch, HistoryClient(records))

        assert len(self.source().fetch(vod)) == 80

    def test_messages_outside_the_vod_window_are_dropped(self, monkeypatch, vod):
        records = [
            kick_record(1, EPOCH - 60, "before"),
            kick_record(2, EPOCH + 30, "inside"),
            kick_record(3, EPOCH + vod.duration_seconds + 60, "after"),
        ]
        install(monkeypatch, HistoryClient(records))

        assert [m.text for m in self.source().fetch(vod).messages] == ["inside"]

    def test_request_count_scales_with_messages_not_duration(self, monkeypatch, vod):
        client = HistoryClient([kick_record(1, EPOCH + 5)])
        install(monkeypatch, client)

        self.source(chunk_seconds=600.0).fetch(vod)

        assert len(client.calls) == 7

    def test_requests_use_the_cursor_form_and_web_platform_header(self, monkeypatch, vod):
        client = HistoryClient([])
        install(monkeypatch, client)

        self.source().fetch(vod)

        call = client.calls[0]
        assert call["url"] == HISTORY_API.format(chat_id=42)
        assert set(call["params"]) == {"cursor"}
        assert call["headers"]["x-app-platform"] == "web"
        assert "Authorization" not in call["headers"]

    def test_auth_token_is_sent_as_bearer_when_configured(self, monkeypatch, vod):
        client = HistoryClient([])
        install(monkeypatch, client)

        self.source(auth_token="abc").fetch(vod)

        assert client.calls[0]["headers"]["Authorization"] == "Bearer abc"

    def test_a_400_stops_the_walk_without_raising(self, monkeypatch, vod):
        client = HistoryClient([kick_record(1, EPOCH + 5)], fail_status=400)
        install(monkeypatch, client)

        assert len(self.source().fetch(vod)) == 0
        assert client.closed

    def test_transient_errors_are_retried(self, monkeypatch, vod):
        client = HistoryClient([kick_record(1, EPOCH + 5)], fail_status=429, fail_on_call=1)
        install(monkeypatch, client)

        assert len(self.source(workers=1).fetch(vod)) == 1

    def test_a_failing_chunk_keeps_the_others(self, monkeypatch, vod):
        records = [kick_record(1, EPOCH + 5, "first"), kick_record(2, EPOCH + 3000, "last")]
        client = HistoryClient(records, explode_on_call=1)
        install(monkeypatch, client)

        index = self.source(workers=1).fetch(vod)

        assert [m.text for m in index.messages] == ["last"]

    def test_duplicates_on_chunk_boundaries_collapse(self, monkeypatch, vod):
        records = [kick_record(1, EPOCH + 600, "edge")]
        install(monkeypatch, HistoryClient(records))

        assert len(self.source().fetch(vod)) == 1

    def test_download_returns_raw_kick_records(self, monkeypatch, vod):
        install(monkeypatch, HistoryClient([kick_record(7, EPOCH + 9, "raw")]))

        raw = self.source().download(vod)

        assert raw[0]["id"] == "m7"
        assert raw[0]["sender"]["username"] == "user7"
        assert "_epoch" not in raw[0]

    def test_inline_emote_tokens_are_parsed(self, monkeypatch, vod):
        install(monkeypatch, HistoryClient([kick_record(1, EPOCH + 1, "[emote:37233:PogU] wow")]))

        message = self.source().fetch(vod).messages[0]

        assert message.text == "PogU wow"
        assert message.emotes == ("PogU",)

    def test_missing_channel_id_skips_without_requests(self, monkeypatch, vod):
        client = HistoryClient([])
        install(monkeypatch, client)
        no_channel = vod.model_copy(update={"channel_id": None})

        assert len(self.source().fetch(no_channel)) == 0
        assert client.calls == []

    def test_missing_start_time_skips_without_requests(self, monkeypatch, vod):
        client = HistoryClient([])
        install(monkeypatch, client)
        no_start = vod.model_copy(update={"started_at_epoch": None})

        assert len(self.source().fetch(no_start)) == 0
        assert client.calls == []

    def test_page_limit_is_honoured(self, monkeypatch, vod):
        records = [kick_record(i, EPOCH + 1 + i * 0.001) for i in range(100)]
        install(monkeypatch, HistoryClient(records))

        assert len(self.source(max_pages_per_chunk=2).fetch(vod)) == 50




class TestBuildChatSource:
    def test_none_returns_the_null_source(self):
        assert isinstance(build_chat_source("none"), NullChatSource)

    def test_kick_returns_the_replay_source(self):
        assert isinstance(build_chat_source("kick"), KickReplayChatSource)

    def test_file_requires_a_path(self):
        with pytest.raises(ValueError):
            build_chat_source("file")

    def test_an_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError):
            build_chat_source("telepathy")

    def test_the_null_source_yields_an_empty_index(self, vod):
        assert len(NullChatSource().fetch(vod)) == 0

    def test_kick_passes_token_and_workers(self):
        source = build_chat_source("kick", auth_token="t", workers=3)
        assert isinstance(source, KickReplayChatSource)
        assert source.auth_token == "t"
        assert source.workers == 3
