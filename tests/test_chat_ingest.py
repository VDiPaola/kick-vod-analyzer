from __future__ import annotations

import json

import pytest

from kick_vod_analyser.ingest.chat import (
    ChatIndex,
    JsonlChatSource,
    KickReplayChatSource,
    NullChatSource,
    build_chat_source,
    extract_emotes,
    load_records,
    normalise_record,
)
from kick_vod_analyser.models import ChatMessage

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


class TestKickReplayChatSource:
    def _page(self, records):
        return FakeResponse({"data": {"messages": records}})

    def test_walks_forward_and_collects_messages(self, monkeypatch, vod):
        from kick_vod_analyser.ingest import chat as chat_module

        pages = [
            self._page(
                [
                    {"id": "1", "content": "a", "timestamp": EPOCH + 10, "username": "v1"},
                    {"id": "2", "content": "b", "timestamp": EPOCH + 20, "username": "v2"},
                ]
            ),
            self._page([{"id": "3", "content": "c", "timestamp": EPOCH + 90, "username": "v3"}]),
        ]
        client = FakeClient(pages)
        monkeypatch.setattr(chat_module, "build_client", lambda timeout: client)

        index = KickReplayChatSource(max_empty_pages=2).fetch(vod)

        assert [m.text for m in index.messages] == ["a", "b", "c"]
        assert client.closed

    def test_duplicate_ids_are_not_counted_twice(self, monkeypatch, vod):
        from kick_vod_analyser.ingest import chat as chat_module

        record = {"id": "1", "content": "a", "timestamp": EPOCH + 10, "username": "v1"}
        client = FakeClient([self._page([record]), self._page([record])])
        monkeypatch.setattr(chat_module, "build_client", lambda timeout: client)

        assert len(KickReplayChatSource(max_empty_pages=1).fetch(vod)) == 1

    def test_messages_past_the_vod_end_are_discarded(self, monkeypatch, vod):
        from kick_vod_analyser.ingest import chat as chat_module

        client = FakeClient(
            [
                self._page(
                    [
                        {"id": "1", "content": "in", "timestamp": EPOCH + 10, "username": "v"},
                        {"id": "2", "content": "out", "timestamp": EPOCH + 99999, "username": "v"},
                    ]
                )
            ]
        )
        monkeypatch.setattr(chat_module, "build_client", lambda timeout: client)

        index = KickReplayChatSource(max_empty_pages=1).fetch(vod)
        assert [m.text for m in index.messages] == ["in"]

    def test_a_non_200_response_stops_the_walk_without_raising(self, monkeypatch, vod):
        from kick_vod_analyser.ingest import chat as chat_module

        client = FakeClient([FakeResponse({}, status_code=403)])
        monkeypatch.setattr(chat_module, "build_client", lambda timeout: client)

        assert len(KickReplayChatSource().fetch(vod)) == 0

    def test_a_transport_error_returns_what_was_collected(self, monkeypatch, vod):
        from kick_vod_analyser.ingest import chat as chat_module

        class ExplodingClient(FakeClient):
            def get(self, url, **kwargs):
                if self.calls:
                    raise ConnectionError("network down")
                self.calls.append(kwargs)
                return FakeResponse(
                    {
                        "data": {
                            "messages": [
                                {"id": "1", "content": "a", "timestamp": EPOCH + 5, "username": "v"}
                            ]
                        }
                    }
                )

        client = ExplodingClient([])
        monkeypatch.setattr(chat_module, "build_client", lambda timeout: client)

        assert len(KickReplayChatSource().fetch(vod)) == 1
        assert client.closed

    def test_empty_pages_advance_the_cursor(self, monkeypatch, vod):
        from kick_vod_analyser.ingest import chat as chat_module

        client = FakeClient([])
        monkeypatch.setattr(chat_module, "build_client", lambda timeout: client)

        KickReplayChatSource(page_step_seconds=600.0, max_empty_pages=3).fetch(vod)

        cursors = [call["start_time"] for call in client.calls]
        assert cursors == sorted(cursors) and len(set(cursors)) == len(cursors)

    def test_missing_channel_id_skips_the_walk(self, vod):
        stripped = vod.model_copy(update={"channel_id": None})
        assert len(KickReplayChatSource().fetch(stripped)) == 0

    def test_missing_start_time_skips_the_walk(self, vod):
        stripped = vod.model_copy(update={"started_at_epoch": None})
        assert len(KickReplayChatSource().fetch(stripped)) == 0


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
