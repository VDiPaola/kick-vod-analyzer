from __future__ import annotations

import pytest
from typer.testing import CliRunner

from kick_vod_analyser import cli as cli_module
from kick_vod_analyser.cli import app
from kick_vod_analyser.models import ChatMessage, VodInfo

runner = CliRunner()

VOD = VodInfo(
    vod_id="abc",
    url="https://kick.com/teststreamer/videos/abc",
    channel_slug="teststreamer",
    channel_id=7,
    title="marathon",
    duration_seconds=36000.0,
    playback_url="https://cdn/master.m3u8",
)


class TestHelp:
    def test_lists_every_command(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("analyse", "estimate", "info", "chat"):
            assert command in result.stdout

    @pytest.mark.parametrize("command", ["analyse", "estimate", "info", "chat"])
    def test_each_command_documents_itself(self, command):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert "--url" in result.stdout


class TestInfo:
    def test_prints_resolved_metadata(self, monkeypatch):
        monkeypatch.setattr(cli_module, "resolve_vod", lambda url, timeout=30.0: VOD)
        result = runner.invoke(app, ["info", "--url", VOD.url])
        assert result.exit_code == 0
        assert "teststreamer" in result.stdout
        assert "10h 0m" in result.stdout

    def test_a_resolution_failure_exits_non_zero(self, monkeypatch):
        def boom(url, timeout=30.0):
            raise RuntimeError("blocked")

        monkeypatch.setattr(cli_module, "resolve_vod", boom)
        assert runner.invoke(app, ["info", "--url", VOD.url]).exit_code != 0


class TestChat:
    def _install(self, monkeypatch, messages, raw=None):
        from kick_vod_analyser.ingest.chat import ChatIndex

        monkeypatch.setattr(cli_module, "resolve_vod", lambda url, timeout=30.0: VOD)

        class FakeSource:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fetch(self, vod):
                return ChatIndex(messages)

            def download(self, vod):
                return raw or []

        monkeypatch.setattr(cli_module, "KickReplayChatSource", FakeSource)

    def test_writes_normalised_jsonl(self, monkeypatch, tmp_path):
        self._install(monkeypatch, [ChatMessage(offset_seconds=5, username="v", text="hi")])
        target = tmp_path / "chat.jsonl"

        result = runner.invoke(app, ["chat", "--url", VOD.url, "--out", str(target)])

        assert result.exit_code == 0, result.stdout
        assert "1 messages" in result.stdout
        assert '"text":"hi"' in target.read_text(encoding="utf-8")

    def test_raw_writes_kick_records(self, monkeypatch, tmp_path):
        self._install(monkeypatch, [], raw=[{"id": "m1", "content": "yo", "sender": {"username": "v"}}])
        target = tmp_path / "raw.jsonl"

        result = runner.invoke(app, ["chat", "--url", VOD.url, "--out", str(target), "--raw"])

        assert result.exit_code == 0, result.stdout
        assert '"id": "m1"' in target.read_text(encoding="utf-8")

    def test_defaults_to_the_vod_out_dir(self, monkeypatch, tmp_path):
        self._install(monkeypatch, [ChatMessage(offset_seconds=5, username="v", text="hi")])
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["chat", "--url", VOD.url])

        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "out" / "abc" / "chat.jsonl").exists()

    def test_no_messages_exits_non_zero(self, monkeypatch, tmp_path):
        self._install(monkeypatch, [])

        result = runner.invoke(app, ["chat", "--url", VOD.url, "--out", str(tmp_path / "c.jsonl")])

        assert result.exit_code == 1


class TestEstimate:
    def test_reports_a_cost_table(self, monkeypatch):
        monkeypatch.setattr(cli_module, "resolve_vod", lambda url, timeout=30.0: VOD)
        result = runner.invoke(app, ["estimate", "--url", VOD.url, "--samples", "100"])
        assert result.exit_code == 0
        assert "Total" in result.stdout
        assert "$" in result.stdout

    def test_works_without_vod_metadata(self, monkeypatch):
        def boom(url, timeout=30.0):
            raise RuntimeError("blocked")

        monkeypatch.setattr(cli_module, "resolve_vod", boom)
        result = runner.invoke(app, ["estimate", "--url", VOD.url])
        assert result.exit_code == 0
        assert "unavailable" in result.stdout

    def test_batch_mode_is_cheaper_than_sync(self, monkeypatch):
        monkeypatch.setattr(cli_module, "resolve_vod", lambda url, timeout=30.0: VOD)
        sync = runner.invoke(app, ["estimate", "--url", VOD.url, "--mode", "sync"])
        batch = runner.invoke(app, ["estimate", "--url", VOD.url, "--mode", "batch"])
        assert "(sync)" in sync.stdout and "(batch)" in batch.stdout

    def test_the_provider_selects_the_default_model(self, monkeypatch):
        monkeypatch.setattr(cli_module, "resolve_vod", lambda url, timeout=30.0: VOD)
        result = runner.invoke(app, ["estimate", "--url", VOD.url, "--provider", "openai"])
        assert "gpt-4o-mini" in result.stdout


class TestAnalyseWiring:
    def _capture(self, monkeypatch):
        captured = {}

        class FakePipeline:
            def __init__(self, settings, chat_source=None, progress=None):
                captured["settings"] = settings
                captured["chat_source"] = chat_source

            def run(self, options):
                captured["options"] = options
                from kick_vod_analyser.pipeline import RunReport

                return RunReport()

        monkeypatch.setattr(cli_module, "Pipeline", FakePipeline)
        return captured

    def test_sampling_overrides_reach_the_settings(self, monkeypatch):
        captured = self._capture(monkeypatch)
        runner.invoke(
            app,
            [
                "analyse",
                "--url", VOD.url,
                "--provider", "mock",
                "--scene-threshold", "0.5",
                "--heartbeat", "600",
                "--max-samples", "25",
            ],
        )
        sampling = captured["settings"].sampling
        assert sampling.scene_threshold == pytest.approx(0.5)
        assert sampling.heartbeat_seconds == pytest.approx(600.0)
        assert sampling.max_samples == 25

    def test_run_options_reflect_the_flags(self, monkeypatch):
        captured = self._capture(monkeypatch)
        runner.invoke(
            app,
            [
                "analyse",
                "--url", VOD.url,
                "--provider", "openai",
                "--model", "gpt-4.1-mini",
                "--mode", "batch",
                "--chat", "none",
                "--no-resume",
                "--keep-frames",
                "--no-wait",
            ],
        )
        options = captured["options"]
        assert options.provider == "openai"
        assert options.model == "gpt-4.1-mini"
        assert options.mode == "batch"
        assert options.resume is False
        assert options.keep_frames is True
        assert options.wait_for_batch is False

    def test_the_chat_flag_selects_the_source(self, monkeypatch):
        from kick_vod_analyser.ingest.chat import NullChatSource

        captured = self._capture(monkeypatch)
        runner.invoke(app, ["analyse", "--url", VOD.url, "--provider", "mock", "--chat", "none"])
        assert isinstance(captured["chat_source"], NullChatSource)

    def test_chat_file_requires_a_path(self, monkeypatch):
        self._capture(monkeypatch)
        result = runner.invoke(
            app, ["analyse", "--url", VOD.url, "--provider", "mock", "--chat", "file"]
        )
        assert result.exit_code != 0

    def test_work_and_out_directories_are_overridable(self, monkeypatch, tmp_path):
        captured = self._capture(monkeypatch)
        runner.invoke(
            app,
            [
                "analyse",
                "--url", VOD.url,
                "--provider", "mock",
                "--work-dir", str(tmp_path / "w"),
                "--out-dir", str(tmp_path / "o"),
            ],
        )
        assert captured["settings"].work_dir == tmp_path / "w"
        assert captured["settings"].out_dir == tmp_path / "o"

    def test_a_completed_run_renders_the_timeline(self, monkeypatch, tmp_path):
        from kick_vod_analyser.models import PrimaryCategory, Segment, Timeline
        from kick_vod_analyser.pipeline import RunReport

        timeline = Timeline(
            vod=VOD,
            segments=[
                Segment(
                    start_seconds=0.0,
                    end_seconds=1800.0,
                    primary_category=PrimaryCategory.GAMING,
                    specific_title_or_context="Valorant",
                    confidence_score=0.91,
                )
            ],
            provider="mock",
            model="mock-v1",
        )
        report = RunReport(
            vod=VOD,
            timeline=timeline,
            outputs={"timeline_json": tmp_path / "timeline.json"},
            cost={
                "requests": 10.0,
                "input_tokens": 1000.0,
                "output_tokens": 100.0,
                "input_cost_usd": 0.001,
                "output_cost_usd": 0.0005,
                "total_cost_usd": 0.0015,
            },
            errors=["one sample failed"],
        )

        class FakePipeline:
            def __init__(self, *a, **k):
                pass

            def run(self, options):
                return report

        monkeypatch.setattr(cli_module, "Pipeline", FakePipeline)
        result = runner.invoke(
            app, ["analyse", "--url", VOD.url, "--provider", "mock", "--out-dir", str(tmp_path)]
        )

        assert result.exit_code == 0
        assert "Gaming: Valorant" in result.stdout
        assert "timeline_json" in result.stdout
        assert "non-fatal" in result.stdout
        assert (tmp_path / VOD.vod_id / "run_report.json").exists()

    def test_dry_run_reports_the_plan_without_a_timeline(self, monkeypatch, tmp_path):
        from kick_vod_analyser.models import SamplePoint
        from kick_vod_analyser.pipeline import RunReport

        report = RunReport(
            vod=VOD,
            sample_points=[SamplePoint(offset_seconds=float(t), trigger="scene") for t in (10, 20)],
            cost={
                "requests": 2.0,
                "input_tokens": 100.0,
                "output_tokens": 10.0,
                "input_cost_usd": 0.0001,
                "output_cost_usd": 0.0,
                "total_cost_usd": 0.0001,
            },
        )

        class FakePipeline:
            def __init__(self, *a, **k):
                pass

            def run(self, options):
                return report

        monkeypatch.setattr(cli_module, "Pipeline", FakePipeline)
        result = runner.invoke(
            app,
            ["analyse", "--url", VOD.url, "--provider", "mock", "--dry-run", "--out-dir", str(tmp_path)],
        )

        assert result.exit_code == 0
        assert "Planned 2 classification points" in result.stdout

    def test_a_run_with_no_timeline_exits_non_zero(self, monkeypatch):
        class FailingPipeline:
            def __init__(self, *a, **k):
                pass

            def run(self, options):
                from kick_vod_analyser.pipeline import RunReport

                return RunReport(errors=["sampling produced no points"])

        monkeypatch.setattr(cli_module, "Pipeline", FailingPipeline)
        result = runner.invoke(app, ["analyse", "--url", VOD.url, "--provider", "mock"])
        assert result.exit_code == 1
        assert "sampling produced no points" in result.stdout
