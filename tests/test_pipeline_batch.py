"""Batch orchestration in the pipeline, driven by a scripted fake classifier."""

from __future__ import annotations


import pytest

from kick_vod_analyser.classify.base import Classifier, ClassificationResponse
from kick_vod_analyser.classify import factory as factory_module
from kick_vod_analyser.ingest.chat import NullChatSource
from kick_vod_analyser.models import Classification, PrimaryCategory, VodInfo
from kick_vod_analyser.pipeline import Pipeline, RunOptions
from tests.conftest import requires_ffmpeg


class ScriptedBatchClassifier(Classifier):
    """Returns a scripted sequence of poll states, then a fixed result set."""

    provider = "scripted"

    def __init__(self, states, *, results=None, model="scripted-v1"):
        super().__init__(model)
        self.states = list(states)
        self.results = results
        self.submitted = None
        self.fetched = False
        self.polls = 0

    def classify(self, requests):
        raise AssertionError("sync path must not be used in batch mode")

    def supports_batch(self):
        return True

    def submit_batch(self, requests, work_dir):
        self.submitted = list(requests)
        return "job-1"

    def poll_batch(self, job_id):
        self.polls += 1
        return self.states.pop(0) if self.states else self.states_final

    @property
    def states_final(self):
        return "JOB_STATE_SUCCEEDED"

    def fetch_batch(self, job_id, work_dir):
        self.fetched = True
        if self.results is not None:
            return self.results
        return [
            ClassificationResponse(
                custom_id=request.custom_id,
                classification=Classification(
                    primary_category=PrimaryCategory.GAMING,
                    specific_title_or_context="Valorant",
                    sub_activity="In-Game Match",
                    confidence_score=0.9,
                    visual_evidence="scripted",
                ),
            )
            for request in self.submitted
        ]


@pytest.fixture
def local_vod(synthetic_video):
    return VodInfo(
        vod_id="synthetic-batch",
        url=str(synthetic_video),
        channel_slug="teststreamer",
        duration_seconds=90.0,
        playback_url=str(synthetic_video),
    )


@pytest.fixture
def batch_pipeline(settings, local_vod, monkeypatch):
    settings.sampling.heartbeat_seconds = 25.0
    settings.batch_poll_seconds = 0.0

    def build(classifier):
        monkeypatch.setattr(factory_module, "build_classifier", lambda *a, **k: classifier)
        import kick_vod_analyser.pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "build_classifier", lambda *a, **k: classifier)
        pipeline = Pipeline(settings, chat_source=NullChatSource(), progress=lambda *a: None)
        monkeypatch.setattr(pipeline, "_resolve", lambda url: local_vod)
        return pipeline

    return build


def batch_options(**overrides):
    defaults = dict(url="local", provider="scripted", mode="batch", chat_source_kind="none")
    defaults.update(overrides)
    return RunOptions(**defaults)


@requires_ffmpeg
class TestBatchOrchestration:
    def test_submits_polls_and_fetches(self, batch_pipeline):
        classifier = ScriptedBatchClassifier(["JOB_STATE_PENDING", "JOB_STATE_SUCCEEDED"])
        report = batch_pipeline(classifier).run(batch_options())

        assert report.batch_job_id == "job-1"
        assert classifier.polls == 2
        assert classifier.fetched
        assert report.timeline is not None
        assert report.results

    def test_polling_stops_at_the_first_terminal_state(self, batch_pipeline):
        classifier = ScriptedBatchClassifier(
            ["JOB_STATE_RUNNING", "JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED", "JOB_STATE_RUNNING"]
        )
        batch_pipeline(classifier).run(batch_options())
        assert classifier.polls == 3

    def test_no_wait_submits_without_polling(self, batch_pipeline):
        classifier = ScriptedBatchClassifier(["JOB_STATE_PENDING"])
        report = batch_pipeline(classifier).run(batch_options(wait_for_batch=False))

        assert report.batch_job_id == "job-1"
        assert classifier.polls == 0
        assert not classifier.fetched
        assert report.results == []
        assert any("not awaited" in error for error in report.errors)

    def test_a_failed_job_is_reported_and_not_fetched(self, batch_pipeline):
        classifier = ScriptedBatchClassifier(["JOB_STATE_FAILED"])
        report = batch_pipeline(classifier).run(batch_options())

        assert not classifier.fetched
        assert any("JOB_STATE_FAILED" in error for error in report.errors)

    @pytest.mark.parametrize("state", ["JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED", "failed"])
    def test_every_failure_state_is_handled(self, batch_pipeline, state):
        classifier = ScriptedBatchClassifier([state])
        report = batch_pipeline(classifier).run(batch_options())
        assert not classifier.fetched
        assert report.errors

    def test_openai_terminal_states_are_recognised(self, batch_pipeline):
        classifier = ScriptedBatchClassifier(["in_progress", "completed"])
        report = batch_pipeline(classifier).run(batch_options())
        assert classifier.fetched
        assert report.results

    def test_a_deadline_breach_is_reported(self, batch_pipeline):
        classifier = ScriptedBatchClassifier(["JOB_STATE_RUNNING"] * 50)
        report = batch_pipeline(classifier).run(
            batch_options(max_batch_wait_seconds=0.0)
        )

        assert not classifier.fetched
        assert any("deadline" in error for error in report.errors)

    def test_partial_success_still_builds_a_timeline(self, batch_pipeline):
        classifier = ScriptedBatchClassifier(["JOB_STATE_PARTIALLY_SUCCEEDED"])
        report = batch_pipeline(classifier).run(batch_options())
        assert classifier.fetched
        assert report.timeline is not None

    def test_unmatched_response_ids_are_reported_not_fatal(self, batch_pipeline):
        classifier = ScriptedBatchClassifier(
            ["JOB_STATE_SUCCEEDED"],
            results=[ClassificationResponse(custom_id="t99999", classification=None, error="lost")],
        )
        report = batch_pipeline(classifier).run(batch_options())

        assert report.timeline is not None
        assert any("no matching grid" in error for error in report.errors)

    def test_a_response_without_a_classification_is_reported(self, batch_pipeline):
        pipeline = batch_pipeline(ScriptedBatchClassifier(["JOB_STATE_SUCCEEDED"]))
        report = pipeline.run(batch_options())
        first_id = report.results[0].offset_seconds

        classifier = ScriptedBatchClassifier(
            ["JOB_STATE_SUCCEEDED"],
            results=[
                ClassificationResponse(
                    custom_id=f"t{int(round(first_id))}", classification=None, error="parse failed"
                )
            ],
        )
        report = batch_pipeline(classifier).run(batch_options(resume=False))
        assert any("parse failed" in error for error in report.errors)

    def test_batch_results_are_cached_for_resume(self, batch_pipeline):
        first = ScriptedBatchClassifier(["JOB_STATE_SUCCEEDED"])
        report_one = batch_pipeline(first).run(batch_options())

        second = ScriptedBatchClassifier(["JOB_STATE_SUCCEEDED"])
        report_two = batch_pipeline(second).run(batch_options())

        assert second.submitted is None, "a fully cached run must not resubmit"
        assert len(report_two.results) == len(report_one.results)

    def test_a_fully_cached_run_reports_no_further_spend(self, batch_pipeline):
        batch_pipeline(ScriptedBatchClassifier(["JOB_STATE_SUCCEEDED"])).run(batch_options())
        report = batch_pipeline(ScriptedBatchClassifier(["JOB_STATE_SUCCEEDED"])).run(
            batch_options()
        )

        assert report.cost["requests"] == 0
        assert report.cost["total_cost_usd"] == 0.0
        assert report.timeline is not None

    def test_the_cost_report_uses_the_batch_discount(self, batch_pipeline):
        classifier = ScriptedBatchClassifier(["JOB_STATE_SUCCEEDED"], model="gemini-2.5-flash-lite")
        report = batch_pipeline(classifier).run(batch_options())

        from kick_vod_analyser.classify.pricing import estimate_cost

        sync = estimate_cost(len(report.results), "gemini-2.5-flash-lite", batch=False)
        assert report.cost["total_cost_usd"] < sync["total_cost_usd"]

    def test_a_provider_without_batch_support_falls_back_to_sync(self, batch_pipeline):
        from kick_vod_analyser.classify.factory import MockClassifier

        classifier = MockClassifier()
        report = batch_pipeline(classifier).run(batch_options())

        assert report.batch_job_id is None
        assert report.results
