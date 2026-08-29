from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from conftest import make_sample
from fastapi.testclient import TestClient

from kick_vod_analyser.api import JobQueue, JobRequest, JobStore, create_app
from kick_vod_analyser.api.jobs import JobResult
from kick_vod_analyser.models import Timeline, VodInfo
from kick_vod_analyser.pipeline import RunReport
from kick_vod_analyser.postprocess.outputs import write_all

VOD = VodInfo(
    vod_id="vod-1",
    url="https://kick.com/teststreamer/videos/vod-1",
    channel_slug="teststreamer",
    channel_id=7,
    title="marathon",
    duration_seconds=3600.0,
)


class FakeRunner:
    """Stands in for the pipeline. Records calls and emits progress."""

    def __init__(self, settings, *, fail_on: set[str] | None = None, block: threading.Event | None = None):
        self.settings = settings
        self.calls: list[JobRequest] = []
        self.fail_on = fail_on or set()
        self.block = block
        self.started = threading.Event()

    def __call__(self, settings, request: JobRequest, progress):
        self.calls.append(request)
        self.started.set()
        progress("resolve", f"resolving {request.url}")
        if self.block is not None:
            self.block.wait(timeout=10)
        if request.url in self.fail_on:
            raise RuntimeError("kick returned 403")

        report = RunReport(vod=VOD)
        report.sample_points = [make_sample(0.0).model_copy()]
        if request.dry_run:
            report.cost = {"requests": 1, "total_cost_usd": 0.0}
            progress("dry-run", "1 samples planned")
            return report

        samples = [make_sample(0.0), make_sample(600.0), make_sample(1200.0)]
        timeline = Timeline(vod=VOD, samples=samples, provider=request.provider, model="mock-v1")
        from kick_vod_analyser.postprocess.smoothing import smooth

        timeline.segments = smooth(samples, VOD.duration_seconds, settings.smoothing)
        report.results = samples
        report.timeline = timeline
        report.grids = 3
        report.outputs = write_all(timeline, settings.vod_out_dir(VOD.vod_id))
        report.cost = {"requests": 3, "total_cost_usd": 0.001}
        progress("done", f"{len(timeline.segments)} segments written")
        return report


@pytest.fixture
def runner(settings):
    return FakeRunner(settings)


@pytest.fixture
def client(settings, runner):
    queue = JobQueue(settings, runner=runner)
    app = create_app(settings, queue=queue)
    with TestClient(app) as client:
        client.queue = queue
        yield client


def submit(client, **overrides):
    body = {"url": "https://kick.com/teststreamer/videos/vod-1", "provider": "mock", **overrides}
    response = client.post("/jobs", json=body)
    assert response.status_code == 202, response.text
    return response.json()


def wait_for(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


class TestHealth:
    def test_health_reports_a_live_worker(self, client):
        body = client.get("/health").json()
        assert body["ok"] is True
        assert body["worker_alive"] is True

    def test_ui_is_served_at_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Kick VOD Analyser" in response.text
        assert "text/html" in response.headers["content-type"]

    def test_openapi_lists_the_job_routes(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        for route in ("/jobs", "/jobs/{job_id}", "/jobs/{job_id}/events", "/queue", "/health"):
            assert route in paths


class TestSubmit:
    def test_a_job_is_accepted_and_runs_to_success(self, client, runner):
        job = submit(client)
        assert job["status"] == "queued"
        assert job["request"]["provider"] == "mock"

        done = wait_for(client, job["job_id"])
        assert done["status"] == "succeeded"
        assert done["vod_id"] == "vod-1"
        assert done["result"]["segments"] >= 1
        assert set(done["result"]["outputs"]) == {"timeline_json", "chapters_vtt", "segments_csv", "summary_md"}
        assert done["started_at"] <= done["finished_at"]
        assert runner.calls[0].url == job["request"]["url"]

    def test_request_options_reach_the_runner(self, client, runner):
        job = submit(
            client,
            mode="batch",
            model="custom-model",
            max_samples=5,
            scene_threshold=0.5,
            heartbeat_seconds=120,
            resume=False,
            keep_frames=True,
            wait_for_batch=False,
        )
        wait_for(client, job["job_id"])
        request = runner.calls[0]
        assert request.mode == "batch"
        assert request.model == "custom-model"
        assert request.max_samples == 5
        assert request.scene_threshold == 0.5
        assert request.heartbeat_seconds == 120
        assert request.resume is False
        assert request.keep_frames is True
        assert request.wait_for_batch is False

    def test_dry_run_succeeds_without_a_timeline(self, client):
        job = submit(client, dry_run=True)
        done = wait_for(client, job["job_id"])
        assert done["status"] == "succeeded"
        assert done["result"]["segments"] == 0
        assert done["result"]["sample_points"] == 1

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"url": ""},
            {"url": "https://kick.com/x/videos/y", "provider": "anthropic"},
            {"url": "https://kick.com/x/videos/y", "mode": "async"},
            {"url": "https://kick.com/x/videos/y", "scene_threshold": 2},
            {"url": "https://kick.com/x/videos/y", "max_samples": -1},
            {"url": "https://kick.com/x/videos/y", "chat": "file"},
        ],
    )
    def test_invalid_bodies_are_rejected(self, client, body):
        assert client.post("/jobs", json=body).status_code == 422

    def test_a_failing_pipeline_marks_the_job_failed(self, settings):
        runner = FakeRunner(settings, fail_on={"https://kick.com/teststreamer/videos/bad"})
        with TestClient(create_app(settings, queue=JobQueue(settings, runner=runner))) as client:
            job = submit(client, url="https://kick.com/teststreamer/videos/bad")
            done = wait_for(client, job["job_id"])
        assert done["status"] == "failed"
        assert "kick returned 403" in done["error"]
        events = [e["stage"] for e in client.get(f"/jobs/{job['job_id']}/events").json()["events"]]
        assert "error" in events

    def test_a_report_without_a_timeline_is_a_failure(self, settings):
        def runner(settings, request, progress):
            report = RunReport(vod=VOD)
            report.errors.append("sampling produced no points")
            return report

        with TestClient(create_app(settings, queue=JobQueue(settings, runner=runner))) as client:
            done = wait_for(client, submit(client)["job_id"])
        assert done["status"] == "failed"
        assert "sampling produced no points" in done["error"]


class TestQueueOrdering:
    def test_jobs_run_one_at_a_time_in_submission_order(self, settings):
        gate = threading.Event()
        runner = FakeRunner(settings, block=gate)
        with TestClient(create_app(settings, queue=JobQueue(settings, runner=runner))) as client:
            first = submit(client, url="https://kick.com/a/videos/1")
            second = submit(client, url="https://kick.com/a/videos/2")
            assert runner.started.wait(timeout=5)
            time.sleep(0.1)

            status = client.get("/queue").json()
            assert status["running"] == 1
            assert status["queued"] == 1
            assert status["current_job_id"] == first["job_id"]
            assert client.get(f"/jobs/{second['job_id']}").json()["status"] == "queued"

            gate.set()
            wait_for(client, first["job_id"])
            wait_for(client, second["job_id"])
        assert [c.url for c in runner.calls] == [
            "https://kick.com/a/videos/1",
            "https://kick.com/a/videos/2",
        ]

    def test_list_filters_by_status_and_orders_newest_first(self, client):
        a = submit(client, url="https://kick.com/a/videos/1")
        b = submit(client, url="https://kick.com/a/videos/2")
        wait_for(client, a["job_id"])
        wait_for(client, b["job_id"])

        listed = client.get("/jobs").json()
        assert [j["job_id"] for j in listed] == [b["job_id"], a["job_id"]]
        assert client.get("/jobs?status=succeeded").json() and not client.get("/jobs?status=failed").json()
        assert client.get("/jobs?status=bogus").status_code == 422


class TestCancelRetryDelete:
    def test_a_queued_job_can_be_cancelled(self, settings):
        gate = threading.Event()
        runner = FakeRunner(settings, block=gate)
        with TestClient(create_app(settings, queue=JobQueue(settings, runner=runner))) as client:
            running = submit(client, url="https://kick.com/a/videos/1")
            queued = submit(client, url="https://kick.com/a/videos/2")
            assert runner.started.wait(timeout=5)

            response = client.delete(f"/jobs/{queued['job_id']}")
            assert response.status_code == 200
            assert response.json()["status"] == "cancelled"

            assert client.delete(f"/jobs/{running['job_id']}").status_code == 409
            assert client.delete(f"/jobs/{queued['job_id']}").status_code == 409

            gate.set()
            wait_for(client, running["job_id"])
        assert len(runner.calls) == 1

    def test_retry_requeues_the_same_request(self, client, runner):
        job = submit(client, max_samples=3)
        wait_for(client, job["job_id"])

        response = client.post(f"/jobs/{job['job_id']}/retry")
        assert response.status_code == 202
        retried = response.json()
        assert retried["job_id"] != job["job_id"]
        assert retried["request"] == job["request"]
        wait_for(client, retried["job_id"])
        assert len(runner.calls) == 2

    def test_retry_refuses_an_unfinished_job(self, settings):
        gate = threading.Event()
        runner = FakeRunner(settings, block=gate)
        with TestClient(create_app(settings, queue=JobQueue(settings, runner=runner))) as client:
            job = submit(client)
            assert runner.started.wait(timeout=5)
            assert client.post(f"/jobs/{job['job_id']}/retry").status_code == 409
            gate.set()
            wait_for(client, job["job_id"])

    def test_delete_record_removes_a_finished_job(self, client):
        job = submit(client)
        wait_for(client, job["job_id"])
        assert client.delete(f"/jobs/{job['job_id']}/record").status_code == 204
        assert client.get(f"/jobs/{job['job_id']}").status_code == 404
        assert client.get(f"/jobs/{job['job_id']}/events").status_code == 404

    def test_unknown_job_ids_are_404(self, client):
        for path in ("/jobs/nope", "/jobs/nope/events", "/jobs/nope/outputs", "/jobs/nope/outputs/timeline_json"):
            assert client.get(path).status_code == 404
        assert client.delete("/jobs/nope").status_code == 404
        assert client.post("/jobs/nope/retry").status_code == 404


class TestEventsAndOutputs:
    def test_events_page_with_a_cursor(self, client):
        job = submit(client)
        wait_for(client, job["job_id"])

        page = client.get(f"/jobs/{job['job_id']}/events").json()
        stages = [e["stage"] for e in page["events"]]
        assert stages[0] == "queue"
        assert "resolve" in stages and stages[-1] == "done"
        assert page["cursor"] > 0

        follow_up = client.get(f"/jobs/{job['job_id']}/events?after={page['cursor']}").json()
        assert follow_up["events"] == []
        assert follow_up["cursor"] == page["cursor"]

    def test_outputs_are_listed_and_downloadable(self, client):
        job = submit(client)
        wait_for(client, job["job_id"])

        outputs = {o["name"]: o for o in client.get(f"/jobs/{job['job_id']}/outputs").json()}
        assert set(outputs) == {"timeline_json", "chapters_vtt", "segments_csv", "summary_md"}
        assert all(o["exists"] and o["size_bytes"] > 0 for o in outputs.values())

        timeline = client.get(f"/jobs/{job['job_id']}/outputs/timeline_json")
        assert timeline.status_code == 200
        assert timeline.json()["vod"]["vod_id"] == "vod-1"

        summary = client.get(f"/jobs/{job['job_id']}/outputs/summary_md")
        assert summary.status_code == 200
        assert summary.headers["content-type"].startswith("text/plain")

        assert client.get(f"/jobs/{job['job_id']}/outputs/nope").status_code == 404

    def test_missing_output_files_are_reported(self, client):
        job = submit(client)
        wait_for(client, job["job_id"])
        for path in client.get(f"/jobs/{job['job_id']}/outputs").json():
            Path(path["path"]).unlink()
        outputs = client.get(f"/jobs/{job['job_id']}/outputs").json()
        assert all(not o["exists"] for o in outputs)
        assert client.get(f"/jobs/{job['job_id']}/outputs/timeline_json").status_code == 404

    def test_outputs_are_empty_before_a_result_exists(self, settings):
        gate = threading.Event()
        runner = FakeRunner(settings, block=gate)
        with TestClient(create_app(settings, queue=JobQueue(settings, runner=runner))) as client:
            job = submit(client)
            assert runner.started.wait(timeout=5)
            assert client.get(f"/jobs/{job['job_id']}/outputs").json() == []
            gate.set()
            wait_for(client, job["job_id"])

    def test_worker_logs_are_exposed(self, client):
        job = submit(client)
        wait_for(client, job["job_id"])
        lines = client.get("/logs").json()
        assert any("resolving" in line["message"] for line in lines)
        assert client.get("/logs?limit=0").status_code == 422


class TestJobStore:
    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "jobs.sqlite"
        store = JobStore(path)
        job = store.create(JobRequest(url="https://kick.com/a/videos/1"))
        store.add_event(job.job_id, "queue", "queued")
        store.update(job.job_id, result=JobResult(vod_id="v", outputs={"x": "y"}))
        store.close()

        reopened = JobStore(path)
        loaded = reopened.get(job.job_id)
        assert loaded is not None
        assert loaded.result.vod_id == "v"
        assert reopened.events(job.job_id)[0][1].message == "queued"
        reopened.close()

    def test_interrupted_jobs_are_failed_on_startup(self, tmp_path, settings):
        path = tmp_path / "jobs.sqlite"
        store = JobStore(path)
        job = store.create(JobRequest(url="https://kick.com/a/videos/1"))
        assert store.transition(job.job_id, "queued", "running", started_at=time.time())
        store.close()

        queue = JobQueue(settings, store=JobStore(path), runner=lambda *a: RunReport())
        queue.start()
        try:
            recovered = queue.store.get(job.job_id)
            assert recovered.status == "failed"
            assert "worker stopped" in recovered.error
        finally:
            queue.stop()

    def test_transition_is_atomic(self, tmp_path):
        store = JobStore(tmp_path / "jobs.sqlite")
        job = store.create(JobRequest(url="https://kick.com/a/videos/1"))
        assert store.transition(job.job_id, "queued", "running")
        assert not store.transition(job.job_id, "queued", "cancelled")
        assert store.get(job.job_id).status == "running"
        store.close()

    def test_queued_ids_are_fifo(self, tmp_path):
        store = JobStore(tmp_path / "jobs.sqlite")
        ids = []
        for n in range(3):
            ids.append(store.create(JobRequest(url=f"https://kick.com/a/videos/{n}")).job_id)
            time.sleep(0.002)
        assert store.queued_ids() == ids
        store.close()


class TestDefaultRunner:
    def test_run_pipeline_wires_settings_and_options(self, settings, monkeypatch):
        from kick_vod_analyser.api import jobs as jobs_module

        captured = {}

        class FakePipeline:
            def __init__(self, run_settings, *, chat_source=None, progress=None):
                captured["settings"] = run_settings
                captured["chat_source"] = chat_source
                progress("resolve", "hello")

            def run(self, options):
                captured["options"] = options
                return RunReport(vod=VOD, sample_points=[make_sample(0.0)])

        monkeypatch.setattr(jobs_module, "Pipeline", FakePipeline)
        events = []
        request = JobRequest(
            url=VOD.url, provider="mock", max_samples=4, scene_threshold=0.2, dry_run=True
        )
        report = jobs_module.run_pipeline(settings, request, lambda s, m: events.append((s, m)))

        assert captured["settings"].sampling.max_samples == 4
        assert captured["settings"].sampling.scene_threshold == 0.2
        assert settings.sampling.max_samples == 0, "caller settings must not be mutated"
        assert captured["options"].provider == "mock"
        assert captured["options"].dry_run is True
        assert events == [("resolve", "hello")]
        assert (settings.vod_out_dir(VOD.vod_id) / "run_report.json").is_file()
        assert report.vod is VOD


class TestCli:
    def test_serve_is_documented(self):
        from typer.testing import CliRunner

        from kick_vod_analyser.cli import app

        result = CliRunner().invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.stdout

    def test_serve_starts_uvicorn_with_the_app(self, monkeypatch, tmp_path):
        import uvicorn
        from typer.testing import CliRunner

        from kick_vod_analyser.cli import app

        captured = {}
        monkeypatch.setattr(uvicorn, "run", lambda application, **kw: captured.update(app=application, **kw))
        result = CliRunner().invoke(
            app, ["serve", "--port", "9999", "--work-dir", str(tmp_path / "w"), "--out-dir", str(tmp_path / "o")]
        )
        assert result.exit_code == 0, result.stdout
        assert captured["port"] == 9999
        assert captured["app"].title == "Kick VOD Analyser"
        assert captured["app"].state.settings.work_dir == tmp_path / "w"
