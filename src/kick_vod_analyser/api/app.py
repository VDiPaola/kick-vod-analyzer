"""FastAPI application exposing the job queue and a debug UI."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .. import __version__
from ..config import Settings, load_settings
from .jobs import (
    Job,
    JobEvent,
    JobQueue,
    JobRequest,
    JobStore,
    PipelineRunner,
    run_pipeline,
)

UI_PATH = Path(__file__).with_name("ui.html")

OUTPUT_MEDIA_TYPES = {
    "timeline_json": "application/json",
    "chapters_vtt": "text/vtt",
    "segments_csv": "text/csv",
    "summary_md": "text/markdown",
}


class QueueStatus(BaseModel):
    worker_alive: bool
    current_job_id: str | None
    queued: int
    running: int
    succeeded: int
    failed: int
    cancelled: int


class EventsPage(BaseModel):
    events: list[JobEvent]
    cursor: int


class OutputEntry(BaseModel):
    name: str
    path: str
    size_bytes: int
    exists: bool


class HealthStatus(BaseModel):
    ok: bool
    version: str
    worker_alive: bool
    uptime_seconds: float


def create_app(
    settings: Settings | None = None,
    *,
    queue: JobQueue | None = None,
    runner: PipelineRunner = run_pipeline,
) -> FastAPI:
    settings = settings or load_settings()
    queue = queue or JobQueue(settings, runner=runner)
    started_at = time.time()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        queue.start()
        try:
            yield
        finally:
            queue.stop()

    app = FastAPI(
        title="Kick VOD Analyser",
        version=__version__,
        description="Queue Kick VODs for activity analysis and inspect the results.",
        lifespan=lifespan,
    )
    app.state.queue = queue
    app.state.settings = settings

    def get_job(job_id: str) -> Job:
        job = queue.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id} not found")
        return job

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        return UI_PATH.read_text(encoding="utf-8")

    @app.get("/health", response_model=HealthStatus)
    def health() -> HealthStatus:
        return HealthStatus(
            ok=True,
            version=__version__,
            worker_alive=queue.is_running,
            uptime_seconds=time.time() - started_at,
        )

    @app.get("/queue", response_model=QueueStatus)
    def queue_status() -> QueueStatus:
        jobs = queue.store.list(limit=10_000)
        counts = {status: 0 for status in ("queued", "running", "succeeded", "failed", "cancelled")}
        for job in jobs:
            counts[job.status] += 1
        return QueueStatus(
            worker_alive=queue.is_running,
            current_job_id=queue.current_job_id,
            **counts,
        )

    @app.post("/jobs", response_model=Job, status_code=202)
    def create_job(request: JobRequest) -> Job:
        if request.chat == "file" and not request.chat_file:
            raise HTTPException(status_code=422, detail="chat_file is required when chat is 'file'")
        return queue.submit(request)

    @app.get("/jobs", response_model=list[Job])
    def list_jobs(
        status: str | None = Query(default=None, pattern="^(queued|running|succeeded|failed|cancelled)$"),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[Job]:
        return queue.store.list(status=status, limit=limit)

    @app.get("/jobs/{job_id}", response_model=Job)
    def read_job(job_id: str) -> Job:
        return get_job(job_id)

    @app.delete("/jobs/{job_id}", response_model=Job)
    def cancel_job(job_id: str) -> Job:
        job = get_job(job_id)
        if job.status == "running":
            raise HTTPException(status_code=409, detail="a running job cannot be cancelled")
        if job.status != "queued":
            raise HTTPException(status_code=409, detail=f"job is already {job.status}")
        cancelled = queue.cancel(job_id)
        assert cancelled is not None
        return cancelled

    @app.delete("/jobs/{job_id}/record", status_code=204)
    def delete_job_record(job_id: str) -> None:
        job = get_job(job_id)
        if not job.is_terminal:
            raise HTTPException(status_code=409, detail="only finished jobs can be deleted")
        queue.store.delete(job_id)

    @app.post("/jobs/{job_id}/retry", response_model=Job, status_code=202)
    def retry_job(job_id: str) -> Job:
        job = get_job(job_id)
        if not job.is_terminal:
            raise HTTPException(status_code=409, detail="only finished jobs can be retried")
        return queue.submit(job.request)

    @app.get("/jobs/{job_id}/events", response_model=EventsPage)
    def job_events(
        job_id: str,
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> EventsPage:
        get_job(job_id)
        rows = queue.store.events(job_id, after=after, limit=limit)
        cursor = rows[-1][0] if rows else after
        return EventsPage(events=[event for _, event in rows], cursor=cursor)

    @app.get("/jobs/{job_id}/outputs", response_model=list[OutputEntry])
    def job_outputs(job_id: str) -> list[OutputEntry]:
        job = get_job(job_id)
        if job.result is None:
            return []
        entries = []
        for name, raw in job.result.outputs.items():
            path = Path(raw)
            exists = path.is_file()
            entries.append(
                OutputEntry(
                    name=name,
                    path=str(path),
                    size_bytes=path.stat().st_size if exists else 0,
                    exists=exists,
                )
            )
        return entries

    @app.get("/jobs/{job_id}/outputs/{name}")
    def job_output_file(job_id: str, name: str, request: Request):
        job = get_job(job_id)
        if job.result is None or name not in job.result.outputs:
            raise HTTPException(status_code=404, detail=f"output {name} not found")
        path = Path(job.result.outputs[name])
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"output file missing: {path}")
        media_type = OUTPUT_MEDIA_TYPES.get(name, "application/octet-stream")
        if name == "summary_md" or "text/plain" in request.headers.get("accept", ""):
            media_type = "text/plain; charset=utf-8"
        return FileResponse(path, media_type=media_type, filename=path.name)

    @app.get("/logs")
    def logs(limit: int = Query(default=200, ge=1, le=500)) -> list[dict]:
        return queue.logs.tail(limit)

    return app


def build_default_app() -> FastAPI:
    """Entry point for `uvicorn kick_vod_analyser.api.app:build_default_app --factory`."""
    return create_app()


__all__ = ["JobQueue", "JobRequest", "JobStore", "build_default_app", "create_app"]
