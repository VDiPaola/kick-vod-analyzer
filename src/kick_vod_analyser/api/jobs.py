"""Persistent job queue with a single background worker."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..config import Settings
from ..ingest.chat import build_chat_source
from ..pipeline import Pipeline, RunOptions, RunReport, write_run_report

log = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
TERMINAL: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    request TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    vod_id TEXT,
    stage TEXT,
    result TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    at REAL NOT NULL,
    stage TEXT NOT NULL,
    message TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events (job_id, id);
"""


class JobRequest(BaseModel):
    """Body of POST /jobs. Mirrors the analyse command options."""

    url: str = Field(min_length=1)
    provider: Literal["gemini", "openai", "mock"] = "gemini"
    model: str | None = None
    mode: Literal["sync", "batch"] = "sync"
    chat: Literal["none", "file", "kick"] = "none"
    chat_file: str | None = None
    scene_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    heartbeat_seconds: float | None = Field(default=None, gt=0)
    max_samples: int | None = Field(default=None, ge=0)
    resume: bool = True
    keep_frames: bool = False
    dry_run: bool = False
    wait_for_batch: bool = True


class JobEvent(BaseModel):
    at: float
    stage: str
    message: str


class JobResult(BaseModel):
    vod_id: str | None = None
    channel_slug: str | None = None
    title: str | None = None
    duration_seconds: float | None = None
    sample_points: int = 0
    grids: int = 0
    results: int = 0
    segments: int = 0
    batch_job_id: str | None = None
    cost: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_report(cls, report: RunReport) -> JobResult:
        vod = report.vod
        return cls(
            vod_id=vod.vod_id if vod else None,
            channel_slug=vod.channel_slug if vod else None,
            title=vod.title if vod else None,
            duration_seconds=vod.duration_seconds if vod else None,
            sample_points=len(report.sample_points),
            grids=report.grids,
            results=len(report.results),
            segments=len(report.timeline.segments) if report.timeline else 0,
            batch_job_id=report.batch_job_id,
            cost=dict(report.cost),
            errors=list(report.errors),
            outputs={name: str(path) for name, path in report.outputs.items()},
        )


class Job(BaseModel):
    job_id: str
    status: JobStatus
    request: JobRequest
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    vod_id: str | None = None
    stage: str | None = None
    result: JobResult | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL


class JobStore:
    """SQLite-backed job table. Safe for use from multiple threads."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create(self, request: JobRequest) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex,
            status="queued",
            request=request,
            created_at=time.time(),
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (job_id, status, request, created_at) VALUES (?, ?, ?, ?)",
                (job.job_id, job.status, request.model_dump_json(), job.created_at),
            )
            self._conn.commit()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list(self, *, status: str | None = None, limit: int = 100) -> list[Job]:
        query = "SELECT * FROM jobs"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(query, (*params, limit)).fetchall()
        return [self._row_to_job(row) for row in rows]

    def queued_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT job_id FROM jobs WHERE status = 'queued' ORDER BY created_at"
            ).fetchall()
        return [row["job_id"] for row in rows]

    def update(self, job_id: str, **fields: object) -> None:
        if not fields:
            return
        if "result" in fields and isinstance(fields["result"], JobResult):
            fields["result"] = fields["result"].model_dump_json()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {assignments} WHERE job_id = ?",
                (*fields.values(), job_id),
            )
            self._conn.commit()

    def transition(self, job_id: str, from_status: str, to_status: str, **fields: object) -> bool:
        """Atomically move a job between statuses. Returns False when the job moved already."""
        if "result" in fields and isinstance(fields["result"], JobResult):
            fields["result"] = fields["result"].model_dump_json()
        assignments = ", ".join(["status = ?", *(f"{key} = ?" for key in fields)])
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE jobs SET {assignments} WHERE job_id = ? AND status = ?",
                (to_status, *fields.values(), job_id, from_status),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def delete(self, job_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            self._conn.execute("DELETE FROM job_events WHERE job_id = ?", (job_id,))
            self._conn.commit()
            return cursor.rowcount == 1

    def add_event(self, job_id: str, stage: str, message: str) -> JobEvent:
        event = JobEvent(at=time.time(), stage=stage, message=message)
        with self._lock:
            self._conn.execute(
                "INSERT INTO job_events (job_id, at, stage, message) VALUES (?, ?, ?, ?)",
                (job_id, event.at, stage, message),
            )
            self._conn.commit()
        return event

    def events(self, job_id: str, *, after: int = 0, limit: int = 1000) -> list[tuple[int, JobEvent]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, at, stage, message FROM job_events "
                "WHERE job_id = ? AND id > ? ORDER BY id LIMIT ?",
                (job_id, after, limit),
            ).fetchall()
        return [
            (row["id"], JobEvent(at=row["at"], stage=row["stage"], message=row["message"]))
            for row in rows
        ]

    def recover_interrupted(self) -> int:
        """Mark jobs left running by a previous process as failed."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE jobs SET status = 'failed', finished_at = ?, "
                "error = 'worker stopped before the job finished' WHERE status = 'running'",
                (time.time(),),
            )
            self._conn.commit()
            return cursor.rowcount

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            job_id=row["job_id"],
            status=row["status"],
            request=JobRequest.model_validate_json(row["request"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            vod_id=row["vod_id"],
            stage=row["stage"],
            result=JobResult.model_validate_json(row["result"]) if row["result"] else None,
            error=row["error"],
        )


PipelineRunner = Callable[[Settings, JobRequest, Callable[[str, str], None]], RunReport]


def run_pipeline(settings: Settings, request: JobRequest, progress: Callable[[str, str], None]) -> RunReport:
    """Default runner: builds a Pipeline for one request and executes it."""
    run_settings = settings.model_copy(deep=True)
    if request.scene_threshold is not None:
        run_settings.sampling.scene_threshold = request.scene_threshold
    if request.heartbeat_seconds is not None:
        run_settings.sampling.heartbeat_seconds = request.heartbeat_seconds
    if request.max_samples is not None:
        run_settings.sampling.max_samples = request.max_samples

    chat_source = build_chat_source(
        request.chat,
        chat_file=Path(request.chat_file) if request.chat_file else None,
        timeout=run_settings.http_timeout,
        auth_token=run_settings.kick_auth_token,
        workers=run_settings.kick_chat_workers,
    )
    pipeline = Pipeline(run_settings, chat_source=chat_source, progress=progress)
    report = pipeline.run(
        RunOptions(
            url=request.url,
            provider=request.provider,
            model=request.model,
            mode=request.mode,
            chat_source_kind=request.chat,
            chat_file=Path(request.chat_file) if request.chat_file else None,
            resume=request.resume,
            keep_frames=request.keep_frames,
            dry_run=request.dry_run,
            wait_for_batch=request.wait_for_batch,
        )
    )
    if report.vod is not None:
        write_run_report(report, run_settings.vod_out_dir(report.vod.vod_id) / "run_report.json")
    return report


@dataclass
class LogBuffer:
    """Ring buffer of recent log records for the debug UI."""

    capacity: int = 500
    records: deque = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, record: dict) -> None:
        with self.lock:
            self.records.append(record)
            while len(self.records) > self.capacity:
                self.records.popleft()

    def tail(self, limit: int = 200) -> list[dict]:
        with self.lock:
            return list(self.records)[-limit:]


class BufferHandler(logging.Handler):
    def __init__(self, buffer: LogBuffer) -> None:
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(
            {
                "at": record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


class JobQueue:
    """Serial worker over the JobStore. One pipeline runs at a time."""

    def __init__(
        self,
        settings: Settings,
        *,
        store: JobStore | None = None,
        runner: PipelineRunner = run_pipeline,
    ) -> None:
        self.settings = settings
        self.store = store or JobStore(settings.work_dir / "jobs.sqlite")
        self.runner = runner
        self.logs = LogBuffer()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_id: str | None = None
        self._handler = BufferHandler(self.logs)

    @property
    def current_job_id(self) -> str | None:
        return self._current_id

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        recovered = self.store.recover_interrupted()
        if recovered:
            log.warning("marked %d interrupted job(s) as failed", recovered)
        package_log = logging.getLogger("kick_vod_analyser")
        if package_log.level == logging.NOTSET or package_log.level > logging.INFO:
            package_log.setLevel(logging.INFO)
        package_log.addHandler(self._handler)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="kva-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
        logging.getLogger("kick_vod_analyser").removeHandler(self._handler)

    def submit(self, request: JobRequest) -> Job:
        job = self.store.create(request)
        self.store.add_event(job.job_id, "queue", f"queued {request.url}")
        self._wake.set()
        return job

    def cancel(self, job_id: str) -> Job | None:
        moved = self.store.transition(
            job_id, "queued", "cancelled", finished_at=time.time(), error="cancelled by request"
        )
        if moved:
            self.store.add_event(job_id, "queue", "cancelled")
        return self.store.get(job_id)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until the queue drains. Test helper."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._current_id is None and not self.store.queued_ids():
                return True
            time.sleep(0.02)
        return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            job_id = self._next_job_id()
            if job_id is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            self._execute(job_id)

    def _next_job_id(self) -> str | None:
        ids = self.store.queued_ids()
        return ids[0] if ids else None

    def _execute(self, job_id: str) -> None:
        if not self.store.transition(job_id, "queued", "running", started_at=time.time()):
            return
        self._current_id = job_id
        job = self.store.get(job_id)
        assert job is not None

        def progress(stage: str, message: str) -> None:
            self.store.add_event(job_id, stage, message)
            self.store.update(job_id, stage=stage)
            log.info("[%s] %s", stage, message)

        try:
            report = self.runner(self.settings, job.request, progress)
        except Exception as exc:
            log.exception("job %s failed", job_id)
            self.store.add_event(job_id, "error", f"{type(exc).__name__}: {exc}")
            self.store.transition(
                job_id,
                "running",
                "failed",
                finished_at=time.time(),
                error=traceback.format_exc(),
            )
        else:
            result = JobResult.from_report(report)
            succeeded = report.timeline is not None or (
                job.request.dry_run and bool(report.sample_points)
            )
            status = "succeeded" if succeeded else "failed"
            self.store.transition(
                job_id,
                "running",
                status,
                finished_at=time.time(),
                vod_id=result.vod_id,
                result=result,
                error=None if succeeded else "; ".join(report.errors) or "no timeline produced",
            )
        finally:
            self._current_id = None
