"""SQLite persistence so a re-run never repays for work already done."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from .models import Classification, SamplePoint, SampleResult, VodInfo

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS vods (
    vod_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    channel_slug TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    payload TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    vod_id TEXT NOT NULL,
    custom_id TEXT NOT NULL,
    offset_seconds REAL NOT NULL,
    trigger TEXT NOT NULL,
    model TEXT NOT NULL,
    classification TEXT NOT NULL,
    grid_path TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (vod_id, custom_id, model)
);

CREATE INDEX IF NOT EXISTS idx_samples_vod ON samples (vod_id, offset_seconds);

CREATE TABLE IF NOT EXISTS scene_points (
    vod_id TEXT NOT NULL,
    offset_seconds REAL NOT NULL,
    trigger TEXT NOT NULL,
    scene_score REAL,
    PRIMARY KEY (vod_id, offset_seconds)
);
"""


class Store:
    """Thin SQLite wrapper. Every method is safe to call on a fresh database."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def save_vod(self, vod: VodInfo) -> None:
        import time

        self._conn.execute(
            "INSERT OR REPLACE INTO vods VALUES (?, ?, ?, ?, ?, ?)",
            (
                vod.vod_id,
                vod.url,
                vod.channel_slug,
                vod.duration_seconds,
                vod.model_dump_json(),
                time.time(),
            ),
        )
        self._conn.commit()

    def load_vod(self, vod_id: str) -> VodInfo | None:
        row = self._conn.execute(
            "SELECT payload FROM vods WHERE vod_id = ?", (vod_id,)
        ).fetchone()
        return VodInfo.model_validate_json(row["payload"]) if row else None

    def save_scene_points(self, vod_id: str, points: list[SamplePoint]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO scene_points VALUES (?, ?, ?, ?)",
            [(vod_id, p.offset_seconds, p.trigger, p.scene_score) for p in points],
        )
        self._conn.commit()

    def load_scene_points(self, vod_id: str) -> list[SamplePoint]:
        rows = self._conn.execute(
            "SELECT offset_seconds, trigger, scene_score FROM scene_points "
            "WHERE vod_id = ? ORDER BY offset_seconds",
            (vod_id,),
        ).fetchall()
        return [
            SamplePoint(
                offset_seconds=row["offset_seconds"],
                trigger=row["trigger"],
                scene_score=row["scene_score"],
            )
            for row in rows
        ]

    def save_results(self, vod_id: str, model: str, results: list[SampleResult]) -> None:
        import time

        now = time.time()
        self._conn.executemany(
            "INSERT OR REPLACE INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    vod_id,
                    f"t{int(round(r.offset_seconds))}",
                    r.offset_seconds,
                    r.trigger,
                    model,
                    r.classification.model_dump_json(),
                    r.grid_path,
                    now,
                )
                for r in results
            ],
        )
        self._conn.commit()

    def load_results(self, vod_id: str, model: str) -> list[SampleResult]:
        rows = self._conn.execute(
            "SELECT offset_seconds, trigger, classification, grid_path FROM samples "
            "WHERE vod_id = ? AND model = ? ORDER BY offset_seconds",
            (vod_id, model),
        ).fetchall()
        return [
            SampleResult(
                offset_seconds=row["offset_seconds"],
                trigger=row["trigger"],
                classification=Classification.model_validate(json.loads(row["classification"])),
                grid_path=row["grid_path"],
            )
            for row in rows
        ]

    def classified_ids(self, vod_id: str, model: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT custom_id FROM samples WHERE vod_id = ? AND model = ?", (vod_id, model)
        ).fetchall()
        return {row["custom_id"] for row in rows}
