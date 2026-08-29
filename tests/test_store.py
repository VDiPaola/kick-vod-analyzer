from __future__ import annotations

import pytest

from kick_vod_analyser.models import PrimaryCategory, SamplePoint
from kick_vod_analyser.store import Store
from tests.conftest import make_sample


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "cache.sqlite") as handle:
        yield handle


class TestVodPersistence:
    def test_round_trips_vod_metadata(self, store, vod):
        store.save_vod(vod)
        assert store.load_vod(vod.vod_id) == vod

    def test_an_unknown_vod_loads_as_none(self, store):
        assert store.load_vod("absent") is None

    def test_saving_twice_updates_rather_than_duplicates(self, store, vod):
        store.save_vod(vod)
        store.save_vod(vod.model_copy(update={"title": "new title"}))
        assert store.load_vod(vod.vod_id).title == "new title"


class TestScenePersistence:
    def test_round_trips_scene_points(self, store):
        points = [
            SamplePoint(offset_seconds=100.0, trigger="scene", scene_score=0.5),
            SamplePoint(offset_seconds=200.0, trigger="scene", scene_score=0.7),
        ]
        store.save_scene_points("v1", points)
        loaded = store.load_scene_points("v1")
        assert [p.offset_seconds for p in loaded] == [100.0, 200.0]
        assert loaded[1].scene_score == pytest.approx(0.7)

    def test_results_are_ordered_by_offset(self, store):
        store.save_scene_points(
            "v1",
            [SamplePoint(offset_seconds=float(t), trigger="scene") for t in (900, 100, 500)],
        )
        offsets = [p.offset_seconds for p in store.load_scene_points("v1")]
        assert offsets == sorted(offsets)

    def test_vods_are_isolated_from_each_other(self, store):
        store.save_scene_points("v1", [SamplePoint(offset_seconds=100.0, trigger="scene")])
        store.save_scene_points("v2", [SamplePoint(offset_seconds=200.0, trigger="scene")])
        assert len(store.load_scene_points("v1")) == 1

    def test_re_saving_the_same_offset_does_not_duplicate(self, store):
        point = SamplePoint(offset_seconds=100.0, trigger="scene")
        store.save_scene_points("v1", [point])
        store.save_scene_points("v1", [point])
        assert len(store.load_scene_points("v1")) == 1

    def test_an_unknown_vod_loads_as_empty(self, store):
        assert store.load_scene_points("absent") == []


class TestResultPersistence:
    def test_round_trips_classifications(self, store):
        results = [make_sample(100.0), make_sample(200.0, PrimaryCategory.REACTION, "YouTube")]
        store.save_results("v1", "gemini-x", results)
        loaded = store.load_results("v1", "gemini-x")
        assert [r.offset_seconds for r in loaded] == [100.0, 200.0]
        assert loaded[1].classification.specific_title_or_context == "YouTube"

    def test_results_are_partitioned_by_model(self, store):
        store.save_results("v1", "model-a", [make_sample(100.0)])
        store.save_results("v1", "model-b", [make_sample(200.0)])
        assert len(store.load_results("v1", "model-a")) == 1
        assert store.load_results("v1", "model-a")[0].offset_seconds == 100.0

    def test_re_running_updates_rather_than_duplicates(self, store):
        store.save_results("v1", "m", [make_sample(100.0, PrimaryCategory.GAMING, "Valorant")])
        store.save_results("v1", "m", [make_sample(100.0, PrimaryCategory.REACTION, "YouTube")])
        loaded = store.load_results("v1", "m")
        assert len(loaded) == 1
        assert loaded[0].classification.primary_category is PrimaryCategory.REACTION

    def test_classified_ids_supports_resume(self, store):
        store.save_results("v1", "m", [make_sample(100.4), make_sample(200.6)])
        assert store.classified_ids("v1", "m") == {"t100", "t201"}

    def test_classified_ids_of_an_unknown_run_is_empty(self, store):
        assert store.classified_ids("absent", "m") == set()

    def test_grid_paths_survive_the_round_trip(self, store):
        sample = make_sample(100.0).model_copy(update={"grid_path": "work/grids/t100.jpg"})
        store.save_results("v1", "m", [sample])
        assert store.load_results("v1", "m")[0].grid_path == "work/grids/t100.jpg"


class TestStoreLifecycle:
    def test_creates_missing_parent_directories(self, tmp_path):
        with Store(tmp_path / "deep" / "nested" / "cache.sqlite") as store:
            assert store.path.exists()

    def test_a_reopened_database_keeps_its_data(self, tmp_path, vod):
        path = tmp_path / "cache.sqlite"
        with Store(path) as store:
            store.save_vod(vod)
        with Store(path) as store:
            assert store.load_vod(vod.vod_id) is not None

    def test_the_schema_is_idempotent(self, tmp_path):
        path = tmp_path / "cache.sqlite"
        for _ in range(3):
            Store(path).close()
        assert path.exists()
