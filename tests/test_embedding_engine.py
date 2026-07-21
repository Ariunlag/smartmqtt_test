from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.embedding_engine import (
    DEFAULT_CLUSTERING_METHOD,
    KEY_ONLY,
    KMEANS_METHOD,
    MODEL_NAME,
    REPRESENTATIONS,
    VALUE_ONLY,
    ClusteringParameters,
    cluster_all_strategies,
    cluster_embeddings,
    cluster_summary,
    construct_all_representations,
    construct_representation,
    get_or_create_embedding_cache,
    representative_topics,
    threshold_to_eps,
)


def record(
    topic: str,
    measurement: str,
    *,
    dataset: str = "beach_weather",
    source: str = "Station A",
) -> dict[str, str]:
    field = {
        "beach_weather": "station_name",
        "beach_water": "beach_name",
        "open_air": "sensor_name",
        "sgim": "data_stream_id",
    }[dataset]
    return {
        "topic": topic,
        "dataset": dataset,
        "measurement_key": measurement,
        field: source,
        "text": f"{field}: {source} | measurement_key: {measurement}",
    }


def test_threshold_is_converted_to_eps() -> None:
    assert threshold_to_eps(0.80) == pytest.approx(0.20)
    assert threshold_to_eps(0.99) == pytest.approx(0.01)


def test_cluster_count_changes_when_threshold_changes() -> None:
    vectors = np.array([[1.0, 0.0], [0.95, 0.31], [0.0, 1.0]])
    loose = cluster_embeddings(vectors, ClusteringParameters(0.80, 2))
    strict = cluster_embeddings(vectors, ClusteringParameters(0.99, 2))
    assert cluster_summary(loose)["cluster_count"] == 1
    assert cluster_summary(strict)["cluster_count"] == 0


def test_dbscan_noise_is_retained_as_minus_one() -> None:
    vectors = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    labels = cluster_embeddings(vectors, ClusteringParameters(0.95, 2))
    assert labels.tolist()[-1] == -1
    assert cluster_summary(labels)["noise_count"] == 1


def test_representatives_are_nearest_to_normalized_cluster_mean() -> None:
    vectors = np.array([[1.0, 0.0], [0.8, 0.2], [-0.2, 0.98]])
    ranked = representative_topics(vectors, [0, 0, 0], 0, 3)
    assert ranked[0][0] == 1
    assert ranked[0][1] >= ranked[1][1] >= ranked[2][1]


def test_only_requested_number_of_representatives_is_returned() -> None:
    vectors = np.eye(3)
    assert len(representative_topics(vectors, [4, 4, 4], 4, 2)) == 2


def test_all_strategies_keep_identical_record_order() -> None:
    records = [
        record("weather/a/temp", "Temperature"),
        record("water/b/turbidity", "Turbidity", dataset="beach_water"),
        record("air/c/pm", "pm2_5_raw", dataset="open_air"),
    ]
    rendered = construct_all_representations(records)
    assert tuple(rendered) == REPRESENTATIONS
    assert all(len(texts) == len(records) for texts in rendered.values())
    for index, value in enumerate(records):
        assert rendered[VALUE_ONLY][index].endswith(value["measurement_key"])


def test_all_strategies_use_identical_clustering_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[ClusteringParameters] = []

    def fake_cluster(vectors: np.ndarray, parameters: ClusteringParameters) -> np.ndarray:
        calls.append(parameters)
        return np.zeros(len(vectors), dtype=int)

    monkeypatch.setattr("app.embedding_engine.cluster_embeddings", fake_cluster)
    parameters = ClusteringParameters(0.83, 3)
    matrices = {strategy: np.eye(2) for strategy in REPRESENTATIONS}
    cluster_all_strategies(matrices, parameters)
    assert calls == [parameters] * 4


def test_key_only_identical_texts_are_detected() -> None:
    first = record("weather/a/temp", "Temperature", source="Station A")
    second = record("weather/b/wind", "Wind Speed", source="Station B")
    assert construct_representation(first, KEY_ONLY) == construct_representation(second, KEY_ONLY)
    assert len({construct_representation(item, KEY_ONLY) for item in (first, second)}) == 1


def test_embedding_cache_invalidates_when_text_hash_changes(tmp_path: Path) -> None:
    class FakeEncoder:
        calls = 0

        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            self.calls += 1
            return np.array([[len(text), 1.0] for text in texts], dtype=np.float32)

    encoder = FakeEncoder()
    cache_path = tmp_path / "value_only.npz"
    records = [record("weather/a/temp", "Temperature")]
    first = get_or_create_embedding_cache(
        cache_path, records, VALUE_ONLY, MODEL_NAME, lambda: encoder
    )
    second = get_or_create_embedding_cache(
        cache_path, records, VALUE_ONLY, MODEL_NAME, lambda: encoder
    )
    changed = [record("weather/a/temp", "Air Temperature")]
    third = get_or_create_embedding_cache(
        cache_path, changed, VALUE_ONLY, MODEL_NAME, lambda: encoder
    )
    assert not first.reused
    assert second.reused
    assert not third.reused
    assert encoder.calls == 2
    assert first.text_hash != third.text_hash


def test_kmeans_requires_explicit_k_and_is_not_default() -> None:
    assert ClusteringParameters().method == DEFAULT_CLUSTERING_METHOD
    with pytest.raises(ValueError, match="explicit k"):
        cluster_embeddings(
            np.eye(2), ClusteringParameters(method=KMEANS_METHOD, k=None)
        )
