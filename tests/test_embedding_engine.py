from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.embedding_engine import (
    DATASET_MODES,
    DEFAULT_CLUSTERING_METHOD,
    KEY_ONLY,
    KMEANS_METHOD,
    MODEL_NAME,
    NORMALIZED_KEY_VALUE,
    REPRESENTATIONS,
    VALUE_ONLY,
    WEIGHTED_KEY_VALUE,
    ClusteringParameters,
    EmbeddingBundle,
    build_weighted_bundle,
    cluster_by_dataset_mode,
    cluster_embeddings,
    cluster_summary,
    construct_all_representations,
    construct_representation,
    filter_bundle,
    get_or_create_embedding_cache,
    l2_normalize,
    ordered_topic_hash,
    representative_topics,
    threshold_to_eps,
    weighted_embedding_matrix,
)


def record(
    topic: str,
    measurement: str,
    *,
    dataset: str = "beach_weather",
    source: str = "Station A",
    title: str = "PM2_5concmassindividual_raw",
) -> dict[str, str]:
    field = {
        "beach_weather": "station_name",
        "beach_water": "beach_name",
        "open_air": "sensor_name",
        "sgim": "data_stream_id",
    }[dataset]
    value = {
        "topic": topic,
        "dataset": dataset,
        "measurement_key": measurement,
        field: source,
        "measurement_title": title,
        "text": f"{field}: {source} | measurement_key: {measurement}",
    }
    if dataset == "sgim":
        value.update(
            {
                "measurement_description": "Description",
                "measurement_medium": "Air",
                "units": "ug/m3",
                "units_abbreviation": "UG/M3",
                "measurement_period_type": "Instantaneous",
            }
        )
    return value


def bundle(embeddings: np.ndarray, topics: list[str], datasets: list[str]) -> EmbeddingBundle:
    return EmbeddingBundle(
        embeddings=l2_normalize(embeddings),
        topics=np.asarray(topics),
        datasets=np.asarray(datasets),
        measurement_keys=np.asarray(["m"] * len(topics)),
        sources=np.asarray(["source"] * len(topics)),
        texts=np.asarray(["text"] * len(topics)),
        model_name=MODEL_NAME,
        text_hash="hash",
        representation=VALUE_ONLY,
        embedding_dimension=embeddings.shape[1],
        ordered_topic_hash=ordered_topic_hash(topics),
    )


def test_value_only_contains_values_and_excludes_field_names() -> None:
    item = record("weather/a/temp", "Temperature", source="Forest Glen 2")
    text = construct_representation(item, VALUE_ONLY)
    assert "Forest Glen 2" in text and "Temperature" in text
    assert "station_name" not in text


def test_key_only_contains_field_names_and_excludes_values() -> None:
    item = record("weather/a/temp", "Temperature", source="Forest Glen 2")
    text = construct_representation(item, KEY_ONLY)
    assert "station_name" in text and "measurement_key" in text
    assert "Forest Glen 2" not in text and "Temperature" not in text


def test_key_value_contains_field_names_and_values() -> None:
    item = record("weather/a/temp", "Temperature", source="Forest Glen 2")
    text = construct_representation(item, "KEY_VALUE")
    assert "station_name: Forest Glen 2" in text
    assert "measurement_key: Temperature" in text


def test_normalized_key_value_lowercases_text() -> None:
    item = record("weather/a/temp", "PM2_5concmassindividual_raw", source="Forest Glen 2")
    assert "forest glen 2" in construct_representation(item, NORMALIZED_KEY_VALUE)


def test_normalized_key_value_replaces_underscores() -> None:
    item = record("weather/a/temp", "PM2_5concmassindividual_raw")
    text = construct_representation(item, NORMALIZED_KEY_VALUE)
    assert "pm2 5concmassindividual raw" in text
    assert "_" not in text


def test_normalized_key_value_collapses_whitespace() -> None:
    item = record("weather/a/temp", "Temperature", source="Forest   Glen")
    assert "forest glen" in construct_representation(item, NORMALIZED_KEY_VALUE)


def test_normalized_key_value_does_not_expand_abbreviations() -> None:
    item = record("weather/a/temp", "pm2_5_raw")
    assert "pm2 5 raw" in construct_representation(item, NORMALIZED_KEY_VALUE)
    assert "particulate matter" not in construct_representation(item, NORMALIZED_KEY_VALUE)


def test_normalized_key_value_does_not_split_camel_case() -> None:
    item = record("weather/a/temp", "TimeWindowBoundary")
    assert "timewindowboundary" in construct_representation(item, NORMALIZED_KEY_VALUE)
    assert "time window boundary" not in construct_representation(item, NORMALIZED_KEY_VALUE)


def test_dataset_specific_field_order_is_stable() -> None:
    item = record("air/c/pm", "pm", dataset="open_air", source="Forest Glen")
    assert construct_representation(item, "KEY_VALUE") == "sensor_name: Forest Glen | measurement_key: pm"


def test_excluded_identifiers_are_never_embedded() -> None:
    item = record("topic/identifier", "Temperature", dataset="sgim", source="33265")
    item.update({"datasourceid": "DS1", "record_id": "R1", "measurement_id": "M1", "timestamp": "T1", "location": "Chicago"})
    for strategy in (VALUE_ONLY, KEY_ONLY, "KEY_VALUE", NORMALIZED_KEY_VALUE):
        text = construct_representation(item, strategy)
        assert "topic/identifier" not in text
        assert "datasourceid" not in text and "record_id" not in text
        assert "measurement_id" not in text and "timestamp" not in text


def test_weighted_embedding_uses_expected_linear_combination() -> None:
    key = np.array([[1.0, 0.0], [0.0, 1.0]])
    value = np.array([[0.0, 1.0], [1.0, 0.0]])
    actual = weighted_embedding_matrix(key, value, 0.25, 0.75)
    expected = l2_normalize(0.25 * key + 0.75 * value)
    np.testing.assert_allclose(actual, expected)


def test_weighted_embedding_is_l2_normalized() -> None:
    actual = weighted_embedding_matrix(np.ones((2, 3)), np.eye(2, 3), 0.5, 0.5)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1.0)


def test_10_90_weighting_differs_from_90_10() -> None:
    key = np.array([[1.0, 0.0]])
    value = np.array([[0.0, 1.0]])
    assert not np.allclose(
        weighted_embedding_matrix(key, value, 0.10, 0.90),
        weighted_embedding_matrix(key, value, 0.90, 0.10),
    )


def test_50_50_weighting_is_reproducible() -> None:
    key = np.array([[1.0, 2.0]])
    value = np.array([[2.0, 1.0]])
    np.testing.assert_array_equal(
        weighted_embedding_matrix(key, value, 0.5, 0.5),
        weighted_embedding_matrix(key, value, 0.5, 0.5),
    )


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        weighted_embedding_matrix(np.eye(2), np.eye(2), 0.2, 0.2)


def test_weight_changes_do_not_invoke_encoder_again(tmp_path: Path) -> None:
    class FakeEncoder:
        calls = 0

        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            self.calls += 1
            return np.asarray([[len(text), 1.0] for text in texts], dtype=np.float32)

    encoder = FakeEncoder()
    records = [record("weather/a/temp", "Temperature")]
    key = get_or_create_embedding_cache(tmp_path / "key.npz", records, KEY_ONLY, MODEL_NAME, lambda: encoder)
    value = get_or_create_embedding_cache(tmp_path / "value.npz", records, VALUE_ONLY, MODEL_NAME, lambda: encoder)
    first = build_weighted_bundle(key, value, 0.10, 0.90)
    second = build_weighted_bundle(key, value, 0.90, 0.10)
    assert encoder.calls == 2
    assert not np.allclose(first.embeddings, second.embeddings)


def test_dbscan_threshold_converts_to_eps() -> None:
    assert threshold_to_eps(0.80) == pytest.approx(0.20)
    assert threshold_to_eps(0.99) == pytest.approx(0.01)


def test_cluster_count_can_change_as_threshold_changes() -> None:
    vectors = np.array([[1.0, 0.0], [0.95, 0.31], [0.0, 1.0]])
    loose = cluster_embeddings(vectors, ClusteringParameters(0.80, 2))
    strict = cluster_embeddings(vectors, ClusteringParameters(0.99, 2))
    assert cluster_summary(loose)["cluster_count"] == 1
    assert cluster_summary(strict)["cluster_count"] == 0


def test_noise_remains_label_minus_one() -> None:
    labels = cluster_embeddings(np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]), ClusteringParameters(0.95, 2))
    assert labels.tolist()[-1] == -1
    assert cluster_summary(labels)["noise_count"] == 1


def test_kmeans_requires_explicit_k() -> None:
    with pytest.raises(ValueError, match="explicit k"):
        cluster_embeddings(np.eye(2), ClusteringParameters(method=KMEANS_METHOD))


def test_kmeans_k_cannot_exceed_selected_topic_count() -> None:
    with pytest.raises(ValueError, match="between 2"):
        cluster_embeddings(np.eye(2), ClusteringParameters(method=KMEANS_METHOD, k=3))


def test_exact_topic_order_is_consistent_across_representations() -> None:
    records = [record("weather/a/temp", "Temperature"), record("water/b/turbidity", "Turbidity", dataset="beach_water")]
    rendered = construct_all_representations(records)
    assert tuple(rendered) == (VALUE_ONLY, KEY_ONLY, "KEY_VALUE", NORMALIZED_KEY_VALUE)
    assert all(len(values) == len(records) for values in rendered.values())


def test_cache_invalidates_when_model_changes(tmp_path: Path) -> None:
    class FakeEncoder:
        calls = 0

        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            self.calls += 1
            return np.ones((len(texts), 2), dtype=np.float32)

    encoder = FakeEncoder()
    path = tmp_path / "value.npz"
    rows = [record("weather/a/temp", "Temperature")]
    get_or_create_embedding_cache(path, rows, VALUE_ONLY, MODEL_NAME, lambda: encoder)
    get_or_create_embedding_cache(path, rows, VALUE_ONLY, "other/model", lambda: encoder)
    assert encoder.calls == 2


def test_cache_invalidates_when_text_hash_changes(tmp_path: Path) -> None:
    class FakeEncoder:
        calls = 0

        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            self.calls += 1
            return np.ones((len(texts), 2), dtype=np.float32)

    encoder = FakeEncoder()
    path = tmp_path / "value.npz"
    get_or_create_embedding_cache(path, [record("t", "Temperature")], VALUE_ONLY, MODEL_NAME, lambda: encoder)
    get_or_create_embedding_cache(path, [record("t", "Wind")], VALUE_ONLY, MODEL_NAME, lambda: encoder)
    assert encoder.calls == 2


def test_cache_invalidates_when_ordered_topic_hash_changes(tmp_path: Path) -> None:
    class FakeEncoder:
        calls = 0

        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            self.calls += 1
            return np.ones((len(texts), 2), dtype=np.float32)

    encoder = FakeEncoder()
    path = tmp_path / "value.npz"
    first = [record("a", "Temperature"), record("b", "Wind")]
    second = [record("b", "Wind"), record("a", "Temperature")]
    get_or_create_embedding_cache(path, first, VALUE_ONLY, MODEL_NAME, lambda: encoder)
    get_or_create_embedding_cache(path, second, VALUE_ONLY, MODEL_NAME, lambda: encoder)
    assert encoder.calls == 2


def test_dataset_filtering_preserves_stable_topic_order() -> None:
    original = bundle(np.eye(3), ["a", "b", "c"], ["sgim", "open_air", "sgim"])
    filtered = filter_bundle(original, ["sgim"])
    assert filtered.topics.tolist() == ["a", "c"]


def test_combined_mode_clusters_selected_datasets_together() -> None:
    labels = cluster_by_dataset_mode(
        np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]),
        ["sgim", "open_air", "sgim"],
        ["sgim", "open_air"],
        ClusteringParameters(0.95, 2),
        DATASET_MODES[0],
    )
    assert list(labels) == ["combined"]
    assert len(labels["combined"]) == 3


def test_separate_mode_returns_independent_results() -> None:
    labels = cluster_by_dataset_mode(
        np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]),
        ["sgim", "open_air", "sgim"],
        ["sgim", "open_air"],
        ClusteringParameters(0.95, 2),
        DATASET_MODES[1],
    )
    assert list(labels) == ["sgim", "open_air"]
    assert [len(values) for values in labels.values()] == [2, 1]


def test_representatives_are_closest_to_normalized_cluster_mean() -> None:
    vectors = np.array([[1.0, 0.0], [0.8, 0.2], [-0.2, 0.98]])
    ranked = representative_topics(vectors, [0, 0, 0], 0, 3)
    assert ranked[0][0] == 1
    assert ranked[0][1] >= ranked[1][1] >= ranked[2][1]


def test_only_requested_number_of_representatives_is_returned() -> None:
    assert len(representative_topics(np.eye(3), [4, 4, 4], 4, 2)) == 2


def test_weighted_bundle_has_no_weight_specific_cache_identity() -> None:
    rows = [record("weather/a/temp", "Temperature")]
    base = bundle(np.array([[1.0, 0.0]]), ["weather/a/temp"], ["beach_weather"])
    key = EmbeddingBundle(**{**base.__dict__, "representation": KEY_ONLY, "texts": np.asarray(["station_name | measurement_key"])})
    value = EmbeddingBundle(**{**base.__dict__, "representation": VALUE_ONLY, "texts": np.asarray(["Station A | Temperature"])})
    weighted = build_weighted_bundle(key, value, 0.3, 0.7)
    assert weighted.representation == WEIGHTED_KEY_VALUE
    assert not Path("embedding_cache/weighted_key_value.npz").exists()
