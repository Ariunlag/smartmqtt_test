from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.embedding_engine import (
    DATASET_FIELD_ORDER,
    DATASET_MODES,
    KEY_ONLY,
    MODEL_NAME,
    NORMALIZED_KEY_VALUE,
    REPRESENTATIONS,
    VALUE_ONLY,
    build_weighted_bundle,
    cluster_all_strategies,
    controlled_benchmark_metrics,
    controlled_cluster_purity,
    cosine_similarity_matrix,
    construct_representation,
    get_or_create_embedding_cache,
    l2_normalize,
    pca_coordinates,
    safe_model_name,
)
from app.synthetic_benchmark import build_synthetic_records, expected_evaluation_label, synthetic_group_counts


def test_build_synthetic_records_returns_exactly_34_records() -> None:
    assert len(build_synthetic_records()) == 34


def test_repeated_calls_are_identical() -> None:
    assert build_synthetic_records() == build_synthetic_records()


def test_every_synthetic_topic_is_unique() -> None:
    rows = build_synthetic_records()
    assert len({row["topic"] for row in rows}) == 34


def test_six_non_outlier_groups_have_five_records() -> None:
    counts = synthetic_group_counts()
    groups = {group: count for group, count in counts.items() if group != "outlier"}
    assert len(groups) == 6
    assert set(groups.values()) == {5}


def test_exactly_four_outliers_exist() -> None:
    assert synthetic_group_counts()["outlier"] == 4


def test_every_record_has_required_fields() -> None:
    required = {
        "topic", "dataset", "source_name", "measurement_key",
        "measurement_description", "unit", "expected_group", "variant_type",
    }
    assert all(required.issubset(row) for row in build_synthetic_records())


def test_every_record_uses_synthetic_dataset() -> None:
    assert {row["dataset"] for row in build_synthetic_records()} == {"synthetic"}


def test_each_semantic_group_has_all_five_variant_types() -> None:
    rows = build_synthetic_records()
    groups = {row["expected_group"] for row in rows if row["expected_group"] != "outlier"}
    for group in groups:
        variants = {row["variant_type"] for row in rows if row["expected_group"] == group}
        assert variants == {"canonical", "synonym", "formatting_variant", "descriptive_variant", "cross_source_variant"}


def test_synthetic_embedding_field_order_is_stable() -> None:
    assert DATASET_FIELD_ORDER["synthetic"] == (
        "source_name", "measurement_key", "measurement_description", "unit"
    )
    row = build_synthetic_records()[0]
    assert construct_representation(row, "KEY_VALUE").startswith("source_name:")
    assert construct_representation(row, "KEY_VALUE").split(" | ")[1].startswith("measurement_key:")


@pytest.mark.parametrize("field", ["expected_group", "variant_type", "topic", "dataset"])
def test_display_only_fields_are_not_embedded(field: str) -> None:
    rows = build_synthetic_records()
    for row in rows:
        for strategy in (VALUE_ONLY, KEY_ONLY, "KEY_VALUE", NORMALIZED_KEY_VALUE):
            text = construct_representation(row, strategy)
            assert field not in text


def test_key_only_is_identical_for_complete_synthetic_schema() -> None:
    rows = build_synthetic_records()
    texts = {construct_representation(row, KEY_ONLY) for row in rows}
    assert texts == {"source_name | measurement_key | measurement_description | unit"}


def test_value_only_differs_between_unrelated_groups() -> None:
    rows = build_synthetic_records()
    air = next(row for row in rows if row["expected_group"] == "air_temperature")
    door = next(row for row in rows if row["measurement_key"] == "door_open_state")
    assert construct_representation(air, VALUE_ONLY) != construct_representation(door, VALUE_ONLY)


def test_normalized_text_lowercases_underscores_and_whitespace() -> None:
    row = next(row for row in build_synthetic_records() if row["variant_type"] == "formatting_variant")
    normalized = construct_representation(row, NORMALIZED_KEY_VALUE)
    assert normalized == normalized.lower()
    assert "_" not in normalized
    assert "  " not in normalized


def test_normalization_does_not_map_synonyms() -> None:
    rows = [row for row in build_synthetic_records() if row["expected_group"] == "air_temperature"]
    canonical = next(row for row in rows if row["variant_type"] == "canonical")
    synonym = next(row for row in rows if row["variant_type"] == "synonym")
    assert construct_representation(canonical, NORMALIZED_KEY_VALUE) != construct_representation(synonym, NORMALIZED_KEY_VALUE)


def test_synthetic_only_mode_returns_only_generated_records() -> None:
    rows = build_synthetic_records()
    assert len(rows) == 34 and {row["dataset"] for row in rows} == {"synthetic"}


def test_real_only_mode_preserves_current_real_record_count() -> None:
    from app.embedding_engine import load_jsonl_records

    root = Path(__file__).resolve().parents[1]
    paths = [root / "data" / "processed" / f"0{i}_{name}_topic_texts.jsonl" for i, name in ((1, "sgim"), (2, "beach_weather"), (3, "beach_water"), (4, "open_air"))]
    assert len(load_jsonl_records(paths)) == 2969


def test_mixed_mode_appends_synthetic_records_after_real_records() -> None:
    from app.embedding_engine import load_jsonl_records

    root = Path(__file__).resolve().parents[1]
    paths = [root / "data" / "processed" / f"0{i}_{name}_topic_texts.jsonl" for i, name in ((1, "sgim"), (2, "beach_weather"), (3, "beach_water"), (4, "open_air"))]
    real = load_jsonl_records(paths)
    mixed = [*real, *build_synthetic_records()]
    assert len(mixed) == 3003
    assert [row["topic"] for row in mixed[:3]] == [row["topic"] for row in real[:3]]
    assert mixed[2969]["dataset"] == "synthetic"


def test_cache_identity_distinguishes_source_modes() -> None:
    model = safe_model_name(MODEL_NAME)
    paths = [Path("embedding_cache") / model / source / "value_only.npz" for source in ("real", "synthetic", "mixed")]
    assert len({str(path) for path in paths}) == 3


def test_synthetic_cache_does_not_overwrite_real_cache(tmp_path: Path) -> None:
    class Encoder:
        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            return np.ones((len(texts), 2), dtype=np.float32)

    row = build_synthetic_records()[0]
    real_path = tmp_path / "real" / "value_only.npz"
    synthetic_path = tmp_path / "synthetic" / "value_only.npz"
    get_or_create_embedding_cache(real_path, [row], VALUE_ONLY, MODEL_NAME, Encoder)
    get_or_create_embedding_cache(synthetic_path, [row], VALUE_ONLY, MODEL_NAME, Encoder)
    assert real_path != synthetic_path and real_path.exists() and synthetic_path.exists()


def test_outliers_receive_distinct_expected_evaluation_labels() -> None:
    rows = [row for row in build_synthetic_records() if row["expected_group"] == "outlier"]
    labels = [expected_evaluation_label(row) for row in rows]
    assert len(set(labels)) == 4
    assert all(label.startswith("outlier_") for label in labels)


def test_controlled_metrics_are_in_valid_ranges() -> None:
    labels = np.array([0, 0, 1, -1])
    expected = ["air_temperature", "air_temperature", "water_temperature", "outlier_battery_voltage"]
    metrics = controlled_benchmark_metrics(labels, expected, ["air_temperature", "air_temperature", "water_temperature", "outlier"])
    assert 0.0 <= metrics["adjusted_rand_index"] <= 1.0
    assert 0.0 <= metrics["normalized_mutual_information"] <= 1.0
    assert 0.0 <= metrics["cluster_purity"] <= 1.0
    assert metrics["noise_count"] == 1


def test_cluster_purity_handles_noise_explicitly() -> None:
    assert controlled_cluster_purity(np.array([0, 0, -1]), ["a", "a", "b"]) == pytest.approx(1.0)


def test_cluster_purity_handles_empty_non_noise_set() -> None:
    assert controlled_cluster_purity(np.array([-1, -1]), ["a", "b"]) == 0.0


def test_pca_returns_two_coordinates_per_record() -> None:
    assert pca_coordinates(np.eye(4, 3)).shape == (4, 2)


def test_clustering_uses_full_dimensional_vectors() -> None:
    vectors = np.eye(4, 3)
    assert vectors.shape[1] == 3
    assert pca_coordinates(vectors).shape[1] == 2
    assert cluster_all_strategies({VALUE_ONLY: vectors}, __import__("app.embedding_engine", fromlist=["ClusteringParameters"]).ClusteringParameters(0.8, 2))[VALUE_ONLY].shape == (4,)


def test_similarity_matrix_is_square_symmetric_and_unit_diagonal() -> None:
    matrix = cosine_similarity_matrix(l2_normalize(np.eye(3)))
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 1.0)


def test_expected_labels_are_not_required_by_clustering_inputs() -> None:
    vectors = np.eye(3, 2)
    labels = cluster_all_strategies({VALUE_ONLY: vectors}, __import__("app.embedding_engine", fromlist=["ClusteringParameters"]).ClusteringParameters(0.8, 2))
    assert labels[VALUE_ONLY].shape == (3,)
