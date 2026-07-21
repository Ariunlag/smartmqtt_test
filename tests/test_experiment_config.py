from __future__ import annotations

from pathlib import Path

import pytest

from app.embedding_engine import (
    DATASET_MODES,
    DEFAULT_CLUSTERING_METHOD,
    KMEANS_METHOD,
    ClusteringParameters,
)
from app.experiment_config import (
    GENERATED_DATA,
    MIXED_DATA,
    REAL_DATA,
    clustering_parameter_limits,
    load_experiment_source,
    validate_clustering_parameters,
)
from app.synthetic_benchmark import build_synthetic_records


def records(*counts: tuple[str, int]) -> list[dict[str, str]]:
    return [
        {"topic": f"{dataset}/{index}", "dataset": dataset}
        for dataset, count in counts
        for index in range(count)
    ]


def test_generated_mode_does_not_call_file_signatures() -> None:
    calls = 0

    def signatures() -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        raise AssertionError("signature loader must not run")

    result = load_experiment_source(
        GENERATED_DATA,
        signature_loader=signatures,
        real_loader=lambda _: [],
        synthetic_loader=build_synthetic_records,
    )
    assert calls == 0
    assert len(result.records) == 34
    assert result.signatures == ()


def test_generated_mode_loads_when_real_files_are_missing() -> None:
    result = load_experiment_source(
        GENERATED_DATA,
        signature_loader=lambda: (_ for _ in ()).throw(FileNotFoundError("missing")),
        real_loader=lambda _: (_ for _ in ()).throw(FileNotFoundError("missing")),
        synthetic_loader=build_synthetic_records,
    )
    assert {row["dataset"] for row in result.records} == {"synthetic"}


@pytest.mark.parametrize("mode", [REAL_DATA, MIXED_DATA])
def test_real_and_mixed_modes_validate_real_files(mode: str) -> None:
    with pytest.raises(FileNotFoundError, match="missing real file"):
        load_experiment_source(
            mode,
            signature_loader=lambda: (_ for _ in ()).throw(FileNotFoundError("missing real file")),
            real_loader=lambda _: [],
            synthetic_loader=build_synthetic_records,
        )


def test_mixed_mode_preserves_real_first_synthetic_second_order() -> None:
    real = records(("sgim", 2))
    result = load_experiment_source(
        MIXED_DATA,
        signature_loader=lambda: ("real-signature",),
        real_loader=lambda _: real,
        synthetic_loader=build_synthetic_records,
    )
    assert result.records[:2] == real
    assert result.records[2]["dataset"] == "synthetic"


def test_combined_mode_maximum_equals_total_selected_count() -> None:
    limits = clustering_parameter_limits(records(("a", 5), ("b", 3)), ["a", "b"], DATASET_MODES[0])
    assert limits.maximum_k == 8
    assert limits.maximum_min_samples == 8


def test_separate_mode_maximum_k_equals_smallest_dataset() -> None:
    limits = clustering_parameter_limits(records(("a", 5), ("b", 3)), ["a", "b"], DATASET_MODES[1])
    assert limits.maximum_k == 3


def test_separate_mode_maximum_min_samples_equals_smallest_dataset() -> None:
    limits = clustering_parameter_limits(records(("a", 5), ("b", 3)), ["a", "b"], DATASET_MODES[1])
    assert limits.maximum_min_samples == 3


def test_k_exceeding_smallest_dataset_is_rejected_before_kmeans() -> None:
    limits = clustering_parameter_limits(records(("a", 5), ("b", 3)), ["a", "b"], DATASET_MODES[1])
    with pytest.raises(ValueError, match="between 2 and 3"):
        validate_clustering_parameters(ClusteringParameters(method=KMEANS_METHOD, k=4), limits)


def test_min_samples_exceeding_smallest_dataset_is_rejected_before_dbscan() -> None:
    limits = clustering_parameter_limits(records(("a", 5), ("b", 3)), ["a", "b"], DATASET_MODES[1])
    with pytest.raises(ValueError, match="between 2 and 3"):
        validate_clustering_parameters(
            ClusteringParameters(method=DEFAULT_CLUSTERING_METHOD, min_samples=4), limits
        )


def test_dataset_with_fewer_than_two_topics_has_readable_error() -> None:
    with pytest.raises(ValueError, match="at least two topics"):
        clustering_parameter_limits(records(("a", 5), ("b", 1)), ["a", "b"], DATASET_MODES[1])


def test_launcher_disables_streamlit_file_watcher() -> None:
    root = Path(__file__).resolve().parents[1]
    assert "--server.fileWatcherType none" in (root / "scripts" / "run_embedding_ui.ps1").read_text()


def test_readme_documents_watcher_and_label_distinction() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    assert "--server.fileWatcherType none" in readme
    assert "The real-data experiment has no" in readme
    assert "expected groups used only for post-clustering" in readme
    assert "never embedded or passed to clustering" in readme


def test_synthetic_record_count_remains_34() -> None:
    assert len(build_synthetic_records()) == 34
