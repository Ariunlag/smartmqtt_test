"""Pure data-source loading and clustering-limit helpers for the explorer."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from app.embedding_engine import (
    DATASET_MODES,
    DEFAULT_CLUSTERING_METHOD,
    KMEANS_METHOD,
    ClusteringParameters,
)


REAL_DATA = "Real datasets"
GENERATED_DATA = "Generated benchmark"
MIXED_DATA = "Real + generated benchmark"
DATA_SOURCE_OPTIONS = (REAL_DATA, GENERATED_DATA, MIXED_DATA)


@dataclass(frozen=True)
class SourceLoadResult:
    records: list[dict[str, Any]]
    signatures: tuple[Any, ...]


@dataclass(frozen=True)
class ClusteringLimits:
    total_topic_count: int
    smallest_dataset_count: int
    maximum_k: int
    maximum_min_samples: int


def load_experiment_source(
    data_source: str,
    *,
    signature_loader: Callable[[], tuple[Any, ...]],
    real_loader: Callable[[tuple[Any, ...]], Sequence[Mapping[str, Any]]],
    synthetic_loader: Callable[[], Sequence[Mapping[str, Any]]],
) -> SourceLoadResult:
    """Load only the inputs required by the selected experiment source."""

    if data_source == GENERATED_DATA:
        return SourceLoadResult([dict(record) for record in synthetic_loader()], ())
    if data_source not in (REAL_DATA, MIXED_DATA):
        raise ValueError(f"Unsupported experiment data source: {data_source}")
    signatures = signature_loader()
    real_records = [dict(record) for record in real_loader(signatures)]
    if data_source == REAL_DATA:
        return SourceLoadResult(real_records, signatures)
    synthetic_records = [dict(record) for record in synthetic_loader()]
    return SourceLoadResult([*real_records, *synthetic_records], signatures)


def clustering_parameter_limits(
    records: Sequence[Mapping[str, Any]],
    selected_datasets: Sequence[str],
    dataset_mode: str,
) -> ClusteringLimits:
    """Calculate valid shared K and min_samples limits for selected datasets."""

    if not selected_datasets:
        raise ValueError("Select at least one dataset before clustering.")
    counts = Counter(str(record["dataset"]) for record in records)
    selected_counts = {dataset: counts.get(dataset, 0) for dataset in selected_datasets}
    too_small = {dataset: count for dataset, count in selected_counts.items() if count < 2}
    if too_small:
        details = ", ".join(f"{dataset} ({count} topic{'s' if count != 1 else ''})" for dataset, count in too_small.items())
        raise ValueError(
            "Each selected dataset must contain at least two topics for clustering. "
            f"Too small: {details}."
        )
    total = sum(selected_counts.values())
    smallest = min(selected_counts.values())
    if dataset_mode == DATASET_MODES[0]:
        maximum = total
    elif dataset_mode == DATASET_MODES[1]:
        maximum = smallest
    else:
        raise ValueError(f"Unsupported dataset mode: {dataset_mode}")
    return ClusteringLimits(total, smallest, maximum, maximum)


def validate_clustering_parameters(
    parameters: ClusteringParameters, limits: ClusteringLimits
) -> None:
    if parameters.method == KMEANS_METHOD:
        if parameters.k is None:
            raise ValueError("K-means requires an explicit K.")
        if not 2 <= parameters.k <= limits.maximum_k:
            raise ValueError(
                f"K must be between 2 and {limits.maximum_k} for the selected dataset mode."
            )
    elif parameters.method == DEFAULT_CLUSTERING_METHOD:
        if not 2 <= parameters.min_samples <= limits.maximum_min_samples:
            raise ValueError(
                "min_samples must be between 2 and "
                f"{limits.maximum_min_samples} for the selected dataset mode."
            )
    else:
        raise ValueError(f"Unsupported clustering method: {parameters.method}")
