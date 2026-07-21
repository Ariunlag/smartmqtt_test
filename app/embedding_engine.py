"""Deterministic metadata representations, embedding caches, and clustering."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


MODEL_OPTIONS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
)
MODEL_NAME = MODEL_OPTIONS[0]
VALUE_ONLY = "VALUE_ONLY"
KEY_ONLY = "KEY_ONLY"
KEY_VALUE = "KEY_VALUE"
NORMALIZED_KEY_VALUE = "NORMALIZED_KEY_VALUE"
WEIGHTED_KEY_VALUE = "WEIGHTED_KEY_VALUE"
BASE_REPRESENTATIONS = (VALUE_ONLY, KEY_ONLY, KEY_VALUE, NORMALIZED_KEY_VALUE)
REPRESENTATIONS = BASE_REPRESENTATIONS + (WEIGHTED_KEY_VALUE,)
REPRESENTATION_LABELS = {
    VALUE_ONLY: "Value only",
    KEY_ONLY: "Key only",
    KEY_VALUE: "Key:value",
    NORMALIZED_KEY_VALUE: "Normalized key:value",
    WEIGHTED_KEY_VALUE: "Weighted key:value",
}
DATASET_FIELD_ORDER = {
    "sgim": (
        "measurement_key",
        "measurement_title",
        "measurement_description",
        "measurement_medium",
        "units",
        "units_abbreviation",
        "measurement_period_type",
    ),
    "beach_weather": ("station_name", "measurement_key"),
    "beach_water": ("beach_name", "measurement_key"),
    "open_air": ("sensor_name", "measurement_key"),
    "synthetic": (
        "source_name",
        "measurement_key",
        "measurement_description",
        "unit",
    ),
}
SOURCE_FIELDS = (
    "source_name",
    "sensor_name",
    "station_name",
    "beach_name",
    "data_stream_id",
    "datasourceid",
)
REQUIRED_FIELDS = ("topic", "measurement_key", "text")
DEFAULT_CLUSTERING_METHOD = "DBSCAN similarity threshold"
KMEANS_METHOD = "K-means fixed-k baseline"
DATASET_MODES = ("Combine selected datasets", "Cluster each selected dataset separately")


class Encoder(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ClusteringParameters:
    similarity_threshold: float = 0.80
    min_samples: int = 2
    method: str = DEFAULT_CLUSTERING_METHOD
    k: int | None = None


@dataclass(frozen=True)
class EmbeddingBundle:
    embeddings: np.ndarray
    topics: np.ndarray
    datasets: np.ndarray
    measurement_keys: np.ndarray
    sources: np.ndarray
    texts: np.ndarray
    model_name: str
    text_hash: str
    reused: bool = False
    representation: str = ""
    embedding_dimension: int = 0
    ordered_topic_hash: str = ""
    expected_groups: np.ndarray | None = None
    variant_types: np.ndarray | None = None
    evaluation_labels: np.ndarray | None = None


def infer_dataset(path: Path) -> str:
    matches = [dataset for dataset in DATASET_FIELD_ORDER if dataset in path.name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Cannot infer exactly one dataset from filename: {path.name}")
    return matches[0]


def load_jsonl_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_topics: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required JSONL input is missing: {path}")
        dataset = infer_dataset(path)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Malformed JSON in {path} at line {line_number}: {error.msg}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object in {path} at line {line_number}")
                missing = [field for field in REQUIRED_FIELDS if field not in record]
                if missing:
                    raise ValueError(
                        f"Missing required field(s) {missing} in {path} at line {line_number}"
                    )
                for field in REQUIRED_FIELDS:
                    if not str(record[field]).strip():
                        raise ValueError(f"Empty {field} in {path} at line {line_number}")
                topic = str(record["topic"])
                if topic in seen_topics:
                    raise ValueError(f"Duplicate topic across combined inputs: {topic}")
                seen_topics.add(topic)
                records.append({**record, "dataset": dataset})
    return records


def source_field_and_value(record: Mapping[str, Any]) -> tuple[str, str]:
    for field in SOURCE_FIELDS:
        value = record.get(field)
        if value is not None and str(value).strip():
            return field, str(value).strip()
    raise ValueError(f"Record has no usable source field: {record.get('topic', '<unknown>')}")


def source_value(record: Mapping[str, Any]) -> str:
    return source_field_and_value(record)[1]


def ordered_metadata(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    dataset = str(record.get("dataset", ""))
    if dataset not in DATASET_FIELD_ORDER:
        raise ValueError(f"Unsupported or missing dataset: {dataset!r}")
    fields = [
        (key, str(record[key]).strip())
        for key in DATASET_FIELD_ORDER[dataset]
        if record.get(key) is not None and str(record[key]).strip()
    ]
    if not fields:
        raise ValueError(f"No embedding metadata for topic: {record.get('topic')}")
    return fields


def normalize_metadata_text(value: str) -> str:
    normalized = value.strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", normalized)


def construct_representation(record: Mapping[str, Any], strategy: str) -> str:
    if strategy not in BASE_REPRESENTATIONS:
        raise ValueError(f"Unsupported direct-text representation: {strategy}")
    fields = ordered_metadata(record)
    if strategy == VALUE_ONLY:
        return " | ".join(value for _, value in fields)
    if strategy == KEY_ONLY:
        return " | ".join(key for key, _ in fields)
    if strategy == KEY_VALUE:
        return " | ".join(f"{key}: {value}" for key, value in fields)
    return " | ".join(
        f"{normalize_metadata_text(key)}: {normalize_metadata_text(value)}"
        for key, value in fields
    )


def construct_representations(records: Sequence[Mapping[str, Any]], strategy: str) -> list[str]:
    return [construct_representation(record, strategy) for record in records]


def construct_all_representations(records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    return {strategy: construct_representations(records, strategy) for strategy in BASE_REPRESENTATIONS}


def weighted_component_texts(records: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    keys = construct_representations(records, KEY_ONLY)
    values = construct_representations(records, VALUE_ONLY)
    combined = [f"KEY COMPONENT:\n{key}\n\nVALUE COMPONENT:\n{value}" for key, value in zip(keys, values)]
    return keys, values, combined


def text_hash(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def ordered_topic_hash(topics: Sequence[str]) -> str:
    return text_hash([str(topic) for topic in topics])


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-") or "model"


def load_sentence_transformer(model_name: str = MODEL_NAME, device: str = "cpu") -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    was_vector = array.ndim == 1
    if was_vector:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("Embeddings must be a vector or a two-dimensional matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)
    return normalized[0] if was_vector else normalized


def encode_texts(model: Encoder, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vectors = model.encode(
        list(texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
    )
    return l2_normalize(np.asarray(vectors, dtype=np.float32))


def weighted_embedding_matrix(
    key_embeddings: np.ndarray,
    value_embeddings: np.ndarray,
    key_weight: float,
    value_weight: float,
) -> np.ndarray:
    if key_weight < 0 or value_weight < 0 or not np.isclose(key_weight + value_weight, 1.0):
        raise ValueError("key_weight and value_weight must be non-negative and sum to one")
    keys = l2_normalize(key_embeddings)
    values = l2_normalize(value_embeddings)
    if keys.shape != values.shape:
        raise ValueError("Key and value embeddings must have identical shapes")
    return l2_normalize(key_weight * keys + value_weight * values)


def _string_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=np.str_)


def embedding_cache_matches(
    path: Path,
    model_name: str,
    input_text_hash: str,
    *,
    representation: str | None = None,
    topic_hash: str | None = None,
    record_count: int | None = None,
    embedding_dimension: int | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as cached:
            required = {
                "embeddings", "topics", "datasets", "measurement_keys", "texts",
                "model_name", "text_hash", "ordered_topic_hash", "embedding_dimension",
                "normalized",
            }
            if not required.issubset(cached.files):
                return False
            if "sources" not in cached.files and "source_values" not in cached.files:
                return False
            embeddings = np.asarray(cached["embeddings"])
            checks = [
                str(cached["model_name"].item()) == model_name,
                str(cached["text_hash"].item()) == input_text_hash,
                bool(cached["normalized"].item()),
                len(cached["topics"])
                == len(cached["datasets"])
                == len(cached["measurement_keys"])
                == len(cached["texts"])
                == len(cached["sources"] if "sources" in cached.files else cached["source_values"]),
                int(cached["embedding_dimension"].item()) == (embeddings.shape[1] if embeddings.ndim == 2 else -1),
            ]
            if representation is not None:
                checks.append("representation" in cached.files and str(cached["representation"].item()) == representation)
            if topic_hash is not None:
                checks.append(str(cached["ordered_topic_hash"].item()) == topic_hash)
            if record_count is not None:
                checks.append(len(cached["topics"]) == record_count)
            if embedding_dimension is not None:
                checks.append(embeddings.ndim == 2 and embeddings.shape[1] == embedding_dimension)
            return all(checks)
    except (OSError, ValueError, KeyError, IndexError):
        return False


def load_embedding_cache(path: Path, *, reused: bool = True) -> EmbeddingBundle:
    with np.load(path, allow_pickle=False) as cached:
        embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
        return EmbeddingBundle(
            embeddings=embeddings,
            topics=cached["topics"].copy(),
            datasets=cached["datasets"].copy(),
            measurement_keys=cached["measurement_keys"].copy(),
            sources=(cached["source_values"] if "source_values" in cached.files else cached["sources"]).copy(),
            texts=cached["texts"].copy(),
            model_name=str(cached["model_name"].item()),
            text_hash=str(cached["text_hash"].item()),
            reused=reused,
            representation=str(cached["representation"].item()) if "representation" in cached.files else "",
            embedding_dimension=int(cached["embedding_dimension"].item()) if "embedding_dimension" in cached.files else embeddings.shape[1],
            ordered_topic_hash=str(cached["ordered_topic_hash"].item()) if "ordered_topic_hash" in cached.files else ordered_topic_hash(cached["topics"]),
            expected_groups=cached["expected_groups"].copy() if "expected_groups" in cached.files else np.asarray([""] * len(cached["topics"])),
            variant_types=cached["variant_types"].copy() if "variant_types" in cached.files else np.asarray([""] * len(cached["topics"])),
            evaluation_labels=cached["evaluation_labels"].copy() if "evaluation_labels" in cached.files else np.asarray([""] * len(cached["topics"])),
        )


def write_embedding_cache(path: Path, bundle: EmbeddingBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            embeddings=bundle.embeddings,
            topics=bundle.topics,
            datasets=bundle.datasets,
            measurement_keys=bundle.measurement_keys,
            sources=bundle.sources,
            source_values=bundle.sources,
            texts=bundle.texts,
            model_name=np.asarray(bundle.model_name),
            embedding_dimension=np.asarray(bundle.embedding_dimension),
            text_hash=np.asarray(bundle.text_hash),
            ordered_topic_hash=np.asarray(bundle.ordered_topic_hash),
            normalized=np.asarray(True),
            representation=np.asarray(bundle.representation),
            expected_groups=bundle.expected_groups if bundle.expected_groups is not None else np.asarray([""] * len(bundle.topics)),
            variant_types=bundle.variant_types if bundle.variant_types is not None else np.asarray([""] * len(bundle.topics)),
            evaluation_labels=bundle.evaluation_labels if bundle.evaluation_labels is not None else np.asarray([""] * len(bundle.topics)),
        )
    os.replace(temporary, path)


def get_or_create_embedding_cache(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    strategy: str,
    model_name: str,
    encoder_factory: Callable[[], Encoder],
) -> EmbeddingBundle:
    if strategy not in BASE_REPRESENTATIONS:
        raise ValueError("Only base representations are persisted as NPZ caches")
    texts = construct_representations(records, strategy)
    topics = _string_array([record["topic"] for record in records])
    input_text_hash = text_hash(texts)
    topic_hash = ordered_topic_hash(topics)
    if embedding_cache_matches(
        path,
        model_name,
        input_text_hash,
        representation=strategy,
        topic_hash=topic_hash,
        record_count=len(records),
    ):
        return load_embedding_cache(path, reused=True)
    embeddings = encode_texts(encoder_factory(), texts)
    bundle = EmbeddingBundle(
        embeddings=embeddings,
        topics=topics,
        datasets=_string_array([record["dataset"] for record in records]),
        measurement_keys=_string_array([record["measurement_key"] for record in records]),
        sources=_string_array([source_value(record) for record in records]),
        texts=_string_array(texts),
        model_name=model_name,
        text_hash=input_text_hash,
        representation=strategy,
        embedding_dimension=embeddings.shape[1] if embeddings.ndim == 2 else 0,
        ordered_topic_hash=topic_hash,
        expected_groups=_string_array([record.get("expected_group", "") for record in records]),
        variant_types=_string_array([record.get("variant_type", "") for record in records]),
        evaluation_labels=_string_array([record.get("expected_evaluation_label", "") for record in records]),
    )
    write_embedding_cache(path, bundle)
    return bundle


def build_weighted_bundle(
    key_bundle: EmbeddingBundle,
    value_bundle: EmbeddingBundle,
    key_weight: float,
    value_weight: float,
) -> EmbeddingBundle:
    if not np.array_equal(key_bundle.topics, value_bundle.topics):
        raise ValueError("Key and value embedding bundles must use identical topic order")
    texts = [
        f"KEY COMPONENT:\n{key}\n\nVALUE COMPONENT:\n{value}"
        for key, value in zip(key_bundle.texts, value_bundle.texts)
    ]
    embeddings = weighted_embedding_matrix(
        key_bundle.embeddings, value_bundle.embeddings, key_weight, value_weight
    )
    return EmbeddingBundle(
        embeddings=embeddings,
        topics=key_bundle.topics.copy(),
        datasets=key_bundle.datasets.copy(),
        measurement_keys=key_bundle.measurement_keys.copy(),
        sources=key_bundle.sources.copy(),
        texts=_string_array(texts),
        model_name=key_bundle.model_name,
        text_hash=text_hash(texts),
        representation=WEIGHTED_KEY_VALUE,
        embedding_dimension=embeddings.shape[1] if embeddings.ndim == 2 else 0,
        ordered_topic_hash=key_bundle.ordered_topic_hash,
        reused=key_bundle.reused and value_bundle.reused,
        expected_groups=key_bundle.expected_groups.copy() if key_bundle.expected_groups is not None else None,
        variant_types=key_bundle.variant_types.copy() if key_bundle.variant_types is not None else None,
        evaluation_labels=key_bundle.evaluation_labels.copy() if key_bundle.evaluation_labels is not None else None,
    )


def threshold_to_eps(similarity_threshold: float) -> float:
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    return 1.0 - similarity_threshold


def cluster_embeddings(embeddings: np.ndarray, parameters: ClusteringParameters) -> np.ndarray:
    vectors = l2_normalize(embeddings)
    if len(vectors) == 0:
        return np.empty(0, dtype=int)
    if parameters.method == DEFAULT_CLUSTERING_METHOD:
        if parameters.min_samples < 2:
            raise ValueError("DBSCAN min_samples must be at least 2")
        return DBSCAN(
            eps=threshold_to_eps(parameters.similarity_threshold),
            min_samples=parameters.min_samples,
            metric="cosine",
        ).fit_predict(vectors)
    if parameters.method == KMEANS_METHOD:
        if parameters.k is None:
            raise ValueError("K-means requires an explicit k")
        if not 2 <= parameters.k <= len(vectors):
            raise ValueError("K-means k must be between 2 and the topic count")
        return KMeans(n_clusters=parameters.k, random_state=0, n_init="auto").fit_predict(vectors)
    raise ValueError(f"Unsupported clustering method: {parameters.method}")


def cluster_all_strategies(
    embeddings_by_strategy: Mapping[str, np.ndarray],
    parameters: ClusteringParameters,
) -> dict[str, np.ndarray]:
    unknown = set(embeddings_by_strategy) - set(REPRESENTATIONS)
    if unknown:
        raise ValueError(f"Unknown representation embeddings: {sorted(unknown)}")
    return {
        strategy: cluster_embeddings(embeddings_by_strategy[strategy], parameters)
        for strategy in REPRESENTATIONS
        if strategy in embeddings_by_strategy
    }


def cluster_by_dataset_mode(
    embeddings: np.ndarray,
    datasets: Sequence[str],
    selected_datasets: Sequence[str],
    parameters: ClusteringParameters,
    mode: str,
) -> dict[str, np.ndarray]:
    selected = set(selected_datasets)
    mask = np.isin(np.asarray(datasets), np.asarray(list(selected)))
    if mode == DATASET_MODES[0]:
        return {"combined": cluster_embeddings(np.asarray(embeddings)[mask], parameters)}
    if mode != DATASET_MODES[1]:
        raise ValueError(f"Unsupported dataset mode: {mode}")
    dataset_array = np.asarray(datasets)
    return {
        dataset: cluster_embeddings(np.asarray(embeddings)[dataset_array == dataset], parameters)
        for dataset in selected_datasets
        if dataset in selected
    }


def filter_bundle(bundle: EmbeddingBundle, datasets: Sequence[str]) -> EmbeddingBundle:
    mask = np.isin(bundle.datasets, np.asarray(list(datasets)))
    return EmbeddingBundle(
        embeddings=bundle.embeddings[mask],
        topics=bundle.topics[mask],
        datasets=bundle.datasets[mask],
        measurement_keys=bundle.measurement_keys[mask],
        sources=bundle.sources[mask],
        texts=bundle.texts[mask],
        model_name=bundle.model_name,
        text_hash=bundle.text_hash,
        reused=bundle.reused,
        representation=bundle.representation,
        embedding_dimension=bundle.embedding_dimension,
        ordered_topic_hash=ordered_topic_hash(bundle.topics[mask]),
        expected_groups=bundle.expected_groups[mask] if bundle.expected_groups is not None else None,
        variant_types=bundle.variant_types[mask] if bundle.variant_types is not None else None,
        evaluation_labels=bundle.evaluation_labels[mask] if bundle.evaluation_labels is not None else None,
    )


def cluster_summary(labels: Sequence[int]) -> dict[str, int | float]:
    counts = Counter(int(label) for label in labels if int(label) != -1)
    sizes = list(counts.values())
    return {
        "cluster_count": len(sizes),
        "noise_count": sum(int(label) == -1 for label in labels),
        "largest_cluster_size": max(sizes, default=0),
        "median_cluster_size": float(np.median(sizes)) if sizes else 0.0,
        "singleton_count": sum(size == 1 for size in sizes),
    }


def ranked_cluster_members(
    embeddings: np.ndarray, labels: Sequence[int], cluster_id: int
) -> list[tuple[int, float]]:
    indices = np.flatnonzero(np.asarray(labels) == cluster_id)
    if len(indices) == 0:
        return []
    members = l2_normalize(np.asarray(embeddings)[indices])
    centroid = l2_normalize(members.mean(axis=0))
    similarities = members @ centroid
    ranked = sorted(zip(indices.tolist(), similarities.tolist()), key=lambda item: (-item[1], item[0]))
    return [(index, float(similarity)) for index, similarity in ranked]


def representative_topics(
    embeddings: np.ndarray, labels: Sequence[int], cluster_id: int, limit: int
) -> list[tuple[int, float]]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    return ranked_cluster_members(embeddings, labels, cluster_id)[:limit]


def similarity_to_cluster_representatives(
    embeddings: np.ndarray, labels: Sequence[int]
) -> np.ndarray:
    similarities = np.full(len(labels), np.nan, dtype=np.float32)
    for cluster_id in sorted({int(label) for label in labels if int(label) != -1}):
        for index, similarity in ranked_cluster_members(embeddings, labels, cluster_id):
            similarities[index] = similarity
    return similarities


def cluster_neighbors(
    selected_index: int, embeddings: np.ndarray, labels: Sequence[int], limit: int
) -> list[tuple[int, float]]:
    labels_array = np.asarray(labels)
    cluster_id = int(labels_array[selected_index])
    if cluster_id == -1:
        return []
    members = np.flatnonzero(labels_array == cluster_id)
    members = members[members != selected_index]
    query = l2_normalize(np.asarray(embeddings)[selected_index])
    scores = l2_normalize(np.asarray(embeddings)[members]) @ query
    ranked = sorted(zip(members.tolist(), scores.tolist()), key=lambda item: (-item[1], item[0]))
    return [(index, float(score)) for index, score in ranked[:limit]]


def controlled_cluster_purity(labels: Sequence[int], expected_labels: Sequence[str]) -> float:
    """Purity over discovered non-noise records; noise is reported separately."""

    labels_array = np.asarray(labels)
    expected_array = np.asarray(expected_labels)
    evaluated = labels_array != -1
    if not np.any(evaluated):
        return 0.0
    correct = 0
    for cluster_id in sorted(set(labels_array[evaluated].tolist())):
        cluster_expected = expected_array[(labels_array == cluster_id) & evaluated]
        if len(cluster_expected):
            correct += Counter(cluster_expected.tolist()).most_common(1)[0][1]
    return float(correct / int(np.sum(evaluated)))


def controlled_benchmark_metrics(
    labels: Sequence[int],
    expected_labels: Sequence[str],
    expected_groups: Sequence[str],
) -> dict[str, float | int]:
    """Calculate deterministic synthetic-only metrics with all noise retained."""

    discovered = np.asarray(labels).astype(int)
    expected = np.asarray(expected_labels).astype(str)
    # DBSCAN noise is retained as one explicit discovered label for ARI/NMI.
    discovered_for_metrics = np.where(discovered == -1, -1_000_000, discovered)
    return {
        "adjusted_rand_index": float(adjusted_rand_score(expected, discovered_for_metrics)),
        "normalized_mutual_information": float(normalized_mutual_info_score(expected, discovered_for_metrics)),
        "cluster_purity": controlled_cluster_purity(discovered, expected),
        "discovered_cluster_count": int(len({int(label) for label in discovered if int(label) != -1})),
        "noise_count": int(np.sum(discovered == -1)),
        "expected_semantic_group_count": int(len({group for group in expected_groups if group != "outlier"})),
    }


def pca_coordinates(embeddings: np.ndarray) -> np.ndarray:
    """Return two PCA coordinates from the original full-dimensional embeddings."""

    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if vectors.shape[1] == 1:
        return np.column_stack((vectors[:, 0], np.zeros(len(vectors), dtype=np.float32)))
    return PCA(n_components=2, random_state=0).fit_transform(vectors).astype(np.float32)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    vectors = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    return np.asarray(vectors @ vectors.T, dtype=np.float32)
