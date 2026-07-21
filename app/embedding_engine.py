"""Model-independent representation, embedding-cache, and clustering helpers."""

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


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VALUE_ONLY = "VALUE_ONLY"
KEY_ONLY = "KEY_ONLY"
KEY_VALUE = "KEY_VALUE"
NORMALIZED_KEY_VALUE = "NORMALIZED_KEY_VALUE"
REPRESENTATIONS = (VALUE_ONLY, KEY_ONLY, KEY_VALUE, NORMALIZED_KEY_VALUE)
REPRESENTATION_LABELS = {
    VALUE_ONLY: "Value only",
    KEY_ONLY: "Key only",
    KEY_VALUE: "Key:value",
    NORMALIZED_KEY_VALUE: "Normalized key:value",
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
}
SOURCE_FIELDS = (
    "sensor_name",
    "station_name",
    "beach_name",
    "data_stream_id",
    "datasourceid",
)
REQUIRED_FIELDS = ("topic", "measurement_key", "text")
DEFAULT_CLUSTERING_METHOD = "DBSCAN threshold"
KMEANS_METHOD = "K-means fixed-k baseline"


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


def infer_dataset(path: Path) -> str:
    name = path.name.lower()
    matches = [dataset for dataset in DATASET_FIELD_ORDER if dataset in name]
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
    fields: list[tuple[str, str]] = []
    for key in DATASET_FIELD_ORDER[dataset]:
        value = record.get(key)
        if value is not None and str(value).strip():
            fields.append((key, str(value).strip()))
    if not fields:
        raise ValueError(f"No embedding metadata for topic: {record.get('topic')}")
    return fields


def normalize_metadata_text(value: str) -> str:
    """Apply the experiment's conservative deterministic normalization."""
    normalized = value.strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", normalized)


def construct_representation(record: Mapping[str, Any], strategy: str) -> str:
    if strategy not in REPRESENTATIONS:
        raise ValueError(f"Unsupported representation: {strategy}")
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


def construct_representations(
    records: Sequence[Mapping[str, Any]], strategy: str
) -> list[str]:
    return [construct_representation(record, strategy) for record in records]


def construct_all_representations(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    return {strategy: construct_representations(records, strategy) for strategy in REPRESENTATIONS}


def text_hash(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


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
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return l2_normalize(np.asarray(vectors, dtype=np.float32))


def _string_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=np.str_)


def embedding_cache_matches(path: Path, model_name: str, input_text_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as cached:
            required = {
                "embeddings", "topics", "datasets", "measurement_keys",
                "texts", "model_name", "text_hash",
            }
            has_sources = "sources" in cached.files or "source_values" in cached.files
            return required.issubset(cached.files) and has_sources and (
                str(cached["model_name"].item()) == model_name
                and str(cached["text_hash"].item()) == input_text_hash
            )
    except (OSError, ValueError, KeyError):
        return False


def load_embedding_cache(path: Path, *, reused: bool = True) -> EmbeddingBundle:
    with np.load(path, allow_pickle=False) as cached:
        return EmbeddingBundle(
            embeddings=np.asarray(cached["embeddings"], dtype=np.float32),
            topics=cached["topics"].copy(),
            datasets=cached["datasets"].copy(),
            measurement_keys=cached["measurement_keys"].copy(),
            sources=(cached["source_values"] if "source_values" in cached.files else cached["sources"]).copy(),
            texts=cached["texts"].copy(),
            model_name=str(cached["model_name"].item()),
            text_hash=str(cached["text_hash"].item()),
            reused=reused,
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
            text_hash=np.asarray(bundle.text_hash),
        )
    os.replace(temporary, path)


def get_or_create_embedding_cache(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    strategy: str,
    model_name: str,
    encoder_factory: Callable[[], Encoder],
) -> EmbeddingBundle:
    texts = construct_representations(records, strategy)
    input_text_hash = text_hash(texts)
    if embedding_cache_matches(path, model_name, input_text_hash):
        return load_embedding_cache(path, reused=True)
    embeddings = encode_texts(encoder_factory(), texts)
    bundle = EmbeddingBundle(
        embeddings=embeddings,
        topics=_string_array([record["topic"] for record in records]),
        datasets=_string_array([record["dataset"] for record in records]),
        measurement_keys=_string_array([record["measurement_key"] for record in records]),
        sources=_string_array([source_value(record) for record in records]),
        texts=_string_array(texts),
        model_name=model_name,
        text_hash=input_text_hash,
        reused=False,
    )
    write_embedding_cache(path, bundle)
    return bundle


def threshold_to_eps(similarity_threshold: float) -> float:
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    return 1.0 - similarity_threshold


def cluster_embeddings(
    embeddings: np.ndarray, parameters: ClusteringParameters
) -> np.ndarray:
    vectors = l2_normalize(embeddings)
    if len(vectors) == 0:
        return np.empty(0, dtype=int)
    if parameters.method == DEFAULT_CLUSTERING_METHOD:
        if parameters.min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        return DBSCAN(
            eps=threshold_to_eps(parameters.similarity_threshold),
            min_samples=parameters.min_samples,
            metric="cosine",
        ).fit_predict(vectors)
    if parameters.method == KMEANS_METHOD:
        if parameters.k is None:
            raise ValueError("K-means requires an explicit k")
        if not 1 <= parameters.k <= len(vectors):
            raise ValueError("K-means k must be between 1 and the topic count")
        return KMeans(n_clusters=parameters.k, random_state=0, n_init="auto").fit_predict(vectors)
    raise ValueError(f"Unsupported clustering method: {parameters.method}")


def cluster_all_strategies(
    embeddings_by_strategy: Mapping[str, np.ndarray],
    parameters: ClusteringParameters,
) -> dict[str, np.ndarray]:
    missing = set(REPRESENTATIONS) - set(embeddings_by_strategy)
    if missing:
        raise ValueError(f"Missing representation embeddings: {sorted(missing)}")
    return {
        strategy: cluster_embeddings(embeddings_by_strategy[strategy], parameters)
        for strategy in REPRESENTATIONS
    }


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
    ranked = sorted(
        zip(indices.tolist(), similarities.tolist()),
        key=lambda item: (-item[1], item[0]),
    )
    return [(index, float(similarity)) for index, similarity in ranked]


def representative_topics(
    embeddings: np.ndarray,
    labels: Sequence[int],
    cluster_id: int,
    limit: int,
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
    selected_index: int,
    embeddings: np.ndarray,
    labels: Sequence[int],
    limit: int,
) -> list[tuple[int, float]]:
    labels_array = np.asarray(labels)
    cluster_id = int(labels_array[selected_index])
    if cluster_id == -1:
        return []
    members = np.flatnonzero(labels_array == cluster_id)
    members = members[members != selected_index]
    query = l2_normalize(np.asarray(embeddings)[selected_index])
    scores = l2_normalize(np.asarray(embeddings)[members]) @ query
    ranked = sorted(
        zip(members.tolist(), scores.tolist()), key=lambda item: (-item[1], item[0])
    )
    return [(index, float(score)) for index, score in ranked[:limit]]
