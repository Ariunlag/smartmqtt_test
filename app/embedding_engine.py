"""Reusable data, representation, embedding, and retrieval helpers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np


MODEL_OPTIONS = ("sentence-transformers/all-MiniLM-L6-v2",)
REPRESENTATIONS = (
    "Current extracted text",
    "Measurement only",
    "Source plus measurement",
    "Source only negative control",
)
SOURCE_FIELDS = (
    "sensor_name",
    "station_name",
    "beach_name",
    "data_stream_id",
    "datasourceid",
)
REQUIRED_FIELDS = ("topic", "measurement_key", "text")


class Encoder(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: Any) -> Any: ...


def infer_dataset(path: Path) -> str:
    name = path.name.lower()
    matches = [
        dataset
        for dataset in ("sgim", "beach_weather", "beach_water", "open_air")
        if dataset in name
    ]
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
                if not str(record["topic"]).strip():
                    raise ValueError(f"Empty topic in {path} at line {line_number}")
                if not str(record["measurement_key"]).strip():
                    raise ValueError(f"Empty measurement_key in {path} at line {line_number}")
                if not str(record["text"]).strip():
                    raise ValueError(f"Empty text in {path} at line {line_number}")
                topic = str(record["topic"])
                if topic in seen_topics:
                    raise ValueError(f"Duplicate topic across combined inputs: {topic}")
                seen_topics.add(topic)
                enriched = dict(record)
                enriched["dataset"] = dataset
                records.append(enriched)
    return records


def source_field_and_value(record: Mapping[str, Any]) -> tuple[str, str]:
    for field in SOURCE_FIELDS:
        value = record.get(field)
        if value is not None and str(value).strip():
            return field, str(value).strip()
    raise ValueError(f"Record has no usable source field: {record.get('topic', '<unknown>')}")


def construct_representation(record: Mapping[str, Any], representation: str) -> str:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unsupported representation: {representation}")
    measurement_key = str(record.get("measurement_key", "")).strip()
    if not measurement_key:
        raise ValueError("measurement_key is required to construct a representation")
    if representation == "Current extracted text":
        text = str(record.get("text", "")).strip()
        if not text:
            raise ValueError("text is required for the current extracted representation")
        return text
    if representation == "Measurement only":
        return f"measurement_key: {measurement_key}"
    _, source = source_field_and_value(record)
    if representation == "Source plus measurement":
        return f"source: {source} | measurement_key: {measurement_key}"
    return f"source: {source}"


def construct_representations(
    records: Sequence[Mapping[str, Any]], representation: str
) -> list[str]:
    return [construct_representation(record, representation) for record in records]


def text_hash(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def load_sentence_transformer(model_name: str, device: str = "cpu") -> Any:
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


def cosine_similarities(candidate_embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    candidates = l2_normalize(candidate_embeddings)
    query = l2_normalize(query_embedding)
    if candidates.ndim != 2 or query.ndim != 1:
        raise ValueError("Candidates must be a matrix and the query must be a vector")
    if candidates.shape[1] != query.shape[0]:
        raise ValueError("Candidate and query embedding dimensions differ")
    return candidates @ query


def top_k_retrieval(
    candidate_records: Sequence[Mapping[str, Any]],
    candidate_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int,
    *,
    query_topic: str | None = None,
    exclude_self: bool = False,
    minimum_similarity: float = -1.0,
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if len(candidate_records) != len(candidate_embeddings):
        raise ValueError("Candidate record and embedding counts differ")
    scores = cosine_similarities(candidate_embeddings, query_embedding)
    eligible: list[int] = []
    for index, (record, score) in enumerate(zip(candidate_records, scores)):
        if exclude_self and query_topic is not None and record.get("topic") == query_topic:
            continue
        if float(score) < minimum_similarity:
            continue
        eligible.append(index)
    ranked = sorted(eligible, key=lambda index: float(scores[index]), reverse=True)[:top_k]
    results: list[dict[str, Any]] = []
    for rank, index in enumerate(ranked, start=1):
        result = dict(candidate_records[index])
        result["rank"] = rank
        result["cosine_similarity"] = float(scores[index])
        result["candidate_index"] = index
        results.append(result)
    return results


def weighted_top_k_vote(neighbors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores: dict[str, float] = defaultdict(float)
    contributors: list[dict[str, Any]] = []
    for neighbor in neighbors:
        similarity = float(neighbor.get("cosine_similarity", 0.0))
        if similarity <= 0:
            continue
        label = str(neighbor.get("measurement_key", "")).strip()
        if not label:
            continue
        scores[label] += similarity
        contributors.append(
            {
                "topic": neighbor.get("topic"),
                "measurement_key": label,
                "cosine_similarity": similarity,
            }
        )
    ordered_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    predicted = next(iter(ordered_scores), None)
    return {
        "predicted_label": predicted,
        "score_by_label": ordered_scores,
        "contributing_neighbors": contributors,
    }
