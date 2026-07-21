from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.embedding_engine import (
    construct_representation,
    load_jsonl_records,
    top_k_retrieval,
    weighted_top_k_vote,
)


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def record(topic: str, measurement: str, **metadata: str) -> dict[str, str]:
    return {
        "topic": topic,
        "measurement_key": measurement,
        "text": f"measurement_key: {measurement}",
        **metadata,
    }


def test_all_four_jsonl_formats_load_with_inferred_dataset(tmp_path: Path) -> None:
    paths = [
        tmp_path / "01_sgim_topic_texts.jsonl",
        tmp_path / "02_beach_weather_topic_texts.jsonl",
        tmp_path / "03_beach_water_topic_texts.jsonl",
        tmp_path / "04_open_air_topic_texts.jsonl",
    ]
    values = [
        record("sgim/1/temp", "Temperature", data_stream_id="1"),
        record("beach_weather/a/temp", "Air Temperature", station_name="A"),
        record("beach_water/b/temp", "Water Temperature", beach_name="B"),
        record("open_air/d/temp", "temperature", datasourceid="D", sensor_name="S"),
    ]
    for path, value in zip(paths, values):
        write_jsonl(path, [value])
    loaded = load_jsonl_records(paths)
    assert [item["dataset"] for item in loaded] == [
        "sgim",
        "beach_weather",
        "beach_water",
        "open_air",
    ]


def test_source_plus_measurement_uses_first_available_source_field() -> None:
    value = record(
        "open_air/d/temp",
        "Temperature",
        sensor_name="Sensor Name",
        datasourceid="Datasource ID",
    )
    assert construct_representation(value, "Source plus measurement") == (
        "source: Sensor Name | measurement_key: Temperature"
    )


def test_measurement_only_excludes_source_name() -> None:
    value = record("weather/a/wind", "Wind Speed", station_name="Named Station")
    text = construct_representation(value, "Measurement only")
    assert text == "measurement_key: Wind Speed"
    assert "Named Station" not in text


def test_source_only_excludes_measurement_key() -> None:
    value = record("water/a/turbidity", "Turbidity", beach_name="Named Beach")
    text = construct_representation(value, "Source only negative control")
    assert text == "source: Named Beach"
    assert "Turbidity" not in text


def test_cosine_similarity_ranking_is_correct() -> None:
    candidates = [record(f"topic/{index}", f"label-{index}") for index in range(3)]
    embeddings = np.array([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]])
    results = top_k_retrieval(candidates, embeddings, np.array([1.0, 0.0]), 3)
    assert [result["topic"] for result in results] == ["topic/0", "topic/1", "topic/2"]
    assert results[0]["cosine_similarity"] == pytest.approx(1.0)


def test_exact_self_match_can_be_excluded() -> None:
    candidates = [record("topic/self", "A"), record("topic/other", "B")]
    embeddings = np.eye(2)
    results = top_k_retrieval(
        candidates,
        embeddings,
        np.array([1.0, 0.0]),
        2,
        query_topic="topic/self",
        exclude_self=True,
    )
    assert [result["topic"] for result in results] == ["topic/other"]


def test_top_k_never_exceeds_candidate_count() -> None:
    candidates = [record("topic/a", "A"), record("topic/b", "B")]
    results = top_k_retrieval(candidates, np.eye(2), np.array([1.0, 0.0]), 25)
    assert len(results) == 2


def test_weighted_vote_sums_scores_by_measurement_key() -> None:
    neighbors = [
        {**record("topic/a", "Temperature"), "cosine_similarity": 0.6},
        {**record("topic/b", "Temperature"), "cosine_similarity": 0.5},
        {**record("topic/c", "Humidity"), "cosine_similarity": 0.9},
    ]
    result = weighted_top_k_vote(neighbors)
    assert result["predicted_label"] == "Temperature"
    assert result["score_by_label"]["Temperature"] == pytest.approx(1.1)
    assert len(result["contributing_neighbors"]) == 3


def test_non_positive_similarities_do_not_force_prediction() -> None:
    neighbors = [
        {**record("topic/a", "Temperature"), "cosine_similarity": 0.0},
        {**record("topic/b", "Humidity"), "cosine_similarity": -0.2},
    ]
    result = weighted_top_k_vote(neighbors)
    assert result["predicted_label"] is None
    assert result["score_by_label"] == {}
    assert result["contributing_neighbors"] == []


def test_missing_required_fields_raise_readable_error(tmp_path: Path) -> None:
    path = tmp_path / "01_sgim_topic_texts.jsonl"
    write_jsonl(path, [{"topic": "sgim/1/temp", "text": "text only"}])
    with pytest.raises(ValueError, match="Missing required field"):
        load_jsonl_records([path])
