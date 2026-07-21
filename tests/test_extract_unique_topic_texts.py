from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scripts import extract_unique_topic_texts as extractor


def write_fixture(root: Path, filename: str, rows: list[dict[str, Any]]) -> None:
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(raw_dir / filename, index=False)


def beach_rows(source_column: str) -> list[dict[str, Any]]:
    return [
        {
            source_column: "Example Source",
            "Measurement Timestamp": "2026-01-01T00:00:00",
            "Measurement ID": "id-1",
            "Measurement Timestamp Label": "January 1",
            "Air Temperature": 0,
            "Wind Speed": 2.5,
        },
        {
            source_column: "Example Source",
            "Measurement Timestamp": "2026-01-01T01:00:00",
            "Measurement ID": "id-2",
            "Measurement Timestamp Label": "January 1",
            "Air Temperature": 1,
            "Wind Speed": None,
        },
    ]


def make_summary(dataset: extractor.Resolved, output_path: Path) -> extractor.Summary:
    stat = dataset.path.stat()
    return extractor.Summary(
        dataset=dataset.definition.key,
        source_path=dataset.path,
        output_path=output_path,
        measurement_columns=list(dataset.measurement_columns),
        duplicate_pair_status={pair: True for pair in extractor.OPEN_AIR_DUPLICATE_PAIRS},
        input_size=stat.st_size,
        input_mtime_ns=stat.st_mtime_ns,
    )


def scan_wide_fixture(root: Path, dataset_key: str) -> extractor.Summary:
    dataset = extractor.resolve_dataset(root, extractor.DEFINITIONS[dataset_key])
    summary = make_summary(dataset, root / "output.jsonl")
    extractor.scan_wide(dataset, summary, chunk_size=2)
    return summary


def test_beach_weather_excludes_measurement_timestamp_label(tmp_path: Path) -> None:
    write_fixture(tmp_path, "02_beach_weather_fixture.csv", beach_rows("Station Name"))
    dataset = extractor.resolve_dataset(tmp_path, extractor.DEFINITIONS["beach_weather"])
    assert "Measurement Timestamp Label" not in dataset.measurement_columns


def test_beach_water_excludes_measurement_timestamp_label(tmp_path: Path) -> None:
    write_fixture(tmp_path, "03_beach_water_fixture.csv", beach_rows("Beach Name"))
    dataset = extractor.resolve_dataset(tmp_path, extractor.DEFINITIONS["beach_water"])
    assert "Measurement Timestamp Label" not in dataset.measurement_columns


def test_actual_measurement_columns_are_retained(tmp_path: Path) -> None:
    write_fixture(tmp_path, "02_beach_weather_fixture.csv", beach_rows("Station Name"))
    dataset = extractor.resolve_dataset(tmp_path, extractor.DEFINITIONS["beach_weather"])
    assert dataset.measurement_columns == ("Air Temperature", "Wind Speed")


def test_generated_topics_are_unique(tmp_path: Path) -> None:
    write_fixture(tmp_path, "02_beach_weather_fixture.csv", beach_rows("Station Name"))
    summary = scan_wide_fixture(tmp_path, "beach_weather")
    topics = [record["topic"] for record in summary.unique_records.values()]
    assert len(topics) == len(set(topics)) == 2


def test_output_records_have_required_fields(tmp_path: Path) -> None:
    write_fixture(tmp_path, "02_beach_weather_fixture.csv", beach_rows("Station Name"))
    summary = scan_wide_fixture(tmp_path, "beach_weather")
    for record in summary.unique_records.values():
        assert {"topic", "measurement_key", "text"} <= record.keys()
        assert record["measurement_key"]
        assert record["text"]


def test_output_records_omit_prohibited_fields(tmp_path: Path) -> None:
    write_fixture(tmp_path, "02_beach_weather_fixture.csv", beach_rows("Station Name"))
    summary = scan_wide_fixture(tmp_path, "beach_weather")
    prohibited = {
        "time",
        "measurement_timestamp",
        "measurement_time",
        "measurement_value",
        "value",
        "timestamp",
        "tags",
        "latitude",
        "longitude",
        "record_id",
    }
    assert all(not prohibited.intersection(record) for record in summary.unique_records.values())


def test_zero_value_counts_as_observed_measurement(tmp_path: Path) -> None:
    rows = beach_rows("Station Name")[:1]
    rows[0]["Wind Speed"] = None
    write_fixture(tmp_path, "02_beach_weather_fixture.csv", rows)
    summary = scan_wide_fixture(tmp_path, "beach_weather")
    assert ("Example Source", "Air Temperature") in summary.unique_records


def test_empty_and_null_values_do_not_create_combinations(tmp_path: Path) -> None:
    rows = [
        {
            "Station Name": "Example Source",
            "Measurement Timestamp": "2026-01-01",
            "Measurement ID": "id-1",
            "Measurement Timestamp Label": "January 1",
            "Air Temperature": "",
        },
        {
            "Station Name": "Example Source",
            "Measurement Timestamp": "2026-01-02",
            "Measurement ID": "id-2",
            "Measurement Timestamp Label": "January 2",
            "Air Temperature": None,
        },
    ]
    write_fixture(tmp_path, "02_beach_weather_fixture.csv", rows)
    summary = scan_wide_fixture(tmp_path, "beach_weather")
    assert summary.unique_records == {}
    assert summary.eligible_observations == 0


def test_open_air_underscore_one_columns_remain_measurements(tmp_path: Path) -> None:
    rows = [
        {
            "datasourceid": "D1",
            "sensor_name": "Sensor 1",
            "time": "2026-01-01",
            "latitude": 1.0,
            "longitude": 2.0,
            "location": "POINT (2 1)",
            "record_id": "r1",
            "temperatureinternalindividual": 20.0,
            "temperatureinternalindividual_1": 20.0,
        }
    ]
    write_fixture(tmp_path, "04_open_air_fixture_raw.csv", rows)
    dataset = extractor.resolve_dataset(tmp_path, extractor.DEFINITIONS["open_air"])
    assert "temperatureinternalindividual" in dataset.measurement_columns
    assert "temperatureinternalindividual_1" in dataset.measurement_columns


def test_sgim_text_fields_use_fixed_order(tmp_path: Path) -> None:
    rows = [
        {
            "Data Stream ID": "7",
            "Measurement Time": "2026-01-01",
            "Measurement Value": 12,
            "Measurement Type": "Temperature",
            "Measurement Title": "Title",
            "Measurement Description": "Description",
            "Measurement Medium": "Atmosphere",
            "Units": "Degrees Celsius",
            "Units Abbreviation": "degC",
            "Measurement Period Type": "Instantaneous",
        }
    ]
    write_fixture(tmp_path, "01_sgim_fixture.csv", rows)
    dataset = extractor.resolve_dataset(tmp_path, extractor.DEFINITIONS["sgim"])
    summary = make_summary(dataset, tmp_path / "output.jsonl")
    extractor.scan_sgim(dataset, summary, chunk_size=1)
    record = summary.unique_records[("7", "Temperature")]
    assert record["text"] == (
        "measurement_type: Temperature | measurement_title: Title | "
        "measurement_description: Description | measurement_medium: Atmosphere | "
        "units: Degrees Celsius | units_abbreviation: degC | "
        "measurement_period_type: Instantaneous"
    )
