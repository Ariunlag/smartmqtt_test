"""Extract one compact textual record per unique logical Chicago IoT topic."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd


DEFAULT_CHUNK_SIZE = 100_000
SENTINELS = (-99999, -99999.992, -100000)
DATASET_ORDER = ("beach_weather", "beach_water", "open_air", "sgim")
OPEN_AIR_DUPLICATE_PAIRS = (
    ("relhumidambientindividual", "relhumidambientindividual_1"),
    ("relhumidinternalindividual", "relhumidinternalindividual_1"),
    ("temperatureambientindividual", "temperatureambientindividual_1"),
    ("temperatureinternalindividual", "temperatureinternalindividual_1"),
)


@dataclass(frozen=True)
class Definition:
    key: str
    prefix: str
    input_patterns: tuple[str, ...]
    output_name: str
    source_alias: str
    timestamp_alias: str
    administrative_aliases: tuple[str, ...] = ()
    location_aliases: tuple[str, ...] = ()
    second_source_alias: str | None = None
    measurement_alias: str | None = None
    stable_text_aliases: tuple[str, ...] = ()
    long_format: bool = False


@dataclass(frozen=True)
class Resolved:
    definition: Definition
    path: Path
    columns: tuple[str, ...]
    source: str
    timestamp: str
    administrative: tuple[str, ...]
    locations: tuple[str, ...]
    second_source: str | None
    measurement: str | None
    measurement_columns: tuple[str, ...]
    stable_text_columns: tuple[tuple[str, str], ...]


@dataclass
class Summary:
    dataset: str
    source_path: Path
    output_path: Path
    measurement_columns: list[str]
    source_rows_scanned: int = 0
    sources: set[str] = field(default_factory=set)
    candidate_combinations: int = 0
    unique_records: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    measurement_non_null_counts: Counter[str] = field(default_factory=Counter)
    sentinel_counts: Counter[str] = field(default_factory=Counter)
    duplicate_pair_status: dict[tuple[str, str], bool] = field(default_factory=dict)
    duplicate_pair_both_non_null: Counter[tuple[str, str]] = field(default_factory=Counter)
    duplicate_pair_equal: Counter[tuple[str, str]] = field(default_factory=Counter)
    input_size: int = 0
    input_mtime_ns: int = 0
    input_unchanged: bool = False
    output_size: int = 0

    @property
    def duplicate_combinations_skipped(self) -> int:
        return self.candidate_combinations - len(self.unique_records)


DEFINITIONS: dict[str, Definition] = {
    "beach_weather": Definition(
        key="beach_weather",
        prefix="beach_weather",
        input_patterns=("02_beach_weather*.csv", "02_beach_weather*.csv*"),
        output_name="02_beach_weather_topic_texts.jsonl",
        source_alias="station_name",
        timestamp_alias="measurement_timestamp",
        administrative_aliases=("measurement_id", "record_id"),
    ),
    "beach_water": Definition(
        key="beach_water",
        prefix="beach_water",
        input_patterns=("03_beach_water*.csv", "03_beach_water*.csv*"),
        output_name="03_beach_water_topic_texts.jsonl",
        source_alias="beach_name",
        timestamp_alias="measurement_timestamp",
        administrative_aliases=("measurement_id", "record_id"),
    ),
    "open_air": Definition(
        key="open_air",
        prefix="open_air",
        input_patterns=("04_open_air*raw*.csv", "04_open_air*raw*.csv*"),
        output_name="04_open_air_topic_texts.jsonl",
        source_alias="datasourceid",
        second_source_alias="sensor_name",
        timestamp_alias="time",
        administrative_aliases=("record_id",),
        location_aliases=("latitude", "longitude", "location"),
    ),
    "sgim": Definition(
        key="sgim",
        prefix="sgim",
        input_patterns=("01_sgim*.csv", "01_sgim*.csv*"),
        output_name="01_sgim_topic_texts.jsonl",
        source_alias="data_stream_id",
        timestamp_alias="measurement_time",
        measurement_alias="measurement_type",
        stable_text_aliases=(
            "measurement_title",
            "measurement_description",
            "measurement_medium",
            "units",
            "units_abbreviation",
            "measurement_period_type",
        ),
        long_format=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("sgim", "beach_weather", "beach_water", "open_air", "all"),
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.strip().lower())).strip("_")


def slug(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "_").replace("/", "_")
    return re.sub(r"_+", "_", text)


def slash_safe(value: Any) -> str:
    return str(value).strip().replace("/", "_")


def present_mask(series: pd.Series) -> pd.Series:
    mask = series.notna()
    if isinstance(series.dtype, pd.StringDtype) or series.dtype == object:
        mask &= series.astype("string").str.strip().ne("").fillna(False)
    return mask


def native_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def locate_input(raw_dir: Path, definition: Definition) -> Path:
    matches: set[Path] = set()
    for pattern in definition.input_patterns:
        matches.update(path.resolve() for path in raw_dir.glob(pattern) if path.is_file())
    matches = {path for path in matches if "mqtt" not in path.name.lower()}
    if not matches:
        raise FileNotFoundError(f"No {definition.key} input found under {raw_dir}")
    if len(matches) > 1:
        rendered = "\n  ".join(str(path) for path in sorted(matches))
        raise RuntimeError(f"Multiple {definition.key} inputs matched:\n  {rendered}")
    return next(iter(matches))


def resolve_column(
    columns: Sequence[str], alias: str, *, required: bool = True
) -> str | None:
    matches = [column for column in columns if normalize_name(column) == alias]
    if len(matches) > 1:
        raise ValueError(f"Alias {alias!r} matched multiple columns: {matches}")
    if matches:
        return matches[0]
    if required:
        raise ValueError(f"Required column {alias!r} is absent; actual columns={list(columns)}")
    return None


def resolve_dataset(root: Path, definition: Definition) -> Resolved:
    path = locate_input(root / "data" / "raw", definition)
    columns = tuple(pd.read_csv(path, nrows=0).columns.tolist())
    source = resolve_column(columns, definition.source_alias)
    timestamp = resolve_column(columns, definition.timestamp_alias)
    assert source is not None and timestamp is not None
    administrative = tuple(
        column
        for alias in definition.administrative_aliases
        if (column := resolve_column(columns, alias, required=False)) is not None
    )
    locations = tuple(
        column
        for alias in definition.location_aliases
        if (column := resolve_column(columns, alias, required=True)) is not None
    )
    second_source = (
        resolve_column(columns, definition.second_source_alias)
        if definition.second_source_alias
        else None
    )
    if definition.long_format:
        measurement = resolve_column(columns, definition.measurement_alias or "")
        assert measurement is not None
        stable: list[tuple[str, str]] = []
        for alias in definition.stable_text_aliases:
            actual = resolve_column(columns, alias, required=False)
            if actual is not None:
                stable.append((alias, actual))
        measurements = (measurement,)
    else:
        measurement = None
        stable = []
        excluded = {source, timestamp, *administrative, *locations}
        if second_source:
            excluded.add(second_source)
        measurements = tuple(column for column in columns if column not in excluded)
    return Resolved(
        definition=definition,
        path=path,
        columns=columns,
        source=source,
        timestamp=timestamp,
        administrative=administrative,
        locations=locations,
        second_source=second_source,
        measurement=measurement,
        measurement_columns=measurements,
        stable_text_columns=tuple(stable),
    )


def selected_columns(dataset: Resolved) -> list[str]:
    columns = [dataset.source]
    if dataset.second_source:
        columns.append(dataset.second_source)
    if dataset.measurement:
        columns.append(dataset.measurement)
    columns.extend(actual for _, actual in dataset.stable_text_columns)
    if not dataset.definition.long_format:
        columns.extend(dataset.measurement_columns)
    return list(dict.fromkeys(columns))


def iter_chunks(dataset: Resolved, chunk_size: int) -> Iterator[pd.DataFrame]:
    if chunk_size <= 0:
        raise ValueError(f"chunk size must be positive; got {chunk_size}")
    dtype: dict[str, str] = {dataset.source: "string"}
    if dataset.second_source:
        dtype[dataset.second_source] = "string"
    if dataset.measurement:
        dtype[dataset.measurement] = "string"
    for _, actual in dataset.stable_text_columns:
        dtype[actual] = "string"
    yield from pd.read_csv(
        dataset.path,
        usecols=selected_columns(dataset),
        dtype=dtype,
        chunksize=chunk_size,
        low_memory=False,
    )


def topic_for(dataset: Resolved, source: str, measurement: str) -> str:
    if dataset.definition.key == "open_air":
        return f"open_air/{slash_safe(source)}/{slash_safe(measurement)}"
    return f"{dataset.definition.prefix}/{slug(source)}/{slug(measurement)}"


def base_record(dataset: Resolved, source: str, measurement: str) -> dict[str, Any]:
    if dataset.definition.key == "beach_weather":
        return {
            "topic": topic_for(dataset, source, measurement),
            "station_name": source,
            "measurement_key": measurement,
            "text": f"station_name: {source} | measurement_key: {measurement}",
        }
    if dataset.definition.key == "beach_water":
        return {
            "topic": topic_for(dataset, source, measurement),
            "beach_name": source,
            "measurement_key": measurement,
            "text": f"beach_name: {source} | measurement_key: {measurement}",
        }
    if dataset.definition.key == "open_air":
        return {
            "topic": topic_for(dataset, source, measurement),
            "datasourceid": source,
            "sensor_name": None,
            "measurement_key": measurement,
            "text": "",
        }
    return {
        "topic": topic_for(dataset, source, measurement),
        "data_stream_id": source,
        "measurement_key": measurement,
    }


def update_open_air_pair_checks(chunk: pd.DataFrame, summary: Summary) -> None:
    for pair in OPEN_AIR_DUPLICATE_PAIRS:
        left, right = pair
        if left not in chunk.columns or right not in chunk.columns:
            summary.duplicate_pair_status[pair] = False
            continue
        both_null = chunk[left].isna() & chunk[right].isna()
        both_non_null = chunk[left].notna() & chunk[right].notna()
        equal = chunk[left].eq(chunk[right])
        chunk_exact = bool((both_null | (both_non_null & equal)).all())
        summary.duplicate_pair_status[pair] = summary.duplicate_pair_status.get(pair, True) and chunk_exact
        summary.duplicate_pair_both_non_null[pair] += int(both_non_null.sum())
        summary.duplicate_pair_equal[pair] += int((both_non_null & equal).sum())


def scan_wide(dataset: Resolved, summary: Summary, chunk_size: int) -> None:
    for chunk_number, chunk in enumerate(iter_chunks(dataset, chunk_size), start=1):
        summary.source_rows_scanned += len(chunk)
        source_valid = present_mask(chunk[dataset.source])
        summary.sources.update(chunk.loc[source_valid, dataset.source].astype(str).tolist())
        if dataset.definition.key == "open_air":
            update_open_air_pair_checks(chunk, summary)

        for measurement in dataset.measurement_columns:
            value_valid = present_mask(chunk[measurement])
            summary.measurement_non_null_counts[measurement] += int(value_valid.sum())
            if dataset.definition.key == "beach_water":
                summary.sentinel_counts[measurement] += int(
                    chunk[measurement].isin(SENTINELS).fillna(False).sum()
                )
            eligible = source_valid & value_valid
            candidate_count = int(eligible.sum())
            summary.candidate_combinations += candidate_count
            if not candidate_count:
                continue
            selection = [dataset.source]
            if dataset.second_source:
                selection.append(dataset.second_source)
            observed = chunk.loc[eligible, selection].drop_duplicates(subset=[dataset.source])
            for row in observed.itertuples(index=False, name=None):
                source = str(row[0])
                key = (source, measurement)
                record = summary.unique_records.get(key)
                if record is None:
                    record = base_record(dataset, source, measurement)
                    summary.unique_records[key] = record
                if dataset.second_source:
                    sensor_name = native_text(row[1])
                    if sensor_name and not record.get("sensor_name"):
                        record["sensor_name"] = sensor_name
                        record["text"] = (
                            f"sensor_name: {sensor_name} | measurement_key: {measurement}"
                        )
        print(
            f"  {dataset.definition.key}: chunk {chunk_number:,}; "
            f"rows={summary.source_rows_scanned:,}; unique_topics={len(summary.unique_records):,}"
        )


def scan_sgim(dataset: Resolved, summary: Summary, chunk_size: int) -> None:
    assert dataset.measurement is not None
    for chunk_number, chunk in enumerate(iter_chunks(dataset, chunk_size), start=1):
        summary.source_rows_scanned += len(chunk)
        source_valid = present_mask(chunk[dataset.source])
        measurement_valid = present_mask(chunk[dataset.measurement])
        eligible = source_valid & measurement_valid
        summary.sources.update(chunk.loc[source_valid, dataset.source].astype(str).tolist())
        summary.measurement_non_null_counts[dataset.measurement] += int(measurement_valid.sum())
        summary.candidate_combinations += int(eligible.sum())
        valid = chunk.loc[eligible]
        if not valid.empty:
            new_pairs = valid.drop_duplicates(subset=[dataset.source, dataset.measurement])
            for source_value, measurement_value in new_pairs[
                [dataset.source, dataset.measurement]
            ].itertuples(index=False, name=None):
                source = str(source_value)
                measurement = str(measurement_value)
                key = (source, measurement)
                summary.unique_records.setdefault(key, base_record(dataset, source, measurement))

            for output_key, actual_column in dataset.stable_text_columns:
                field_valid = present_mask(valid[actual_column])
                field_rows = valid.loc[
                    field_valid, [dataset.source, dataset.measurement, actual_column]
                ].drop_duplicates(subset=[dataset.source, dataset.measurement])
                for source_value, measurement_value, field_value in field_rows.itertuples(
                    index=False, name=None
                ):
                    record = summary.unique_records[(str(source_value), str(measurement_value))]
                    if output_key not in record:
                        text_value = native_text(field_value)
                        if text_value:
                            record[output_key] = text_value
        print(
            f"  sgim: chunk {chunk_number:,}; rows={summary.source_rows_scanned:,}; "
            f"unique_topics={len(summary.unique_records):,}"
        )

    fixed_order = (
        "measurement_key",
        "measurement_title",
        "measurement_description",
        "measurement_medium",
        "units",
        "units_abbreviation",
        "measurement_period_type",
    )
    text_labels = {"measurement_key": "measurement_type"}
    for record in summary.unique_records.values():
        parts = []
        for key in fixed_order:
            value = native_text(record.get(key))
            if value:
                parts.append(f"{text_labels.get(key, key)}: {value}")
        record["text"] = " | ".join(parts)


def validate_records(records: Sequence[dict[str, Any]]) -> None:
    topics: set[str] = set()
    forbidden_keys = {
        "time",
        "measurement_time",
        "measurement_value",
        "value",
        "timestamp",
        "tags",
        "record_id",
        "measurement_id",
        "resource_id",
        "latitude",
        "longitude",
        "location",
    }
    for record in records:
        if not all(key in record for key in ("topic", "measurement_key", "text")):
            raise ValueError(f"Record lacks a required key: {record}")
        if not str(record["text"]).strip():
            raise ValueError(f"Record has empty text: {record}")
        topic = str(record["topic"])
        if topic in topics:
            raise ValueError(f"Duplicate topic: {topic}")
        topics.add(topic)
        prohibited = forbidden_keys.intersection(record)
        if prohibited:
            raise ValueError(f"Record contains prohibited keys {sorted(prohibited)}: {record}")
        if any(isinstance(value, (dict, list)) for value in record.values()):
            raise ValueError(f"Record contains nested data: {record}")
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
        json.loads(encoded)


def write_output(path: Path, records: Sequence[dict[str, Any]], overwrite: bool) -> None:
    print(f"Output path: {path.resolve()}")
    if path.exists():
        print(f"WARNING: output already exists: {path.resolve()}")
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def print_mapping(dataset: Resolved) -> None:
    print(f"\n=== {dataset.definition.key}: inferred mapping ===")
    print(f"Source CSV path: {dataset.path.resolve()}")
    mapping = {
        "source": dataset.source,
        "second_source": dataset.second_source,
        "timestamp_excluded": dataset.timestamp,
        "administrative_excluded": list(dataset.administrative),
        "location_excluded": list(dataset.locations),
        "measurement": dataset.measurement,
        "measurement_columns": list(dataset.measurement_columns),
        "stable_text_columns": dict(dataset.stable_text_columns),
    }
    print(json.dumps(mapping, ensure_ascii=False, indent=2))


def print_summary(summary: Summary) -> None:
    records = list(summary.unique_records.values())
    entirely_null = [
        column for column in summary.measurement_columns if summary.measurement_non_null_counts[column] == 0
    ]
    print(f"\n=== {summary.dataset}: final summary ===")
    print(f"Source CSV path: {summary.source_path.resolve()}")
    print(f"Source rows scanned: {summary.source_rows_scanned:,}")
    print(f"Source count: {len(summary.sources):,}")
    print(f"Measurement-column count: {len(summary.measurement_columns):,}")
    print(f"Candidate combinations found: {summary.candidate_combinations:,}")
    print(f"Duplicate combinations skipped: {summary.duplicate_combinations_skipped:,}")
    print(f"Unique topic count: {len(records):,}")
    print(f"Entirely-null measurement columns: {entirely_null}")
    print(f"Sentinel counts by column: {dict(summary.sentinel_counts)}")
    if summary.dataset == "open_air":
        topics_per_sensor = Counter(
            str(record["datasourceid"]) for record in records
        )
        counts = sorted(topics_per_sensor.values())
        median = float(pd.Series(counts).median()) if counts else 0.0
        print(f"Minimum topics per sensor: {min(counts) if counts else 0}")
        print(f"Maximum topics per sensor: {max(counts) if counts else 0}")
        print(f"Median topics per sensor: {median}")
        suffix_counts = Counter(
            str(record["measurement_key"])
            for record in records
            if str(record["measurement_key"]).endswith("_1")
        )
        print(f"Counts for _1 columns: {dict(suffix_counts)}")
        print("_1 pair exact-duplicate results across scanned rows:")
        for pair in OPEN_AIR_DUPLICATE_PAIRS:
            print(
                f"- {pair[0]} vs {pair[1]}: exact={summary.duplicate_pair_status.get(pair, False)}, "
                f"both_non_null={summary.duplicate_pair_both_non_null[pair]:,}, "
                f"equal={summary.duplicate_pair_equal[pair]:,}"
            )
    print("First 5 output records:")
    for record in records[:5]:
        print(json.dumps(record, ensure_ascii=False))
    print(f"Output path: {summary.output_path.resolve()}")
    print(f"Output size: {summary.output_size:,} bytes")
    print(f"Input unchanged: {'PASS' if summary.input_unchanged else 'FAIL'}")
    print("Output validation: PASS")


def process(dataset: Resolved, output_path: Path, chunk_size: int, overwrite: bool) -> Summary:
    print_mapping(dataset)
    if output_path.exists() and not overwrite:
        print(f"WARNING: output already exists: {output_path.resolve()}")
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    before = dataset.path.stat()
    summary = Summary(
        dataset=dataset.definition.key,
        source_path=dataset.path,
        output_path=output_path,
        measurement_columns=list(dataset.measurement_columns),
        duplicate_pair_status={pair: True for pair in OPEN_AIR_DUPLICATE_PAIRS},
        input_size=before.st_size,
        input_mtime_ns=before.st_mtime_ns,
    )
    if dataset.definition.key == "sgim":
        free = shutil.disk_usage(output_path.parent).free
        print(f"SGIM input size: {before.st_size:,} bytes")
        print(f"Current free disk space: {free:,} bytes")
        print("SGIM extraction output is bounded by unique source/type pairs, not source row count.")
        scan_sgim(dataset, summary, chunk_size)
    else:
        scan_wide(dataset, summary, chunk_size)

    records = list(summary.unique_records.values())
    validate_records(records)
    write_output(output_path, records, overwrite)
    after = dataset.path.stat()
    summary.input_unchanged = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    summary.output_size = output_path.stat().st_size
    print_summary(summary)
    return summary


def run(args: argparse.Namespace) -> None:
    root = project_root()
    keys = DATASET_ORDER if args.dataset == "all" else (args.dataset,)
    resolved = [resolve_dataset(root, DEFINITIONS[key]) for key in keys]
    print("=== Exact selected inputs ===")
    for dataset in resolved:
        print(f"{dataset.definition.key}: {dataset.path.resolve()}")
    for dataset in resolved:
        output_path = root / "data" / "processed" / dataset.definition.output_name
        process(dataset, output_path, args.chunk_size, args.overwrite)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
