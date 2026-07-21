"""Small deterministic in-memory benchmark for inspecting clustering behavior."""

from __future__ import annotations


VARIANT_TYPES = (
    "canonical",
    "synonym",
    "formatting_variant",
    "descriptive_variant",
    "cross_source_variant",
)


def _record(
    group: str,
    variant: str,
    topic_suffix: str,
    source_name: str,
    measurement_key: str,
    description: str,
    unit: str,
) -> dict[str, str]:
    text = (
        f"source_name: {source_name} | measurement_key: {measurement_key} | "
        f"measurement_description: {description} | unit: {unit}"
    )
    return {
        "topic": f"synthetic/{group}/{topic_suffix}",
        "dataset": "synthetic",
        "source_name": source_name,
        "measurement_key": measurement_key,
        "measurement_description": description,
        "unit": unit,
        "expected_group": group,
        "variant_type": variant,
        "text": text,
    }


def build_synthetic_records() -> list[dict[str, str]]:
    """Return exactly 34 stable records: six five-topic groups and four outliers."""

    groups = [
        (
            "air_temperature",
            "degrees Celsius",
            (
                ("canonical", "Rooftop North", "air_temperature", "temperature of ambient air"),
                ("synonym", "Weather Yard", "ambient air temperature", "temperature describing outdoor atmosphere"),
                ("formatting_variant", "Field Station 7", "AIR_TEMPERATURE", "ambient  air temperature in weather conditions"),
                ("descriptive_variant", "Park Edge", "outdoor_temp", "temperature measured in the outdoor atmosphere"),
                ("cross_source_variant", "Garden South", "temperature measured in outdoor air", "ambient air conditions outside the building"),
            ),
        ),
        (
            "water_temperature",
            "degrees Celsius",
            (
                ("canonical", "Lake Pier", "water_temperature", "temperature of sampled lake water"),
                ("synonym", "Beach Buoy", "lake water temperature", "aquatic temperature during beach sampling"),
                ("formatting_variant", "Harbor Probe", "WATER_TEMPERATURE", "water  temperature under aquatic conditions"),
                ("descriptive_variant", "River Intake", "beach_water_temp", "temperature measured in sampled water"),
                ("cross_source_variant", "South Shore", "temperature of sampled water", "lake and beach water temperature observation"),
            ),
        ),
        (
            "relative_humidity",
            "percent",
            (
                ("canonical", "Rooftop North", "relative_humidity", "relative moisture of outdoor air"),
                ("synonym", "Weather Yard", "air moisture percentage", "percentage of water vapor in ambient air"),
                ("formatting_variant", "Field Station 7", "RELATIVE_HUMIDITY", "outdoor  air moisture percentage"),
                ("descriptive_variant", "Park Edge", "ambient humidity", "humidity conditions surrounding the weather station"),
                ("cross_source_variant", "Garden South", "moisture content of outdoor air", "relative moisture in the outdoor atmosphere"),
            ),
        ),
        (
            "wind_speed",
            "meters per second",
            (
                ("canonical", "Rooftop North", "wind_speed", "speed of moving outdoor air"),
                ("synonym", "Weather Yard", "air flow velocity", "velocity of wind moving across the site"),
                ("formatting_variant", "Field Station 7", "WIND_SPEED", "ambient  wind velocity in the atmosphere"),
                ("descriptive_variant", "Park Edge", "ambient wind velocity", "rate of moving air during weather conditions"),
                ("cross_source_variant", "Garden South", "speed of moving outdoor air", "wind movement through the outdoor atmosphere"),
            ),
        ),
        (
            "pm25_concentration",
            "micrograms per cubic meter",
            (
                ("canonical", "Rooftop North", "pm2_5_concentration", "mass concentration of airborne fine particles"),
                ("synonym", "Traffic Corner", "fine particulate concentration", "airborne PM2.5 particle mass in air"),
                ("formatting_variant", "Field Station 7", "PM2_5_CONCENTRATION", "airborne  fine particle concentration"),
                ("descriptive_variant", "Park Edge", "airborne PM2.5 mass", "concentration of fine particles suspended in air"),
                ("cross_source_variant", "Industrial Perimeter", "concentration of fine particles in air", "mass of airborne particulate matter"),
            ),
        ),
        (
            "water_turbidity",
            "NTU",
            (
                ("canonical", "Lake Pier", "water_turbidity", "cloudiness caused by suspended material in water"),
                ("synonym", "Beach Buoy", "water cloudiness", "optical cloudiness of sampled aquatic water"),
                ("formatting_variant", "Harbor Probe", "WATER_TURBIDITY", "water  turbidity from suspended particles"),
                ("descriptive_variant", "River Intake", "suspended particle turbidity", "optical clarity affected by material in water"),
                ("cross_source_variant", "South Shore", "optical clarity of sampled water", "suspended matter changing water clarity"),
            ),
        ),
    ]
    records: list[dict[str, str]] = []
    for group, unit, variants in groups:
        for index, (variant, source, key, description) in enumerate(variants, start=1):
            records.append(_record(group, variant, f"{index:02d}", source, key, description, unit))

    outliers = (
        ("battery_voltage", "Battery Cabinet", "battery_voltage", "electrical potential of a storage battery", "volts"),
        ("door_open_state", "North Entrance", "door_open_state", "whether a physical door is open or closed", "boolean"),
        ("vibration_amplitude", "Pump Housing", "vibration_amplitude", "mechanical vibration magnitude of rotating equipment", "millimeters per second"),
        ("soil_conductivity", "Garden Plot", "soil_conductivity", "electrical conductivity of soil material", "microsiemens per centimeter"),
    )
    for group, source, key, description, unit in outliers:
        records.append(_record(group, "outlier", "01", source, key, description, unit))
        records[-1]["expected_group"] = "outlier"
        records[-1]["expected_evaluation_label"] = f"outlier_{key}"
    for row in records:
        row.setdefault("expected_evaluation_label", row["expected_group"])
    return records


def expected_evaluation_label(record: dict[str, str]) -> str:
    """Return a distinct evaluation label for each unrelated outlier."""

    if record["expected_group"] != "outlier":
        return record["expected_group"]
    return f"outlier_{record['measurement_key']}"


def synthetic_group_counts(records: list[dict[str, str]] | None = None) -> dict[str, int]:
    rows = records if records is not None else build_synthetic_records()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["expected_group"]] = counts.get(row["expected_group"], 0) + 1
    return counts
