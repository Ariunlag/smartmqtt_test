# SmartMQTT unique-topic extraction

This project extracts one compact text record per unique logical IoT topic from
four Chicago sensor datasets. Raw CSV inputs are read from `data/raw/` in chunks.

## Requirements

- Python 3.10+
- [pandas](https://pandas.pydata.org/) >= 2.0.0 &mdash; data loading and chunked CSV processing
- [transformers](https://huggingface.co/docs/transformers) >= 4.30.0 &mdash; optional HuggingFace utility support
- [pytest](https://docs.pytest.org/) >= 7.0.0 &mdash; running the test suite

## Installation

1. **Clone the repository** (if you haven't already):

   ```powershell
   git clone <repo-url>
   cd SmartMQTT_RQ1
   ```

2. **Create and activate a virtual environment** (recommended):

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```

3. **Install the required packages**:

   ```powershell
   pip install -r requirements.txt
   ```

4. **Place raw CSV datasets** in `data/raw/`. The raw files are Git-ignored, so
   you will need to download them separately (e.g., from the Chicago Data
   Portal).

5. **Run the extraction** (see [Reproduce](#reproduce) below).

## Outputs

- `data/processed/01_sgim_topic_texts.jsonl`
- `data/processed/02_beach_weather_topic_texts.jsonl`
- `data/processed/03_beach_water_topic_texts.jsonl`
- `data/processed/04_open_air_topic_texts.jsonl`

Each JSONL line describes one unique logical topic using stable textual metadata
only. Timestamped telemetry values are not included, and the files contain no
nested tag objects.

Wide datasets use each source plus every observed, non-null measurement column.
SGIM is already in long format and uses `data_stream_id + measurement_type`.
For both beach datasets, `Measurement Timestamp Label` is excluded because it is
timestamp metadata rather than a sensor measurement.

## Reproduce

Run all datasets:

```powershell
python scripts/extract_unique_topic_texts.py
```

Run Beach Weather only:

```powershell
python scripts/extract_unique_topic_texts.py --dataset beach_weather
```

Replace the existing Beach Water output:

```powershell
python scripts/extract_unique_topic_texts.py --dataset beach_water --overwrite
```

The original CSV files are ignored by Git and remain unchanged. Existing output
files are protected by default; pass `--overwrite` when replacement is intended.

## Local embedding clustering explorer

The one-page Streamlit application in `app/embedding_explorer.py` compares how
four deterministic metadata-text representations affect unsupervised clustering
of the same logical topics. This is clustering, not classification, and it does
not calculate accuracy or treat `measurement_key` as ground truth.

The experiment holds the record order, embedding model, L2 normalization, and
clustering parameters constant while comparing:

- `VALUE_ONLY`;
- `KEY_ONLY`;
- `KEY_VALUE`;
- `NORMALIZED_KEY_VALUE`.

Dataset-specific metadata fields retain their fixed order. Normalization is
deterministic: text is lowercased, underscores become spaces, and repeated
whitespace is collapsed. It does not infer synonyms or scientific meanings.

Every strategy uses `sentence-transformers/all-MiniLM-L6-v2` on CPU. Its
normalized embedding matrix and aligned metadata are persisted in ignored NPZ
files under `embedding_cache/`. A cache is reused only when both its model name
and representation-text hash match the current input; otherwise it is rebuilt.

DBSCAN with cosine distance is the default clustering method. The UI converts
the selected similarity threshold to DBSCAN distance with
`eps = 1 - similarity_threshold`, uses label `-1` for noise, and lets the number
of clusters emerge from the threshold. K-means is available only as an explicit
fixed-k baseline.

The explorer reads:

- `data/processed/01_sgim_topic_texts.jsonl`
- `data/processed/02_beach_weather_topic_texts.jsonl`
- `data/processed/03_beach_water_topic_texts.jsonl`
- `data/processed/04_open_air_topic_texts.jsonl`

### Environment setup

Create and activate the project-local environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `.venv` already exists, activate it and install or update the requirements.

Run tests with a project-local temporary directory:

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMP = $env:TEMP
python -m pytest -q --basetemp="$PWD\.tmp\pytest"
```

Launch Streamlit directly:

```powershell
python -m streamlit run app/embedding_explorer.py
```

Alternatively, use the launcher, which activates `.venv` and configures the
project-local temporary directory automatically:

```powershell
.\scripts\run_embedding_ui.ps1
```

Choose datasets and clustering parameters in the sidebar and press **Run
clustering**. The primary comparison table reports cluster and noise behavior
across all four representations. Each representation tab shows clusters ordered
by size, centroid-nearest representative topics, expandable full membership,
separate noise records, and a lazily generated CSV download.
