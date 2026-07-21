# SmartMQTT unique-topic extraction

This project extracts one compact text record per unique logical IoT topic from
four Chicago sensor datasets. Raw CSV inputs are read from `data/raw/` in chunks.

## Requirements

- Python 3.10+
- [pandas](https://pandas.pydata.org/) &mdash; data loading and chunked CSV processing
- [streamlit](https://streamlit.io/) >= 1.55 &mdash; local clustering explorer
- [sentence-transformers](https://www.sbert.net/) &mdash; CPU sentence embeddings
- [scikit-learn](https://scikit-learn.org/) &mdash; DBSCAN and K-means
- [numpy](https://numpy.org/) &mdash; normalized matrices and cache files
- [plotly](https://plotly.com/python/) &mdash; controlled PCA and heatmap views
- [pytest](https://docs.pytest.org/) &mdash; running the test suite

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
metadata representations affect unsupervised clustering of the same logical IoT
topics. This is clustering, not classification. The real-data experiment has no
ground-truth labels and does not calculate real-data accuracy. The generated
controlled benchmark contains expected groups used only for post-clustering
diagnostics; they are never embedded or passed to clustering. Embedding
similarity does not prove scientific equivalence, and synthetic metrics do not
represent real-data accuracy.

### Metadata representation approaches

The experiment keeps topic order, model, normalization, and clustering
parameters fixed while changing representation or vector weighting:

- `VALUE_ONLY`: metadata values without field names.
- `KEY_ONLY`: field names without values; this is a structural control, so
  repeated texts and large similarity-tie clusters are expected.
- `KEY_VALUE`: field names and values in the fixed dataset-specific order.
- `NORMALIZED_KEY_VALUE`: `KEY_VALUE` with lowercase conversion,
  underscore-to-space conversion, whitespace collapse, and trimming only. It
  does not split camelCase, expand abbreviations, map synonyms, or normalize
  ontology terms.
- `WEIGHTED_KEY_VALUE`: separately generated `KEY_ONLY` and `VALUE_ONLY`
  embeddings combined as:

  ```text
  weighted_embedding =
  L2_normalize(
      key_weight * key_embedding
      +
      value_weight * value_embedding
  )
  ```

Weighting is performed on embedding vectors; text is never repeated to simulate
weight. The weights sum to 100%. Presets are 10/90, 30/70, 50/50, 70/30, and
90/10 key/value, plus a custom key-weight slider. A 10/90 setting emphasizes
metadata content, 30/70 gives moderate schema influence, 50/50 treats schema and
values equally, 70/30 emphasizes schema, and 90/10 is primarily structural.
Changing weights reuses cached `KEY_ONLY` and `VALUE_ONLY` embeddings, reruns
clustering, and does not invoke the transformer again. Weighting creates no
semantic ground truth and does not prove scientific equivalence.

The supported models are `sentence-transformers/all-MiniLM-L6-v2` (default) and
`sentence-transformers/all-mpnet-base-v2`, both run on CPU by default. Base
normalized embedding matrices are cached under
`embedding_cache/{safe_model_name}/`. Each cache stores aligned topics,
datasets, measurement keys, source values, exact representation text, model,
dimension, text hash, ordered-topic hash, and normalized status. Caches are
atomically replaced when model, representation, text hash, ordered topic hash,
record count, or dimension differs. `WEIGHTED_KEY_VALUE` is always generated in
memory and never persisted as a weight-specific NPZ.

DBSCAN is the primary method. The UI uses cosine distance and
`eps = 1 - similarity_threshold`; a high threshold generally produces stricter,
smaller groups and more noise, while a low threshold generally produces broader
groups and less noise. Label `-1` is noise. K-means requires an explicit fixed
K and is included only as a baseline. The dataset mode can combine selected
datasets so cross-dataset groups can emerge, or cluster each selected dataset
separately to evaluate within-dataset structure.

The explorer reads the four processed JSONL files in stable order and displays
one tab for each selected representation. It reports active parameters, cache
status, cluster summaries, centroid-nearest representatives, optional full
membership, noise, selected-topic comparisons, and parameterized CSV downloads.

### Environment setup

Create and activate the project-local environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run tests with a project-local temporary directory:

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMP = $env:TEMP
python -m pytest -q --basetemp="$PWD\.tmp\pytest"
```

Launch Streamlit directly with:

```powershell
python -m streamlit run app/embedding_explorer.py --server.fileWatcherType none
```

The disabled file watcher avoids irrelevant optional Transformers vision-module
imports; torchvision is not required for this text-embedding experiment. You
can also use the launcher:

```powershell
.\scripts\run_embedding_ui.ps1
```

## Generated controlled benchmark

The explorer also includes a smaller deterministic diagnostic benchmark. Real
records are numerous and contain repeated terminology, so this controlled case
makes representation and threshold behavior easier to inspect manually without
replacing conclusions from the real datasets.

The benchmark contains exactly 34 records constructed programmatically in memory:
six semantic groups with five variants each—air temperature, water temperature,
relative humidity, wind speed, PM2.5 concentration, and water turbidity—plus
four unrelated outliers for battery voltage, door-open state, vibration
amplitude, and soil conductivity. It does not call an external text-generation
service and is never written to `data/processed/` or `data/raw/`.

The sidebar's **Experiment data** control supports **Real datasets** (the four
processed files), **Generated benchmark** (only the 34 in-memory records), and
**Real + generated benchmark** (the 2,969 real records followed by the 34
synthetic records). Dataset filtering and cache identities keep these sources
separate.

`expected_group`, `expected_evaluation_label`, and `variant_type` are evaluation
metadata only. They are displayed after clustering, never embedded, and never
passed to clustering. Controlled metrics are not real-data accuracy. For
DBSCAN, noise remains label `-1`; ARI and NMI treat all noise as one explicit
discovered label and report noise separately, while purity evaluates discovered
non-noise records only.

The generated-only view provides controlled benchmark metrics, deterministic
two-dimensional PCA plots, and a cosine-similarity heatmap. PCA is for
visualization only; clustering uses original full-dimensional normalized
embeddings. The heatmap is shown because the benchmark has only 34 records, and
exact cluster counts are not guaranteed.

The five approaches have these diagnostic purposes:

- `VALUE_ONLY`: semantic metadata values without field names.
- `KEY_ONLY`: shared schema names only; an intentional structural control.
- `KEY_VALUE`: field names plus metadata values.
- `NORMALIZED_KEY_VALUE`: `KEY_VALUE` with lowercase, underscore replacement,
  whitespace collapse, and trimming only.
- `WEIGHTED_KEY_VALUE`: a normalized weighted combination of separately encoded
  `KEY_ONLY` and `VALUE_ONLY` vectors.

```text
weighted_embedding =
L2_normalize(
    key_weight * key_embedding
    +
    value_weight * value_embedding
)
```

Higher value weight is intended to emphasize metadata content and higher key
weight is intended to emphasize schema structure, but neither guarantees an
exact cluster outcome.
