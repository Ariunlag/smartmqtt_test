"""Streamlit interface for the configurable SmartMQTT clustering experiment."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.embedding_engine import (
    BASE_REPRESENTATIONS,
    DATASET_MODES,
    DEFAULT_CLUSTERING_METHOD,
    KMEANS_METHOD,
    KEY_ONLY,
    MODEL_NAME,
    MODEL_OPTIONS,
    NORMALIZED_KEY_VALUE,
    REPRESENTATIONS,
    REPRESENTATION_LABELS,
    VALUE_ONLY,
    WEIGHTED_KEY_VALUE,
    ClusteringParameters,
    EmbeddingBundle,
    build_weighted_bundle,
    cluster_all_strategies,
    cluster_neighbors,
    cluster_summary,
    controlled_benchmark_metrics,
    cosine_similarity_matrix,
    construct_representation,
    filter_bundle,
    get_or_create_embedding_cache,
    load_jsonl_records,
    load_sentence_transformer,
    ordered_topic_hash,
    pca_coordinates,
    ranked_cluster_members,
    representative_topics,
    safe_model_name,
    similarity_to_cluster_representatives,
    threshold_to_eps,
)
from app.synthetic_benchmark import build_synthetic_records, expected_evaluation_label, synthetic_group_counts
from app.experiment_config import (
    DATA_SOURCE_OPTIONS,
    GENERATED_DATA,
    MIXED_DATA,
    REAL_DATA,
    clustering_parameter_limits,
    load_experiment_source,
    validate_clustering_parameters,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_INPUT_FILES = (
    ROOT / "data" / "processed" / "01_sgim_topic_texts.jsonl",
    ROOT / "data" / "processed" / "02_beach_weather_topic_texts.jsonl",
    ROOT / "data" / "processed" / "03_beach_water_topic_texts.jsonl",
    ROOT / "data" / "processed" / "04_open_air_topic_texts.jsonl",
)
DATASET_NAMES = ("sgim", "beach_weather", "beach_water", "open_air")
REQUIRED_BASES = {
    VALUE_ONLY: (VALUE_ONLY,),
    KEY_ONLY: (KEY_ONLY,),
    "KEY_VALUE": ("KEY_VALUE",),
    NORMALIZED_KEY_VALUE: (NORMALIZED_KEY_VALUE,),
    WEIGHTED_KEY_VALUE: (KEY_ONLY, VALUE_ONLY),
}
WEIGHT_PRESETS = {
    "10% key / 90% value": (0.10, 0.90),
    "30% key / 70% value": (0.30, 0.70),
    "50% key / 50% value": (0.50, 0.50),
    "70% key / 30% value": (0.70, 0.30),
    "90% key / 10% value": (0.90, 0.10),
}


@st.cache_resource(show_spinner=False)
def cached_model(model_name: str) -> Any:
    return load_sentence_transformer(model_name, device="cpu")


@st.cache_data(show_spinner=False)
def cached_records(signatures: tuple[tuple[str, int, int], ...]) -> list[dict[str, Any]]:
    return load_jsonl_records(Path(path) for path, _, _ in signatures)


def cached_base_bundle(
    records: Sequence[Mapping[str, Any]], model_name: str, strategy: str, data_source: str
) -> EmbeddingBundle:
    source_slug = {REAL_DATA: "real", GENERATED_DATA: "synthetic", MIXED_DATA: "mixed"}[data_source]
    cache_path = ROOT / "embedding_cache" / safe_model_name(model_name) / source_slug / f"{strategy.lower()}.npz"
    return get_or_create_embedding_cache(
        cache_path, records, strategy, model_name, lambda: cached_model(model_name)
    )


def file_signatures() -> tuple[tuple[str, int, int], ...]:
    signatures = []
    for path in REAL_INPUT_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Required input file is missing: {path}")
        stat = path.stat()
        signatures.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(signatures)


@st.cache_data(show_spinner=False)
def cached_synthetic_records() -> list[dict[str, str]]:
    return build_synthetic_records()


def required_bases(selected_representations: Sequence[str]) -> tuple[str, ...]:
    needed: list[str] = []
    for representation in selected_representations:
        for base in REQUIRED_BASES[representation]:
            if base not in needed:
                needed.append(base)
    return tuple(base for base in BASE_REPRESENTATIONS if base in needed)


def display_texts(
    bundle: EmbeddingBundle,
    index: int,
    key_bundle: EmbeddingBundle | None = None,
    value_bundle: EmbeddingBundle | None = None,
) -> dict[str, str]:
    result = {"representation_text": str(bundle.texts[index])}
    if key_bundle is not None and value_bundle is not None:
        result["key_component_text"] = str(key_bundle.texts[index])
        result["value_component_text"] = str(value_bundle.texts[index])
    else:
        result["key_component_text"] = ""
        result["value_component_text"] = ""
    return result


def topic_rows(
    bundle: EmbeddingBundle,
    ranked: Sequence[tuple[int, float]],
    *,
    key_bundle: EmbeddingBundle | None = None,
    value_bundle: EmbeddingBundle | None = None,
    key_weight: float | None = None,
    value_weight: float | None = None,
) -> pd.DataFrame:
    rows = []
    for index, similarity in ranked:
        row = {
            "topic": str(bundle.topics[index]),
            "dataset": str(bundle.datasets[index]),
            "measurement_key": str(bundle.measurement_keys[index]),
            "source": str(bundle.sources[index]),
            "exact embedded text": str(bundle.texts[index]),
            "similarity to cluster centroid": similarity,
        }
        if bundle.expected_groups is not None and str(bundle.expected_groups[index]):
            row["expected_group"] = str(bundle.expected_groups[index])
            row["expected_evaluation_label"] = str(bundle.evaluation_labels[index]) if bundle.evaluation_labels is not None else ""
            row["variant_type"] = str(bundle.variant_types[index]) if bundle.variant_types is not None else ""
        if key_bundle is not None and value_bundle is not None:
            row.update(
                {
                    "key component text": str(key_bundle.texts[index]),
                    "value component text": str(value_bundle.texts[index]),
                    "key weight": key_weight,
                    "value weight": value_weight,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_export_csv(
    strategy: str,
    bundle: EmbeddingBundle,
    labels: np.ndarray,
    *,
    selected_datasets: Sequence[str],
    data_source: str,
    dataset_mode: str,
    method: str,
    similarity_threshold: float | None,
    min_samples: int | None,
    k: int | None,
    key_weight: float | None = None,
    value_weight: float | None = None,
    key_bundle: EmbeddingBundle | None = None,
    value_bundle: EmbeddingBundle | None = None,
) -> bytes:
    similarities = similarity_to_cluster_representatives(bundle.embeddings, labels)
    threshold = similarity_threshold if method == DEFAULT_CLUSTERING_METHOD else None
    eps = threshold_to_eps(threshold) if threshold is not None else None
    key_texts = key_bundle.texts if key_bundle is not None else [""] * len(bundle.topics)
    value_texts = value_bundle.texts if value_bundle is not None else [""] * len(bundle.topics)
    frame = pd.DataFrame(
        {
            "model": bundle.model_name,
            "data_source": data_source,
            "representation": strategy,
            "selected_datasets": ",".join(selected_datasets),
            "dataset_mode": dataset_mode,
            "clustering_method": method,
            "similarity_threshold": threshold,
            "eps": eps,
            "min_samples": min_samples if method == DEFAULT_CLUSTERING_METHOD else None,
            "k": k if method == KMEANS_METHOD else None,
            "key_weight": key_weight if strategy == WEIGHTED_KEY_VALUE else None,
            "value_weight": value_weight if strategy == WEIGHTED_KEY_VALUE else None,
            "cluster_id": labels,
            "is_noise": labels == -1,
            "topic": bundle.topics,
            "dataset": bundle.datasets,
            "source": bundle.sources,
            "measurement_key": bundle.measurement_keys,
            "representation_text": bundle.texts,
            "key_component_text": key_texts,
            "value_component_text": value_texts,
            "similarity_to_cluster_centroid": similarities,
        }
    )
    if bundle.expected_groups is not None:
        frame["expected_group"] = bundle.expected_groups
        frame["expected_evaluation_label"] = bundle.evaluation_labels if bundle.evaluation_labels is not None else ""
        frame["variant_type"] = bundle.variant_types if bundle.variant_types is not None else ""
    else:
        frame["expected_group"] = ""
        frame["expected_evaluation_label"] = ""
        frame["variant_type"] = ""
    return frame.to_csv(index=False).encode("utf-8")


def cluster_rows(
    bundle: EmbeddingBundle,
    labels: np.ndarray,
    *,
    show_all_members: bool,
    representative_limit: int,
    include_noise: bool,
    method: str,
    key_bundle: EmbeddingBundle | None,
    value_bundle: EmbeddingBundle | None,
    key_weight: float | None,
    value_weight: float | None,
    export_kwargs: dict[str, Any],
) -> None:
    counts = Counter(int(label) for label in labels if int(label) != -1)
    ordered_clusters = sorted(counts, key=lambda cluster_id: (-counts[cluster_id], cluster_id))
    if bundle.representation == KEY_ONLY and len(set(bundle.texts.tolist())) < len(bundle.texts):
        st.warning(
            "This representation contains repeated or identical texts. Similarity ties and large clusters are expected."
        )
    ready_key = f"csv-ready-{bundle.representation}-{bundle.text_hash}-{export_kwargs.get('dataset_name', 'combined')}"
    if st.button(
        f"Prepare {REPRESENTATION_LABELS[bundle.representation]} CSV",
        key=f"prepare-{ready_key}",
        icon=":material/download:",
    ):
        st.session_state[ready_key] = True
    if st.session_state.get(ready_key, False):
        st.download_button(
            f"Download {REPRESENTATION_LABELS[bundle.representation]} assignments",
            data=cluster_export_csv(
                bundle.representation,
                bundle,
                labels,
                selected_datasets=export_kwargs["selected_datasets"],
                data_source=export_kwargs["data_source"],
                dataset_mode=export_kwargs["dataset_mode"],
                method=export_kwargs["method"],
                similarity_threshold=export_kwargs["similarity_threshold"],
                min_samples=export_kwargs["min_samples"],
                k=export_kwargs["k"],
                key_weight=key_weight,
                value_weight=value_weight,
                key_bundle=key_bundle,
                value_bundle=value_bundle,
            ),
            file_name=f"{bundle.representation.lower()}_clusters.csv",
            mime="text/csv",
            key=f"download-{ready_key}",
            icon=":material/download:",
        )
    if not ordered_clusters:
        st.info("No non-noise clusters emerged from the selected parameters.")
    for cluster_id in ordered_clusters:
        member_indices = np.flatnonzero(labels == cluster_id)
        member_keys = bundle.measurement_keys[member_indices].tolist()
        dominant = Counter(member_keys).most_common(5)
        ranked_all = ranked_cluster_members(bundle.embeddings, labels, cluster_id)
        mean_similarity = float(np.mean([score for _, score in ranked_all]))
        expected_groups = (
            [str(value) for value in bundle.expected_groups[member_indices].tolist() if str(value)]
            if bundle.expected_groups is not None
            else []
        )
        variant_types = (
            [str(value) for value in bundle.variant_types[member_indices].tolist() if str(value)]
            if bundle.variant_types is not None
            else []
        )
        with st.container(border=True):
            st.subheader(f"Cluster {cluster_id} · {len(member_indices):,} topics")
            st.write(
                f"**Datasets:** {', '.join(sorted(set(bundle.datasets[member_indices])))}  \n"
                f"**Unique measurement keys:** {len(set(member_keys)):,}  \n"
                f"**Most common measurement keys:** "
                + ", ".join(f"{key} ({count})" for key, count in dominant)
                + f"  \n**Mean similarity to normalized cluster centroid:** {mean_similarity:.6f}"
            )
            if expected_groups:
                st.write(
                    "**Expected groups represented:** " + ", ".join(sorted(set(expected_groups))) + "  \n"
                    "**Dominant expected group:** " + Counter(expected_groups).most_common(1)[0][0] + "  \n"
                    "**Expected-group composition:** " + ", ".join(f"{group}: {count}" for group, count in Counter(expected_groups).most_common()) + "  \n"
                    "**Variant types represented:** " + ", ".join(sorted(set(variant_types)))
                )
            st.markdown(f"**Top {min(representative_limit, len(member_indices))} representatives**")
            st.dataframe(
                topic_rows(
                    bundle,
                    representative_topics(bundle.embeddings, labels, cluster_id, representative_limit),
                    key_bundle=key_bundle,
                    value_bundle=value_bundle,
                    key_weight=key_weight,
                    value_weight=value_weight,
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "similarity to cluster centroid": st.column_config.NumberColumn(format="%.6f")
                },
            )
            expander = st.expander(
                "Show all topics in this cluster",
                key=f"all-{bundle.representation}-{bundle.text_hash}-{cluster_id}-{export_kwargs.get('dataset_name', 'combined')}",
                on_change="rerun",
            )
            if show_all_members and expander.open:
                with expander:
                    st.dataframe(
                        topic_rows(
                            bundle,
                            ranked_all,
                            key_bundle=key_bundle,
                            value_bundle=value_bundle,
                            key_weight=key_weight,
                            value_weight=value_weight,
                        ),
                        hide_index=True,
                        width="stretch",
                    )
            elif not show_all_members:
                st.caption("Enable ‘Show all cluster members’ to render the full cluster table.")
    if method == DEFAULT_CLUSTERING_METHOD and include_noise:
        indices = np.flatnonzero(labels == -1)
        st.subheader(f"Noise · {len(indices):,} topics")
        if len(indices):
            preview = [(int(index), float("nan")) for index in indices[:representative_limit]]
            st.dataframe(
                topic_rows(bundle, preview, key_bundle=key_bundle, value_bundle=value_bundle, key_weight=key_weight, value_weight=value_weight),
                hide_index=True,
                width="stretch",
            )
            expander = st.expander("Show all noise topics", on_change="rerun")
            if expander.open:
                with expander:
                    st.dataframe(
                        topic_rows(bundle, [(int(index), float("nan")) for index in indices], key_bundle=key_bundle, value_bundle=value_bundle, key_weight=key_weight, value_weight=value_weight),
                        hide_index=True,
                        width="stretch",
                    )
        else:
            st.caption("No noise topics at this threshold.")
    elif method == KMEANS_METHOD:
        st.caption("K-means has no DBSCAN noise label; noise is not applicable.")


def comparison_rows(
    bundles: Mapping[str, EmbeddingBundle],
    labels_by_strategy: Mapping[str, np.ndarray],
    *,
    selected_datasets: Sequence[str],
    dataset_mode: str,
    parameters: ClusteringParameters,
    key_weight: float,
    value_weight: float,
) -> pd.DataFrame:
    rows = []
    for strategy in bundles:
        summary = cluster_summary(labels_by_strategy[strategy])
        rows.append(
            {
                "representation": strategy,
                "selected_datasets": ",".join(selected_datasets),
                "dataset_mode": dataset_mode,
                "model": bundles[strategy].model_name,
                "topic_count": len(bundles[strategy].topics),
                "unique_text_count": len(set(bundles[strategy].texts.tolist())),
                "key_weight": key_weight if strategy == WEIGHTED_KEY_VALUE else None,
                "value_weight": value_weight if strategy == WEIGHTED_KEY_VALUE else None,
                "clustering_method": parameters.method,
                "similarity_threshold": parameters.similarity_threshold if parameters.method == DEFAULT_CLUSTERING_METHOD else None,
                "eps": threshold_to_eps(parameters.similarity_threshold) if parameters.method == DEFAULT_CLUSTERING_METHOD else None,
                "min_samples": parameters.min_samples if parameters.method == DEFAULT_CLUSTERING_METHOD else None,
                "k": parameters.k if parameters.method == KMEANS_METHOD else None,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def selected_topic_comparison(
    selected_topic: str,
    bundles: Mapping[str, EmbeddingBundle],
    labels_by_strategy: Mapping[str, np.ndarray],
    limit: int,
    *,
    key_weight: float,
    value_weight: float,
) -> pd.DataFrame:
    rows = []
    for strategy, bundle in bundles.items():
        matches = np.flatnonzero(bundle.topics == selected_topic)
        if not len(matches):
            continue
        selected = int(matches[0])
        labels = labels_by_strategy[strategy]
        assigned = int(labels[selected])
        neighbors = cluster_neighbors(selected, bundle.embeddings, labels, limit)
        neighbor_text = " | ".join(
            f"{bundle.topics[index]} ({score:.4f})" for index, score in neighbors
        )
        rows.append(
            {
                "representation": strategy,
                "cluster_id": assigned,
                "is_noise": assigned == -1,
                "cluster_member_count": int(np.sum(labels == assigned)) if assigned != -1 else 0,
                "nearest cluster neighbors": neighbor_text or "None (noise or singleton)",
                "key_weight": key_weight if strategy == WEIGHTED_KEY_VALUE else None,
                "value_weight": value_weight if strategy == WEIGHTED_KEY_VALUE else None,
            }
        )
        if bundle.expected_groups is not None and str(bundle.expected_groups[selected]):
            rows[-1]["expected_group"] = str(bundle.expected_groups[selected])
            rows[-1]["variant_type"] = str(bundle.variant_types[selected]) if bundle.variant_types is not None else ""
            rows[-1]["neighbor expected groups"] = " | ".join(
                f"{bundle.expected_groups[index]} ({bundle.variant_types[index]})" for index, _ in neighbors
            ) or "None"
    return pd.DataFrame(rows)


def all_topic_bundles(
    full_bundles: Mapping[str, EmbeddingBundle],
    selected_representations: Sequence[str],
    selected_datasets: Sequence[str],
    key_weight: float,
    value_weight: float,
) -> dict[str, EmbeddingBundle]:
    result: dict[str, EmbeddingBundle] = {}
    weighted: EmbeddingBundle | None = None
    if WEIGHTED_KEY_VALUE in selected_representations:
        weighted = build_weighted_bundle(
            full_bundles[KEY_ONLY], full_bundles[VALUE_ONLY], key_weight, value_weight
        )
    for strategy in selected_representations:
        bundle = weighted if strategy == WEIGHTED_KEY_VALUE else full_bundles[strategy]
        result[strategy] = filter_bundle(bundle, selected_datasets)
    return result


DIAGNOSTIC_CAPTIONS = {
    VALUE_ONLY: "Uses textual metadata values without schema field names. Related concepts may separate more clearly, depending on the model and clustering parameters.",
    KEY_ONLY: "All generated records share one field structure. Identical texts, similarity ties, and one large cluster are expected; this is a structural negative control.",
    "KEY_VALUE": "Combines repeated schema names with meaningful metadata values, so shared keys may affect the relative influence of value content.",
    NORMALIZED_KEY_VALUE: "Removes only conservative formatting differences. Capitalization and underscores are reduced, but synonyms are not standardized.",
    WEIGHTED_KEY_VALUE: "Combines separately cached KEY_ONLY and VALUE_ONLY embeddings. Higher value weight generally emphasizes content; higher key weight generally emphasizes schema structure.",
}


def render_synthetic_diagnostics(
    bundles: Mapping[str, EmbeddingBundle],
    labels_by_strategy: Mapping[str, np.ndarray],
    selected_representations: Sequence[str],
    key_weight: float,
    value_weight: float,
) -> None:
    st.subheader("Controlled synthetic benchmark metrics")
    st.caption(
        "These metrics use deterministic generated expected groups and apply only to the controlled benchmark. "
        "They are not accuracy measurements for the real datasets. DBSCAN noise is retained as one explicit discovered label for ARI/NMI and is reported separately; purity evaluates non-noise records only."
    )
    metric_rows = []
    for strategy in selected_representations:
        bundle = bundles[strategy]
        expected = bundle.evaluation_labels.tolist() if bundle.evaluation_labels is not None else []
        groups = bundle.expected_groups.tolist() if bundle.expected_groups is not None else []
        metric_rows.append({"representation": strategy, **controlled_benchmark_metrics(labels_by_strategy[strategy], expected, groups)})
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, width="stretch")

    st.subheader("Synthetic representation diagnostics")
    for strategy in selected_representations:
        st.caption(f"**{strategy}:** {DIAGNOSTIC_CAPTIONS[strategy]}")
    for strategy in selected_representations:
        bundle = bundles[strategy]
        st.markdown(f"### {REPRESENTATION_LABELS[strategy]}")
        color_by = st.selectbox(
            f"PCA color for {strategy}",
            ("Discovered cluster", "Expected group"),
            key=f"pca-color-{strategy}",
        )
        coordinates = pca_coordinates(bundle.embeddings)
        frame = pd.DataFrame(
            {
                "PC1": coordinates[:, 0],
                "PC2": coordinates[:, 1],
                "topic": bundle.topics,
                "expected_group": bundle.expected_groups,
                "variant_type": bundle.variant_types,
                "discovered_cluster": labels_by_strategy[strategy],
                "measurement_key": bundle.measurement_keys,
                "source_name": bundle.sources,
                "exact_representation_text": bundle.texts,
            }
        )
        if strategy == WEIGHTED_KEY_VALUE:
            frame["key_weight"] = key_weight
            frame["value_weight"] = value_weight
        color_column = "discovered_cluster" if color_by == "Discovered cluster" else "expected_group"
        figure = px.scatter(
            frame,
            x="PC1",
            y="PC2",
            color=color_column,
            hover_data=["topic", "expected_group", "variant_type", "discovered_cluster", "measurement_key", "source_name", "exact_representation_text"],
            title=f"PCA · {REPRESENTATION_LABELS[strategy]}",
        )
        st.plotly_chart(figure, width="stretch")
        st.caption("PCA is visualization only; clustering uses the original full-dimensional normalized embeddings.")
        show_values = st.toggle("Show similarity values", value=False, key=f"heatmap-values-{strategy}")
        order = sorted(
            range(len(bundle.topics)),
            key=lambda index: (str(bundle.expected_groups[index]), str(bundle.variant_types[index]), str(bundle.topics[index])),
        )
        ordered_embeddings = bundle.embeddings[order]
        labels = bundle.topics[order].tolist()
        similarities = cosine_similarity_matrix(ordered_embeddings)
        heatmap = go.Figure(
            data=go.Heatmap(
                z=similarities,
                x=labels,
                y=labels,
                text=np.round(similarities, 2) if show_values else None,
                texttemplate="%{text:.2f}" if show_values else None,
                colorscale="Viridis",
                zmin=-1,
                zmax=1,
            )
        )
        heatmap.update_layout(title=f"Cosine similarity · {REPRESENTATION_LABELS[strategy]}", height=700)
        st.plotly_chart(heatmap, width="stretch")


st.set_page_config(page_title="SmartMQTT embedding clustering explorer", layout="wide")
st.title("SmartMQTT embedding clustering explorer")
st.write(
    "Compare how five metadata representations affect unsupervised clustering of the same logical IoT topics. "
    "This is clustering, not classification."
)
st.info(
    "DBSCAN uses cosine distance with eps = 1 - similarity_threshold. Cluster counts emerge from the threshold and min_samples. "
    "The same ordered topics and selected clustering parameters are held constant across representations. Similarity does not prove scientific equivalence."
)

st.session_state.setdefault("experiment_requested", False)
st.session_state.setdefault("data_source", DATA_SOURCE_OPTIONS[0])
with st.sidebar:
    st.header("Experiment controls")
    data_source = st.selectbox("Experiment data", DATA_SOURCE_OPTIONS, key="data_source")

try:
    loaded_source = load_experiment_source(
        data_source,
        signature_loader=file_signatures,
        real_loader=cached_records,
        synthetic_loader=cached_synthetic_records,
    )
    records = loaded_source.records
except (FileNotFoundError, ValueError) as error:
    st.error(f"Cannot load topic-text inputs for {data_source}: {error}")
    st.stop()

with st.sidebar:
    synthetic_only = data_source == GENERATED_DATA
    dataset_options = ["synthetic"] if synthetic_only else [*DATASET_NAMES, "synthetic"] if data_source == MIXED_DATA else list(DATASET_NAMES)
    selection_key = f"dataset_selection_{data_source}"
    default_datasets = ["synthetic"] if synthetic_only else dataset_options
    st.session_state.setdefault(selection_key, default_datasets)
    if synthetic_only:
        selected_datasets = ["synthetic"]
        st.caption("Generated benchmark records use dataset = synthetic.")
    else:
        selected_datasets = st.multiselect(
            "Datasets to include", dataset_options, key=selection_key
        )
        left, right = st.columns(2)
        if left.button("Select all", key=f"select-all-datasets-{data_source}"):
            st.session_state[selection_key] = list(dataset_options)
            st.rerun()
        if right.button("Clear all", key=f"clear-all-datasets-{data_source}"):
            st.session_state[selection_key] = []
            st.rerun()
    if data_source != REAL_DATA:
        generated = [record for record in records if record["dataset"] == "synthetic"]
        st.caption(f"Generated records: {len(generated)}")
        st.caption(f"Expected groups: {synthetic_group_counts(generated)}")
    dataset_mode = st.selectbox("Dataset mode", DATASET_MODES, index=0)
    limit_error: str | None = None
    limits = None
    if selected_datasets:
        try:
            limits = clustering_parameter_limits(records, selected_datasets, dataset_mode)
        except ValueError as error:
            limit_error = str(error)
    selected_representations = st.multiselect(
        "Representations", REPRESENTATIONS, default=list(REPRESENTATIONS)
    )
    model_name = st.selectbox("Embedding model", MODEL_OPTIONS, index=0)
    key_weight = 0.30
    value_weight = 0.70
    if WEIGHTED_KEY_VALUE in selected_representations:
        preset = st.selectbox("Weight preset", [*WEIGHT_PRESETS, "Custom"], index=1)
        if preset == "Custom":
            key_percent = st.slider("Key weight (%)", 0, 100, 30, 5)
            key_weight = key_percent / 100.0
            value_weight = 1.0 - key_weight
        else:
            key_weight, value_weight = WEIGHT_PRESETS[preset]
        st.caption(f"Key weight: {key_weight:.0%} · Value weight: {value_weight:.0%}")
    method = st.selectbox("Clustering method", (DEFAULT_CLUSTERING_METHOD, KMEANS_METHOD), index=0)
    with st.form("experiment-controls"):
        threshold = None
        min_samples = None
        k = None
        if method == DEFAULT_CLUSTERING_METHOD:
            threshold = st.slider("Similarity threshold", 0.50, 0.99, 0.80, 0.01)
            max_min_samples = limits.maximum_min_samples if limits is not None else 2
            min_samples = st.number_input("Minimum samples", 2, max(2, max_min_samples), 2, 1)
        else:
            max_k = limits.maximum_k if limits is not None else 2
            k = st.number_input("Number of clusters (K)", 2, max(2, max_k), min(20, max(2, max_k)), 1)
            if dataset_mode == DATASET_MODES[1]:
                st.caption(
                    "In separate mode, K is limited by the smallest selected dataset because the same K is applied independently to every selected dataset."
                )
        representative_limit = st.number_input("Top representatives per cluster", 1, 20, 5, 1)
        include_noise = st.toggle("Include noise", value=True)
        show_all_members = st.toggle("Show all cluster members", value=False)
        submitted = st.form_submit_button("Run experiment", type="primary", icon=":material/hub:")
    st.caption(f"CPU device · {model_name}")

if submitted:
    st.session_state.experiment_requested = True
if not selected_datasets:
    st.warning("Select at least one dataset.")
    st.stop()
if limit_error is not None or limits is None:
    st.error(limit_error or "Unable to calculate clustering parameter limits.")
    st.stop()
if not selected_representations:
    st.warning("Select at least one representation.")
    st.stop()
if not st.session_state.experiment_requested:
    st.subheader("Ready to compare representations")
    st.write("Choose the experiment controls and press **Run experiment**. Base caches are created only for the selected representations.")
    st.stop()

parameters = ClusteringParameters(
    similarity_threshold=float(threshold if threshold is not None else 0.80),
    min_samples=int(min_samples if min_samples is not None else 2),
    method=method,
    k=int(k) if k is not None else None,
)
try:
    validate_clustering_parameters(parameters, limits)
except ValueError as error:
    st.error(str(error))
    st.stop()
try:
    needed = required_bases(selected_representations)
    with st.spinner("Loading or creating base embedding caches…"):
        full_bundles = {
            strategy: cached_base_bundle(records, model_name, strategy, data_source)
            for strategy in needed
        }
        bundles = all_topic_bundles(
            full_bundles, selected_representations, selected_datasets, key_weight, value_weight
        )
        reference_topics = next(iter(bundles.values())).topics
        if any(not np.array_equal(reference_topics, bundle.topics) for bundle in bundles.values()):
            raise ValueError("Selected representations do not use the same topic order")
        if len(reference_topics) < (parameters.min_samples if method == DEFAULT_CLUSTERING_METHOD else parameters.k or 2):
            raise ValueError("The selected topic count is smaller than the clustering parameter")
except Exception as error:
    st.error(f"Embedding or experiment setup failed: {error}")
    st.stop()

labels_by_strategy: dict[str, np.ndarray] = {}
if dataset_mode == DATASET_MODES[0]:
    labels_by_strategy = cluster_all_strategies(
        {strategy: bundle.embeddings for strategy, bundle in bundles.items()}, parameters
    )
embedding_dimension = next(iter(bundles.values())).embedding_dimension
component_key_bundle = (
    filter_bundle(full_bundles[KEY_ONLY], selected_datasets)
    if WEIGHTED_KEY_VALUE in selected_representations
    else None
)
component_value_bundle = (
    filter_bundle(full_bundles[VALUE_ONLY], selected_datasets)
    if WEIGHTED_KEY_VALUE in selected_representations
    else None
)

labels_by_dataset: dict[str, dict[str, np.ndarray]] = {}
if dataset_mode == DATASET_MODES[1]:
    labels_by_dataset = {
        dataset: cluster_all_strategies(
            {strategy: filter_bundle(bundle, [dataset]).embeddings for strategy, bundle in bundles.items()},
            parameters,
        )
        for dataset in selected_datasets
    }

st.subheader("Active experiment parameters")
active = {
    "data source": data_source,
    "selected datasets": ", ".join(selected_datasets),
    "dataset mode": dataset_mode,
    "topic count": len(next(iter(bundles.values())).topics),
    "model": model_name,
    "embedding dimension": embedding_dimension,
    "device": "CPU",
    "representations": ", ".join(selected_representations),
    "clustering method": method,
    "similarity threshold": threshold if threshold is not None else "not applicable",
    "eps": threshold_to_eps(threshold) if threshold is not None else "not applicable",
    "min_samples": min_samples if min_samples is not None else "not applicable",
    "K": k if k is not None else "not applicable",
    "weights": f"key {key_weight:.0%} / value {value_weight:.0%}" if WEIGHTED_KEY_VALUE in selected_representations else "not applicable",
    "top representatives per cluster": representative_limit,
}
st.dataframe(pd.DataFrame([active]), hide_index=True, width="stretch")
st.caption(
    "Cache status: "
    + " · ".join(
        f"{strategy}={'loaded' if full_bundles[strategy].reused else 'created'}"
        for strategy in needed
    )
    + ". Weighted embeddings are recomputed in memory and never persisted as NPZ."
)

st.subheader("Cross-strategy comparison")
if dataset_mode == DATASET_MODES[0]:
    comparison = comparison_rows(
        bundles,
        labels_by_strategy,
        selected_datasets=selected_datasets,
        dataset_mode=dataset_mode,
        parameters=parameters,
        key_weight=key_weight,
        value_weight=value_weight,
    )
else:
    comparison = pd.concat(
        [
            comparison_rows(
                {strategy: filter_bundle(bundle, [dataset]) for strategy, bundle in bundles.items()},
                labels_by_dataset[dataset],
                selected_datasets=[dataset],
                dataset_mode=dataset_mode,
                parameters=parameters,
                key_weight=key_weight,
                value_weight=value_weight,
            )
            for dataset in selected_datasets
        ],
        ignore_index=True,
    )
st.dataframe(comparison, hide_index=True, width="stretch")

if WEIGHTED_KEY_VALUE in bundles:
    st.info(
        "Weighted key:value uses separately cached KEY_ONLY and VALUE_ONLY embeddings. "
        f"Key weight: {key_weight:.0%}; value weight: {value_weight:.0%}."
    )

st.subheader("Selected-topic comparison")
topic_options = next(iter(bundles.values())).topics.tolist()
default_topic_index = 0
if data_source == DATA_SOURCE_OPTIONS[1]:
    default_topic_index = next(
        (index for index, group in enumerate(next(iter(bundles.values())).expected_groups.tolist()) if group == "air_temperature"),
        0,
    )
selected_topic = st.selectbox("Topic", topic_options, index=default_topic_index)
key_bundle = component_key_bundle
value_bundle = component_value_bundle
topic_bundles = bundles
topic_labels = labels_by_strategy
if dataset_mode == DATASET_MODES[1]:
    selected_dataset_for_topic = str(
        next(
            bundle.datasets[np.flatnonzero(bundle.topics == selected_topic)[0]]
            for bundle in bundles.values()
            if len(np.flatnonzero(bundle.topics == selected_topic))
        )
    )
    topic_bundles = {
        strategy: filter_bundle(bundle, [selected_dataset_for_topic])
        for strategy, bundle in bundles.items()
    }
    topic_labels = labels_by_dataset[selected_dataset_for_topic]
    key_bundle = filter_bundle(component_key_bundle, [selected_dataset_for_topic]) if component_key_bundle is not None else None
    value_bundle = filter_bundle(component_value_bundle, [selected_dataset_for_topic]) if component_value_bundle is not None else None
st.dataframe(
    selected_topic_comparison(
        selected_topic,
        topic_bundles,
        topic_labels,
        int(representative_limit),
        key_weight=key_weight,
        value_weight=value_weight,
    ),
    hide_index=True,
    width="stretch",
)

if data_source == DATA_SOURCE_OPTIONS[1]:
    benchmark_labels = labels_by_strategy if dataset_mode == DATASET_MODES[0] else labels_by_dataset["synthetic"]
    render_synthetic_diagnostics(
        bundles,
        benchmark_labels,
        selected_representations,
        key_weight,
        value_weight,
    )

st.subheader("Cluster details")
tabs = st.tabs(
    [REPRESENTATION_LABELS[strategy] for strategy in selected_representations],
    on_change="rerun",
    key="representation-tabs",
)
for strategy, tab in zip(selected_representations, tabs):
    if not tab.open:
        continue
    with tab:
        bundle = bundles[strategy]
        weighted_keys = key_bundle if strategy == WEIGHTED_KEY_VALUE else None
        weighted_values = value_bundle if strategy == WEIGHTED_KEY_VALUE else None
        export_kwargs = {
            "selected_datasets": selected_datasets,
            "data_source": data_source,
            "dataset_mode": dataset_mode,
            "method": method,
            "similarity_threshold": threshold,
            "min_samples": int(min_samples) if min_samples is not None else None,
            "k": int(k) if k is not None else None,
            "dataset_name": "combined",
        }
        if dataset_mode == DATASET_MODES[0]:
            cluster_rows(
                bundle,
                labels_by_strategy[strategy],
                show_all_members=show_all_members,
                representative_limit=int(representative_limit),
                include_noise=include_noise,
                method=method,
                key_bundle=weighted_keys,
                value_bundle=weighted_values,
                key_weight=key_weight if strategy == WEIGHTED_KEY_VALUE else None,
                value_weight=value_weight if strategy == WEIGHTED_KEY_VALUE else None,
                export_kwargs=export_kwargs,
            )
        else:
            for dataset in selected_datasets:
                dataset_bundle = filter_bundle(bundle, [dataset])
                dataset_labels = labels_by_dataset[dataset][strategy]
                st.markdown(f"### {dataset}")
                dataset_export_kwargs = {**export_kwargs, "dataset_name": dataset}
                dataset_key = filter_bundle(weighted_keys, [dataset]) if weighted_keys is not None else None
                dataset_value = filter_bundle(weighted_values, [dataset]) if weighted_values is not None else None
                cluster_rows(
                    dataset_bundle,
                    dataset_labels,
                    show_all_members=show_all_members,
                    representative_limit=int(representative_limit),
                    include_noise=include_noise,
                    method=method,
                    key_bundle=dataset_key,
                    value_bundle=dataset_value,
                    key_weight=key_weight if strategy == WEIGHTED_KEY_VALUE else None,
                    value_weight=value_weight if strategy == WEIGHTED_KEY_VALUE else None,
                    export_kwargs=dataset_export_kwargs,
                )
