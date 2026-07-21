"""Streamlit UI for comparing metadata representations with clustering."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from app.embedding_engine import (
    DEFAULT_CLUSTERING_METHOD,
    KMEANS_METHOD,
    MODEL_NAME,
    REPRESENTATIONS,
    REPRESENTATION_LABELS,
    ClusteringParameters,
    EmbeddingBundle,
    cluster_all_strategies,
    cluster_neighbors,
    cluster_summary,
    get_or_create_embedding_cache,
    load_jsonl_records,
    load_sentence_transformer,
    ranked_cluster_members,
    representative_topics,
    similarity_to_cluster_representatives,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILES = (
    ROOT / "data" / "processed" / "01_sgim_topic_texts.jsonl",
    ROOT / "data" / "processed" / "02_beach_weather_topic_texts.jsonl",
    ROOT / "data" / "processed" / "03_beach_water_topic_texts.jsonl",
    ROOT / "data" / "processed" / "04_open_air_topic_texts.jsonl",
)
DATASET_NAMES = ("sgim", "beach_weather", "beach_water", "open_air")
CACHE_FILES = {
    strategy: ROOT / "embedding_cache" / f"{strategy.lower()}.npz"
    for strategy in REPRESENTATIONS
}
TAB_LABELS = [REPRESENTATION_LABELS[strategy] for strategy in REPRESENTATIONS]


@st.cache_resource(show_spinner="Loading all-MiniLM-L6-v2 on CPU…")
def cached_model() -> Any:
    return load_sentence_transformer(MODEL_NAME, device="cpu")


@st.cache_data(show_spinner=False)
def cached_records(
    signatures: tuple[tuple[str, int, int], ...]
) -> list[dict[str, Any]]:
    return load_jsonl_records(Path(path) for path, _, _ in signatures)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_embedding_bundle(
    signatures: tuple[tuple[str, int, int], ...], strategy: str
) -> EmbeddingBundle:
    records = cached_records(signatures)
    return get_or_create_embedding_cache(
        CACHE_FILES[strategy], records, strategy, MODEL_NAME, cached_model
    )


def file_signatures() -> tuple[tuple[str, int, int], ...]:
    signatures = []
    for path in INPUT_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Required input file is missing: {path}")
        stat = path.stat()
        signatures.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(signatures)


def subset_bundle(bundle: EmbeddingBundle, datasets: Sequence[str]) -> EmbeddingBundle:
    mask = np.isin(bundle.datasets, np.asarray(datasets))
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
    )


def topic_rows(
    bundle: EmbeddingBundle,
    ranked: Sequence[tuple[int, float]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "topic": bundle.topics[index],
                "dataset": bundle.datasets[index],
                "measurement_key": bundle.measurement_keys[index],
                "source": bundle.sources[index],
                "exact embedded text": bundle.texts[index],
                "similarity to cluster centroid": similarity,
            }
            for index, similarity in ranked
        ]
    )


def cluster_export_csv(
    strategy: str,
    threshold: float,
    min_samples: int,
    bundle: EmbeddingBundle,
    labels: np.ndarray,
) -> bytes:
    similarities = similarity_to_cluster_representatives(bundle.embeddings, labels)
    frame = pd.DataFrame(
        {
            "representation": strategy,
            "threshold": threshold,
            "min_samples": min_samples,
            "cluster_id": labels,
            "is_noise": labels == -1,
            "topic": bundle.topics,
            "dataset": bundle.datasets,
            "source": bundle.sources,
            "measurement_key": bundle.measurement_keys,
            "representation_text": bundle.texts,
            "similarity_to_cluster_representative": similarities,
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def render_noise(bundle: EmbeddingBundle, labels: np.ndarray, limit: int) -> None:
    indices = np.flatnonzero(labels == -1)
    if len(indices) == 0:
        st.caption("No noise topics at this threshold.")
        return
    st.subheader(f"Noise · {len(indices):,} topics")
    preview = [(int(index), float("nan")) for index in indices[:limit]]
    st.dataframe(topic_rows(bundle, preview), hide_index=True, width="stretch")
    expander = st.expander(
        "Show all noise topics",
        key=f"noise-{bundle.text_hash}-{len(indices)}",
        on_change="rerun",
    )
    if expander.open:
        with expander:
            all_rows = [(int(index), float("nan")) for index in indices]
            st.dataframe(topic_rows(bundle, all_rows), hide_index=True, width="stretch")


def render_clusters(
    strategy: str,
    bundle: EmbeddingBundle,
    labels: np.ndarray,
    representative_limit: int,
    include_noise: bool,
    threshold: float,
    min_samples: int,
) -> None:
    counts = Counter(int(label) for label in labels if int(label) != -1)
    ordered_clusters = sorted(counts, key=lambda cluster_id: (-counts[cluster_id], cluster_id))
    st.caption(
        f"Clusters are ordered by size. Representative topics are nearest to each "
        f"cluster's normalized mean embedding. Model: `{MODEL_NAME}`."
    )
    ready_key = f"csv-ready-{strategy}-{bundle.text_hash}-{threshold:.2f}-{min_samples}"
    if st.button(
        f"Prepare {REPRESENTATION_LABELS[strategy]} CSV",
        key=f"prepare-{ready_key}",
        icon=":material/download:",
    ):
        st.session_state[ready_key] = True
    if st.session_state.get(ready_key, False):
        csv_bytes = cluster_export_csv(
            strategy, threshold, min_samples, bundle, labels
        )
        st.download_button(
            f"Download {REPRESENTATION_LABELS[strategy]} assignments",
            data=csv_bytes,
            file_name=f"{strategy.lower()}_clusters.csv",
            mime="text/csv",
            icon=":material/download:",
            key=f"download-{ready_key}",
        )
    if not ordered_clusters:
        st.warning("No clusters emerged from the selected parameters.")
    for cluster_id in ordered_clusters:
        member_indices = np.flatnonzero(labels == cluster_id)
        member_keys = bundle.measurement_keys[member_indices].tolist()
        dominant = Counter(member_keys).most_common(5)
        ranked_all = ranked_cluster_members(bundle.embeddings, labels, cluster_id)
        mean_similarity = float(np.mean([score for _, score in ranked_all]))
        with st.container(border=True):
            st.subheader(f"Cluster {cluster_id} · {len(member_indices):,} topics")
            st.write(
                f"**Datasets:** {', '.join(sorted(set(bundle.datasets[member_indices])))}  \n"
                f"**Unique measurement keys:** {len(set(member_keys)):,}  \n"
                f"**Dominant measurement keys:** "
                + ", ".join(f"{key} ({count})" for key, count in dominant)
                + f"  \n**Mean similarity to representative:** {mean_similarity:.6f}"
            )
            st.markdown(f"**Top {min(representative_limit, len(member_indices))} representative topics**")
            representatives = representative_topics(
                bundle.embeddings, labels, cluster_id, representative_limit
            )
            st.dataframe(
                topic_rows(bundle, representatives),
                hide_index=True,
                width="stretch",
                column_config={
                    "similarity to cluster centroid": st.column_config.NumberColumn(format="%.6f")
                },
            )
            expander = st.expander(
                "Show all topics in this cluster",
                key=f"all-{strategy}-{bundle.text_hash}-{cluster_id}",
                on_change="rerun",
            )
            if expander.open:
                with expander:
                    st.dataframe(
                        topic_rows(bundle, ranked_all),
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "similarity to cluster centroid": st.column_config.NumberColumn(format="%.6f")
                        },
                    )
    if include_noise:
        render_noise(bundle, labels, representative_limit)


def comparison_rows(
    bundles: Mapping[str, EmbeddingBundle],
    labels_by_strategy: Mapping[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for strategy in REPRESENTATIONS:
        summary = cluster_summary(labels_by_strategy[strategy])
        rows.append(
            {
                "representation": strategy,
                "similarity_threshold": threshold,
                "cluster_count": summary["cluster_count"],
                "noise_count": summary["noise_count"],
                "largest_cluster_size": summary["largest_cluster_size"],
                "median_cluster_size": summary["median_cluster_size"],
                "unique_text_count": len(set(bundles[strategy].texts.tolist())),
            }
        )
    return pd.DataFrame(rows)


def selected_topic_comparison(
    selected_topic: str,
    bundles: Mapping[str, EmbeddingBundle],
    labels_by_strategy: Mapping[str, np.ndarray],
    limit: int,
) -> pd.DataFrame:
    rows = []
    baseline_neighbors: tuple[str, ...] | None = None
    for strategy in REPRESENTATIONS:
        bundle = bundles[strategy]
        selected = int(np.flatnonzero(bundle.topics == selected_topic)[0])
        labels = labels_by_strategy[strategy]
        neighbors = cluster_neighbors(selected, bundle.embeddings, labels, limit)
        neighbor_topics = tuple(str(bundle.topics[index]) for index, _ in neighbors)
        if baseline_neighbors is None:
            baseline_neighbors = neighbor_topics
        rows.append(
            {
                "representation": strategy,
                "cluster_id": int(labels[selected]),
                "cluster_neighbor_count": int(np.sum(labels == labels[selected]) - 1)
                if labels[selected] != -1
                else 0,
                "nearest cluster neighbors": " | ".join(neighbor_topics) or "None (noise or singleton)",
                "differs from VALUE_ONLY": neighbor_topics != baseline_neighbors,
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="SmartMQTT embedding clustering explorer", layout="wide")
st.title("SmartMQTT embedding clustering explorer")
st.write(
    "Compare how four deterministic metadata-text representations change "
    "unsupervised topic clusters. This is clustering, not classification."
)
st.info(
    "DBSCAN uses cosine distance and derives its cluster count from the selected "
    "similarity threshold (`eps = 1 - similarity_threshold`). Label `-1` is noise. "
    "The same all-MiniLM-L6-v2 model, normalized embeddings, record order, and "
    "clustering parameters are used for every representation."
)

try:
    signatures = file_signatures()
    records = cached_records(signatures)
except (FileNotFoundError, ValueError) as error:
    st.error(f"Cannot load topic-text inputs: {error}")
    st.stop()

st.session_state.setdefault("clustering_requested", False)
with st.sidebar:
    st.header("Clustering controls")
    with st.form("clustering-controls"):
        selected_datasets = st.multiselect(
            "Datasets to include", DATASET_NAMES, default=DATASET_NAMES
        )
        similarity_threshold = st.slider(
            "Similarity threshold", 0.50, 0.99, 0.80, 0.01
        )
        min_samples = st.number_input("Minimum samples", 1, 50, 2, 1)
        representative_limit = st.number_input(
            "Topics displayed per cluster", 1, 50, 5, 1
        )
        include_noise = st.toggle("Include noise", value=True)
        with st.expander("Advanced clustering method", expanded=False):
            method = st.selectbox(
                "Clustering method", (DEFAULT_CLUSTERING_METHOD, KMEANS_METHOD)
            )
            k = None
            if method == KMEANS_METHOD:
                k = st.number_input("K-means k", 1, max(1, len(records)), 10, 1)
        submitted = st.form_submit_button(
            "Run clustering", type="primary", icon=":material/hub:"
        )
    st.caption(f"Embedding model: {MODEL_NAME} · CPU")

if submitted:
    st.session_state.clustering_requested = True

if not selected_datasets:
    st.warning("Select at least one dataset.")
    st.stop()

if not st.session_state.clustering_requested:
    st.subheader("Ready to compare representations")
    st.write(
        "Choose the datasets and threshold, then press **Run clustering**. "
        "The first run creates the four ignored NPZ embedding caches; matching "
        "model and text hashes reuse them on later runs."
    )
    placeholder_tabs = st.tabs(TAB_LABELS)
    for tab, label in zip(placeholder_tabs, TAB_LABELS):
        with tab:
            st.caption(f"{label} cluster details will appear after clustering runs.")
    st.stop()

parameters = ClusteringParameters(
    similarity_threshold=float(similarity_threshold),
    min_samples=int(min_samples),
    method=method,
    k=int(k) if k is not None else None,
)

try:
    with st.spinner("Loading or creating normalized embedding caches…"):
        full_bundles = {
            strategy: cached_embedding_bundle(signatures, strategy)
            for strategy in REPRESENTATIONS
        }
        reference_topics = full_bundles[REPRESENTATIONS[0]].topics
        if any(
            not np.array_equal(reference_topics, full_bundles[strategy].topics)
            for strategy in REPRESENTATIONS[1:]
        ):
            raise ValueError("Embedding caches do not use the same topic order")
        bundles = {
            strategy: subset_bundle(full_bundles[strategy], selected_datasets)
            for strategy in REPRESENTATIONS
        }
        labels_by_strategy = cluster_all_strategies(
            {strategy: bundle.embeddings for strategy, bundle in bundles.items()},
            parameters,
        )
except Exception as error:
    st.error(f"Embedding or clustering failed: {error}")
    st.stop()

st.subheader("Primary cross-strategy comparison")
comparison = comparison_rows(bundles, labels_by_strategy, similarity_threshold)
st.dataframe(
    comparison,
    hide_index=True,
    width="stretch",
    column_config={
        "similarity_threshold": st.column_config.NumberColumn(format="%.2f"),
        "median_cluster_size": st.column_config.NumberColumn(format="%.1f"),
    },
)

st.subheader("Representation summaries")
for strategy in REPRESENTATIONS:
    summary = cluster_summary(labels_by_strategy[strategy])
    with st.container(border=True):
        st.markdown(f"**{REPRESENTATION_LABELS[strategy]}**")
        with st.container(horizontal=True):
            st.metric("Topics", len(bundles[strategy].topics), border=True)
            st.metric(
                "Unique texts", len(set(bundles[strategy].texts.tolist())), border=True
            )
            st.metric("Clusters", summary["cluster_count"], border=True)
            st.metric("Noise", summary["noise_count"], border=True)
            st.metric("Largest", summary["largest_cluster_size"], border=True)
            st.metric("Median", f"{summary['median_cluster_size']:.1f}", border=True)
            st.metric("Singletons", summary["singleton_count"], border=True)

st.subheader("Selected-topic cluster neighbors")
selected_topic = st.selectbox(
    "Topic to compare", bundles[REPRESENTATIONS[0]].topics.tolist()
)
st.dataframe(
    selected_topic_comparison(
        selected_topic, bundles, labels_by_strategy, int(representative_limit)
    ),
    hide_index=True,
    width="stretch",
)
st.caption(
    "Neighbor lists contain other topics assigned to the selected topic's cluster, "
    "ranked by cosine similarity to the selected topic."
)

tabs = st.tabs(TAB_LABELS, on_change="rerun", key="representation-tabs")
for strategy, tab in zip(REPRESENTATIONS, tabs):
    if tab.open:
        with tab:
            render_clusters(
                strategy,
                bundles[strategy],
                labels_by_strategy[strategy],
                int(representative_limit),
                include_noise,
                float(similarity_threshold),
                int(min_samples),
            )
