"""Streamlit UI for transparent nearest-neighbor topic-text exploration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.embedding_engine import (
    MODEL_OPTIONS,
    REPRESENTATIONS,
    SOURCE_FIELDS,
    construct_representation,
    construct_representations,
    encode_texts,
    load_jsonl_records,
    load_sentence_transformer,
    source_field_and_value,
    text_hash,
    top_k_retrieval,
    weighted_top_k_vote,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILES = (
    ROOT / "data" / "processed" / "01_sgim_topic_texts.jsonl",
    ROOT / "data" / "processed" / "02_beach_weather_topic_texts.jsonl",
    ROOT / "data" / "processed" / "03_beach_water_topic_texts.jsonl",
    ROOT / "data" / "processed" / "04_open_air_topic_texts.jsonl",
)
DATASET_NAMES = ("sgim", "beach_weather", "beach_water", "open_air")


@st.cache_resource(show_spinner="Loading the sentence-transformer model on CPU…")
def cached_model(model_name: str) -> Any:
    return load_sentence_transformer(model_name, device="cpu")


@st.cache_data(show_spinner=False)
def cached_records(
    signatures: tuple[tuple[str, int, int], ...]
) -> list[dict[str, Any]]:
    return load_jsonl_records(Path(path) for path, _, _ in signatures)


@st.cache_data(show_spinner=False)
def cached_representations(
    serialized_records: tuple[str, ...], representation: str
) -> tuple[str, ...]:
    records = [json.loads(value) for value in serialized_records]
    return tuple(construct_representations(records, representation))


@st.cache_data(show_spinner="Encoding topic text locally…")
def cached_embeddings(
    model_name: str,
    representation: str,
    input_text_hash: str,
    texts: tuple[str, ...],
) -> np.ndarray:
    del representation, input_text_hash  # Included explicitly in the cache key.
    return encode_texts(cached_model(model_name), texts)


def file_signatures() -> tuple[tuple[str, int, int], ...]:
    signatures = []
    for path in INPUT_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Required input file is missing: {path}")
        stat = path.stat()
        signatures.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(signatures)


def serialize_records(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(json.dumps(record, sort_keys=True, ensure_ascii=False) for record in records)


def representations_for(
    records: Sequence[Mapping[str, Any]], representation: str
) -> tuple[str, ...]:
    return cached_representations(serialize_records(records), representation)


def embeddings_for(
    texts: Sequence[str], model_name: str, representation: str
) -> np.ndarray:
    values = tuple(texts)
    return cached_embeddings(model_name, representation, text_hash(values), values)


def records_for_dataset(
    records: Sequence[dict[str, Any]], dataset: str
) -> list[dict[str, Any]]:
    if dataset == "all":
        return list(records)
    return [record for record in records if record["dataset"] == dataset]


def source_value(record: Mapping[str, Any]) -> str:
    try:
        _, value = source_field_and_value(record)
        return value
    except ValueError:
        return ""


def retrieval_table(
    neighbors: Sequence[Mapping[str, Any]], candidate_texts: Sequence[str]
) -> pd.DataFrame:
    rows = []
    for neighbor in neighbors:
        rows.append(
            {
                "rank": neighbor["rank"],
                "cosine_similarity": neighbor["cosine_similarity"],
                "dataset": neighbor.get("dataset"),
                "topic": neighbor.get("topic"),
                "measurement_key": neighbor.get("measurement_key"),
                "source_value": source_value(neighbor),
                "candidate_representation": candidate_texts[
                    int(neighbor["candidate_index"])
                ],
            }
        )
    return pd.DataFrame(rows)


def similarity_chart(table: pd.DataFrame, title: str) -> None:
    if table.empty:
        return
    chart_data = table.sort_values("cosine_similarity", ascending=True).copy()
    chart_data["label"] = chart_data["rank"].astype(str) + ". " + chart_data[
        "measurement_key"
    ].astype(str)
    figure = px.bar(
        chart_data,
        x="cosine_similarity",
        y="label",
        orientation="h",
        hover_data=["topic", "dataset", "source_value"],
        title=title,
    )
    figure.update_layout(yaxis_title="Neighbor", xaxis_title="Cosine similarity")
    st.plotly_chart(figure, width="stretch")


def transparency_panel(
    query_embedding: np.ndarray,
    neighbors: Sequence[Mapping[str, Any]],
    vote: Mapping[str, Any],
) -> None:
    with st.expander("How this prediction was produced"):
        st.markdown(
            """
1. Query text was converted into an embedding vector.
2. Candidate topic text records were converted into vectors using the same model.
3. Embeddings were L2-normalized.
4. Cosine similarity was calculated.
5. Candidates were sorted from highest to lowest similarity.
6. Top-1 prediction used the nearest candidate's measurement_key.
7. Weighted prediction summed top-k similarity scores by measurement_key.
"""
        )
        st.write(f"Query vector dimension: `{query_embedding.shape[0]}`")
        st.write(f"Query vector L2 norm: `{np.linalg.norm(query_embedding):.8f}`")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "rank": neighbor["rank"],
                        "topic": neighbor["topic"],
                        "measurement_key": neighbor["measurement_key"],
                        "cosine_similarity": neighbor["cosine_similarity"],
                    }
                    for neighbor in neighbors
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.write("Weighted score for each candidate label:")
        st.json(vote["score_by_label"])
        if st.checkbox("Display first 20 query embedding components", key=f"components-{id(query_embedding)}"):
            st.code(np.array2string(query_embedding[:20], precision=6))


def prediction_panel(
    query_embedding: np.ndarray,
    neighbors: Sequence[Mapping[str, Any]],
    candidate_texts: Sequence[str],
    title: str,
) -> None:
    table = retrieval_table(neighbors, candidate_texts)
    if table.empty:
        st.warning("No candidates met the selected similarity threshold.")
        return
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={"cosine_similarity": st.column_config.NumberColumn(format="%.6f")},
    )
    similarity_chart(table, title)
    vote = weighted_top_k_vote(neighbors)
    top_one = neighbors[0]["measurement_key"]
    first, second = st.columns(2)
    first.metric("Top-1 predicted measurement_key", str(top_one))
    weighted = vote["predicted_label"] or "No confident weighted prediction"
    second.metric("Weighted top-k predicted measurement_key", weighted)
    st.write("Weighted score per label:")
    st.dataframe(
        pd.DataFrame(
            [
                {"measurement_key": label, "weighted_score": score}
                for label, score in vote["score_by_label"].items()
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.write("Neighbors contributing positive weight:")
    st.dataframe(
        pd.DataFrame(vote["contributing_neighbors"]),
        width="stretch",
        hide_index=True,
    )
    transparency_panel(query_embedding, neighbors, vote)


def run_retrieval(
    query_text: str,
    candidates: Sequence[dict[str, Any]],
    candidate_representation: str,
    model_name: str,
    top_k: int,
    minimum_similarity: float,
    *,
    query_topic: str | None = None,
    exclude_self: bool = False,
) -> tuple[np.ndarray, tuple[str, ...], list[dict[str, Any]]]:
    candidate_texts = representations_for(candidates, candidate_representation)
    candidate_vectors = embeddings_for(
        candidate_texts, model_name, candidate_representation
    )
    query_vector = embeddings_for(
        (query_text,), model_name, f"query::{candidate_representation}"
    )[0]
    neighbors = top_k_retrieval(
        candidates,
        candidate_vectors,
        query_vector,
        top_k,
        query_topic=query_topic,
        exclude_self=exclude_self,
        minimum_similarity=minimum_similarity,
    )
    return query_vector, candidate_texts, neighbors


def render_audit(records: list[dict[str, Any]]) -> None:
    st.subheader("Combined topic-text dataset audit")
    frame = pd.DataFrame(records)
    counts = frame.groupby("dataset").size().rename("records")
    missing_count = sum(
        not record.get("topic")
        or not record.get("measurement_key")
        or not str(record.get("text", "")).strip()
        for record in records
    )
    duplicate_count = len(frame) - frame["topic"].nunique()
    columns = st.columns(5)
    columns[0].metric("Total records", len(frame))
    columns[1].metric("Unique topics", frame["topic"].nunique())
    columns[2].metric("Unique measurement keys", frame["measurement_key"].nunique())
    columns[3].metric("Duplicate topics", duplicate_count)
    columns[4].metric("Missing required fields", missing_count)
    st.write("Records by dataset:")
    st.dataframe(counts.reset_index(), hide_index=True, width="stretch")
    st.write("Measurement counts by dataset:")
    measurement_counts = (
        frame.groupby(["dataset", "measurement_key"]).size().rename("topic_count").reset_index()
    )
    st.dataframe(measurement_counts, hide_index=True, width="stretch")
    search = st.text_input("Search topic records", key="audit-search").strip().lower()
    display = frame.copy()
    if search:
        display = display[
            display.astype(str).apply(
                lambda row: row.str.lower().str.contains(search, regex=False).any(), axis=1
            )
        ]
    st.dataframe(display, width="stretch", hide_index=True)
    selected_topic = st.selectbox("Inspect raw JSON record", display["topic"].tolist())
    selected = next(record for record in records if record["topic"] == selected_topic)
    st.json(selected)


def render_existing_topic(
    records: list[dict[str, Any]],
    source_dataset: str,
    target_dataset: str,
    representation: str,
    model_name: str,
    top_k: int,
    exclude_self: bool,
    minimum_similarity: float,
) -> None:
    query_records = records_for_dataset(records, source_dataset)
    query_topic = st.selectbox(
        "Existing query topic", [record["topic"] for record in query_records]
    )
    query_record = next(record for record in query_records if record["topic"] == query_topic)
    query_text = construct_representation(query_record, representation)
    st.write("Raw query record:")
    st.json(query_record)
    st.write(f"Measurement key: `{query_record['measurement_key']}`")
    st.write("Source metadata:")
    st.json({field: query_record[field] for field in SOURCE_FIELDS if field in query_record})
    st.write("Exact representation sent to the embedding model:")
    st.code(query_text)
    effective_exclusion = exclude_self and (
        target_dataset == "all" or target_dataset == source_dataset
    )
    if st.button("Run existing-topic retrieval", type="primary"):
        candidates = records_for_dataset(records, target_dataset)
        try:
            query_vector, candidate_texts, neighbors = run_retrieval(
                query_text,
                candidates,
                representation,
                model_name,
                top_k,
                minimum_similarity,
                query_topic=query_topic,
                exclude_self=effective_exclusion,
            )
        except Exception as error:  # Model/download errors should be readable in the UI.
            st.error(f"Embedding or retrieval failed: {error}")
            return
        st.write(f"Embedding dimension: `{query_vector.shape[0]}`")
        st.write(f"Embedding L2 norm: `{np.linalg.norm(query_vector):.8f}`")
        prediction_panel(
            query_vector, neighbors, candidate_texts, "Top-k existing-topic matches"
        )


def render_free_text(
    records: list[dict[str, Any]],
    target_dataset: str,
    representation: str,
    model_name: str,
    top_k: int,
    minimum_similarity: float,
) -> None:
    raw_query = st.text_area(
        "Free-text query",
        value="air temperature",
        help="Examples: wind speed, water turbidity, soil moisture, ambient humidity, PM2.5 mass concentration",
    ).strip()
    wrap = st.checkbox("Use wrapper: query: {user text}")
    embedded_text = f"query: {raw_query}" if wrap else raw_query
    st.write("Raw query:")
    st.code(raw_query)
    st.write("Final text sent to the embedding model:")
    st.code(embedded_text)
    if st.button("Run free-text retrieval", type="primary", disabled=not raw_query):
        candidates = records_for_dataset(records, target_dataset)
        try:
            query_vector, candidate_texts, neighbors = run_retrieval(
                embedded_text,
                candidates,
                representation,
                model_name,
                top_k,
                minimum_similarity,
            )
        except Exception as error:
            st.error(f"Embedding or retrieval failed: {error}")
            return
        st.write(f"Embedding dimension: `{query_vector.shape[0]}`")
        st.write(f"Embedding L2 norm: `{np.linalg.norm(query_vector):.8f}`")
        prediction_panel(query_vector, neighbors, candidate_texts, "Top-k free-text matches")


def render_comparison(
    records: list[dict[str, Any]],
    source_dataset: str,
    target_dataset: str,
    model_name: str,
    exclude_self: bool,
    minimum_similarity: float,
) -> None:
    query_records = records_for_dataset(records, source_dataset)
    query_topic = st.selectbox(
        "Topic to compare across representations",
        [record["topic"] for record in query_records],
        key="comparison-topic",
    )
    query_record = next(record for record in query_records if record["topic"] == query_topic)
    st.json(query_record)
    if st.button("Compare all representations", type="primary"):
        candidates = records_for_dataset(records, target_dataset)
        effective_exclusion = exclude_self and (
            target_dataset == "all" or target_dataset == source_dataset
        )
        comparison_rows: list[dict[str, Any]] = []
        for representation in REPRESENTATIONS:
            query_text = construct_representation(query_record, representation)
            try:
                _, candidate_texts, neighbors = run_retrieval(
                    query_text,
                    candidates,
                    representation,
                    model_name,
                    5,
                    minimum_similarity,
                    query_topic=query_topic,
                    exclude_self=effective_exclusion,
                )
            except Exception as error:
                st.error(f"{representation} failed: {error}")
                return
            top = neighbors[0] if neighbors else {}
            comparison_rows.append(
                {
                    "representation": representation,
                    "embedded_text": query_text,
                    "top_1_topic": top.get("topic"),
                    "top_1_measurement_key": top.get("measurement_key"),
                    "cosine_similarity": top.get("cosine_similarity"),
                    "changed_from_query_label": (
                        top.get("measurement_key") != query_record["measurement_key"]
                        if top
                        else None
                    ),
                }
            )
            st.markdown(f"#### {representation}")
            st.write("Embedded text:")
            st.code(query_text)
            if neighbors:
                st.write(
                    f"Top-1: `{top['topic']}` → `{top['measurement_key']}` "
                    f"(cosine `{top['cosine_similarity']:.6f}`)"
                )
                st.dataframe(
                    retrieval_table(neighbors, candidate_texts),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.warning("No candidate met the similarity threshold.")
        st.subheader("Representation comparison")
        st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
        st.caption(
            "Changes involving source-bearing representations may indicate that source names are influencing retrieval."
        )


def main() -> None:
    st.set_page_config(page_title="SmartMQTT Embedding Explorer", layout="wide")
    st.title("SmartMQTT Embedding Explorer")
    st.write(
        "Inspect sentence-embedding nearest neighbors and provisional measurement-key voting."
    )
    st.warning(
        "This is nearest-neighbor semantic retrieval, not a trained classifier. "
        "measurement_key is a provisional label. Similarity does not establish scientific "
        "equivalence. Cross-dataset accuracy cannot be claimed until a curated ground-truth "
        "mapping exists. Source names may introduce location bias. The source-only "
        "representation is a negative-control representation."
    )
    try:
        records = cached_records(file_signatures())
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Cannot load topic-text inputs: {error}")
        st.stop()

    with st.sidebar:
        st.header("Retrieval controls")
        model_name = st.selectbox("Model name", MODEL_OPTIONS)
        representation = st.selectbox("Representation", REPRESENTATIONS)
        source_dataset = st.selectbox("Source dataset", DATASET_NAMES)
        target_dataset = st.selectbox("Target dataset", ("all", *DATASET_NAMES))
        top_k = st.slider("Top-k", 1, 25, 10)
        exclude_self = st.checkbox("Exclude exact topic/self-match", value=True)
        minimum_similarity = st.slider(
            "Minimum cosine similarity", -1.0, 1.0, 0.0, 0.01
        )
        if st.button("Clear embedding cache"):
            cached_embeddings.clear()
            st.success("Embedding cache cleared.")
        st.caption("Inference device: CPU")

    audit_tab, existing_tab, free_text_tab, comparison_tab = st.tabs(
        (
            "Dataset Audit",
            "Existing Topic Explorer",
            "Free-Text Query",
            "Representation Comparison",
        )
    )
    with audit_tab:
        render_audit(records)
    with existing_tab:
        render_existing_topic(
            records,
            source_dataset,
            target_dataset,
            representation,
            model_name,
            top_k,
            exclude_self,
            minimum_similarity,
        )
    with free_text_tab:
        render_free_text(
            records,
            target_dataset,
            representation,
            model_name,
            top_k,
            minimum_similarity,
        )
    with comparison_tab:
        render_comparison(
            records,
            source_dataset,
            target_dataset,
            model_name,
            exclude_self,
            minimum_similarity,
        )


if __name__ == "__main__":
    main()
