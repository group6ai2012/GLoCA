from __future__ import annotations

import streamlit as st

from src.inference import (
    SUPPORTED_CDC_DEMO_RUNS,
    artifact_fingerprint,
    build_exemplar_index,
    decode_uploaded_image,
    load_cdc_bundle,
    predict_top_clusters,
)


st.set_page_config(
    page_title="CDC + GLoCA Cluster Explorer",
    page_icon="🌿",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _cached_bundle(dataset: str, fingerprint: tuple[int, ...]):
    del fingerprint
    return load_cdc_bundle(dataset)


@st.cache_data(show_spinner=False)
def _cached_exemplar_index(dataset: str, fingerprint: tuple[int, ...]):
    bundle = _cached_bundle(dataset, fingerprint)
    return build_exemplar_index(bundle, examples_per_cluster=5)


def _render_results(dataset: str, uploaded_bytes: bytes, top_n: int) -> None:
    try:
        image = decode_uploaded_image(uploaded_bytes)
        fingerprint = artifact_fingerprint(dataset)
        with st.spinner("Loading the seed-42 CDC + GLoCA model…"):
            bundle = _cached_bundle(dataset, fingerprint)
        with st.spinner("Running deterministic inference…"):
            predictions = predict_top_clusters(bundle, image, top_n)
        with st.spinner("Selecting original-dataset exemplars…"):
            exemplar_index = _cached_exemplar_index(dataset, fingerprint)
    except Exception as exc:
        st.error(f"Inference could not be completed: {exc}")
        return

    preview, summary = st.columns([1, 2])
    with preview:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)
    with summary:
        st.subheader("Most confident clusters")
        st.dataframe(
            {
                "Cluster ID": [item.cluster_id for item in predictions],
                "Calibrated probability": [item.probability for item in predictions],
            },
            hide_index=True,
            use_container_width=True,
            column_config={
                "Calibrated probability": st.column_config.ProgressColumn(
                    "Calibrated probability",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                )
            },
        )

    st.divider()
    for prediction in predictions:
        st.subheader(
            f"Cluster {prediction.cluster_id} — {prediction.probability:.1%} for the upload"
        )
        st.caption(
            "Five original training images with the highest calibration-head affinity "
            "for this cluster."
        )
        columns = st.columns(5)
        for column, exemplar in zip(
            columns, exemplar_index[prediction.cluster_id], strict=True
        ):
            with column:
                st.image(str(exemplar.path), use_container_width=True)
                st.caption(
                    f"{exemplar.path.name}\n\nCluster affinity: {exemplar.probability:.1%}"
                )


def main() -> None:
    st.title("CDC + GLoCA Plant Disease Cluster Explorer")
    st.write(
        "Upload a plant image to compare it with the unsupervised clusters learned by "
        "the main seed-42 CDC + GLoCA gated experiments."
    )
    st.info(
        "This research model predicts anonymous cluster IDs, not disease names. "
        "Its calibrated softmax scores are cluster affinities and must not be treated "
        "as agricultural or medical diagnosis confidence."
    )

    labels = {
        key: run.display_name for key, run in SUPPORTED_CDC_DEMO_RUNS.items()
    }
    with st.sidebar:
        st.header("Inference settings")
        dataset = st.selectbox(
            "Training dataset",
            options=list(labels),
            format_func=labels.get,
        )
        top_n = st.slider("Clusters to show", min_value=1, max_value=10, value=3)
        st.caption("Model: CDC + gloca_gated · seed 42 · latest checkpoint")

    uploaded = st.file_uploader(
        "Plant image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Maximum file size: 10 MB.",
    )
    analyze = st.button(
        "Analyze image",
        type="primary",
        disabled=uploaded is None,
    )
    if analyze and uploaded is not None:
        _render_results(dataset, uploaded.getvalue(), top_n)


main()
