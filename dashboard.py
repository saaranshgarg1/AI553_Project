#!/usr/bin/env python3
"""Local audio embedding viewer. Run: streamlit run dashboard.py --server.address localhost"""

import csv
import hashlib
from collections import Counter
from io import BytesIO

import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from demo import REFS
from tada.data import ROOT, SYSTEMS, manifest, ssl
from tada.features import W2VBert

K = 21
COLORS = {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e",
          "grid": "#e1e0d9", "reference": "#898781", "neighbor": "#2a78d6", "query": "#eb6834"}


def generator(label):
    return "human speech" if label == "bonafide" else SYSTEMS[label][1]


@st.cache_resource
def reference_bank():
    """Fit once: the coordinate system never changes when a query changes."""
    pool = manifest().query("partition == 'eval'").reset_index(drop=True)
    X = ssl(4, pool.uid)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=42).fit(X)
    xy = pca.transform(X)
    nn = NearestNeighbors(n_neighbors=K + 1).fit(X)
    with open(REFS, newline="") as f:
        hashes = {r["sha256"]: r["uid"] for r in csv.DictReader(f)}
    return pool, X, pca, xy, nn, hashes


@st.cache_resource
def encoder():
    return W2VBert()


def decode(content):
    """Bound decoded audio before inference; uploads never become filesystem paths."""
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("File exceeds 20 MB. Choose a smaller recording.")
    with sf.SoundFile(BytesIO(content)) as f:
        duration = len(f) / f.samplerate
        if not 0.2 <= duration <= 30:
            raise ValueError("Choose a recording between 0.2 and 30 seconds.")
        y = f.read(dtype="float32", always_2d=True).mean(axis=1)
        sr = f.samplerate
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    if not np.isfinite(y).all():
        raise ValueError("Recording contains invalid samples. Export it again as WAV or FLAC.")
    return y


@st.cache_data(max_entries=8, show_spinner=False)
def embed_upload(content):
    return encoder()(decode(content))[4][None]


def neighbors(nn, pool, q, query_uid=None):
    """Search original embeddings, then remove the recognized reference UID."""
    distances, indices = nn.kneighbors(q)
    keep = pool.uid.values[indices[0]] != query_uid
    return distances[0][keep][:K], indices[0][keep][:K]


def class_centres(pool, X):
    """One mean embedding per class; visualization only, not the kNN classifier."""
    labels = np.array(sorted(pool.label.unique()))
    centres = np.stack([X[pool.label.values == label].mean(axis=0) for label in labels])
    return labels, centres


def map_figure(labels, centres, pca, q=None):
    xy = pca.transform(centres)
    distances = np.linalg.norm(centres - q[0], axis=1) if q is not None else None
    fig = go.Figure()
    hover = "%{text}<br>Class mean embedding"
    if distances is not None:
        hover += "<br>Distance to audio (1024D): %{customdata:.2f}"
    fig.add_trace(go.Scatter(
        x=xy[:, 0], y=xy[:, 1], mode="markers+text", name="Class centres",
        text=labels, textposition="top center",
        marker=dict(size=14, color=COLORS["reference"],
                    line=dict(color=COLORS["surface"], width=2)),
        customdata=distances, hovertemplate=hover + "<extra></extra>"))
    if q is not None:
        point = pca.transform(q)[0]
        nearest = int(np.argmin(distances))
        # Only one connector; its label is the original-space distance, not 2D length.
        fig.add_trace(go.Scatter(
            x=[point[0], xy[nearest, 0]], y=[point[1], xy[nearest, 1]],
            mode="lines", name="Nearest centre (1024D)",
            line=dict(color=COLORS["neighbor"], width=2), hoverinfo="skip"))
        fig.add_annotation(
            x=(point[0] + xy[nearest, 0]) / 2, y=(point[1] + xy[nearest, 1]) / 2,
            text=f"{labels[nearest]} · d={distances[nearest]:.2f} (1024D)",
            showarrow=False, yshift=-20, bgcolor=COLORS["surface"])
        fig.add_trace(go.Scatter(
            x=[point[0]], y=[point[1]], mode="markers+text", name="Selected audio",
            text=["Selected audio"], textposition="bottom center",
            marker=dict(size=20, symbol="diamond", color=COLORS["query"],
                        line=dict(color=COLORS["surface"], width=2)),
            hovertemplate="Selected audio<extra></extra>"))
    variance = pca.explained_variance_ratio_ * 100
    fig.update_layout(
        height=560, margin=dict(l=16, r=16, t=30, b=16),
        paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["surface"],
        font=dict(family="Arial, sans-serif", color=COLORS["ink"], size=14),
        legend=dict(orientation="h", y=1.1, x=0), dragmode="pan",
        hovermode="closest", hoverdistance=30, uirevision="fixed-reference-map",
        xaxis=dict(title=f"PC1 · {variance[0]:.1f}% variance", gridcolor=COLORS["grid"], zeroline=False),
        yaxis=dict(title=f"PC2 · {variance[1]:.1f}% variance", gridcolor=COLORS["grid"], zeroline=False))
    return fig


def main():
    st.set_page_config(page_title="TADA · Audio map", layout="wide")
    st.title("Where does this audio land?")
    st.write("Frozen w2v-BERT 2.0 · layer 4 · 21-neighbor attribution")
    st.caption("Local processing. Reference examples use stored embeddings; new recordings use the frozen model.")
    try:
        with st.spinner("Loading reference embeddings and fitting the fixed map…"):
            pool, X, pca, xy, nn, hashes = reference_bank()
            labels, centres = class_centres(pool, X)
    except (OSError, KeyError, ValueError) as exc:
        st.error(f"Cannot load reference bank: {exc}. Run the project's feature extraction first.")
        return

    controls, chart = st.columns([1, 3], gap="large")
    with controls:
        st.subheader("Choose audio")
        source = st.radio("Audio source", ["Demo clip", "Upload audio"], horizontal=True)
        content = None
        if source == "Demo clip":
            files = sorted((ROOT / "demo_clips").glob("*.flac"))
            chosen = st.selectbox("Demo clip", files, format_func=lambda p: p.name)
            if chosen:
                content = chosen.read_bytes()
        else:
            uploaded = st.file_uploader("Audio file", type=["wav", "flac", "ogg", "mp3"],
                                        max_upload_size=20)
            if uploaded is not None:
                content = uploaded.getvalue()
        digest = hashlib.sha256(content).hexdigest() if content is not None else None
        uid = hashes.get(digest)
        matches = pool.index[pool.uid == uid]
        reference = int(matches[0]) if len(matches) else None
        if content is not None:
            # Decode first, so corrupt uploads do not render a broken player or invoke the model.
            try:
                y = decode(content)
            except (ValueError, RuntimeError, OSError) as exc:
                st.error(f"Cannot read audio: {exc}. Try WAV or FLAC.")
                content = None
            else:
                st.audio(y, sample_rate=16000)
                st.caption(f"{len(y) / 16000:.1f} seconds · resampled to 16 kHz mono")
        if reference is not None:
            label = pool.iloc[reference].label
            st.write(f"True label: **{label} · {generator(label)}**")
            st.caption("This reference UID is excluded from its own neighbor search.")
        else:
            st.caption("Ground truth unavailable for custom recordings.")
        if st.button("Locate audio", type="primary", disabled=content is None, width="stretch"):
            try:
                with st.spinner("Locating audio… first model load may take a minute."):
                    q = X[reference:reference + 1] if reference is not None else embed_upload(content)
                    dist, idx = neighbors(nn, pool, q, uid)
                st.session_state.located = (digest, q, dist, idx)
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"Analysis failed: {exc}. Try a shorter clip; check model cache and GPU memory.")
        st.caption("Gray circles: class centres. Orange diamond: your audio. Blue line: nearest centre.")

    result = st.session_state.get("located")
    active = result is not None and result[0] == digest and content is not None
    q, dist, idx = result[1:] if active else (None, None, None)
    with chart:
        st.subheader("Reference map")
        st.plotly_chart(map_figure(pool, xy, pca, q, idx), width="stretch", theme=None,
                        config={"displaylogo": False, "scrollZoom": True})
        st.caption(f"{len(pool):,} reference clips · 13 generators + human speech · "
                   f"PCA shows {pca.explained_variance_ratio_.sum():.1%} of embedding variance.")
        st.info("2D is a visual aid. Neighbors and votes use all 1,024 dimensions, "
                "so a nearby dot on this map may not be a nearest neighbor.")
        if not active:
            st.write("Choose a clip and press **Locate audio** to place it on the map.")

    if active:
        votes = Counter(pool.label.values[idx])
        pred, count = votes.most_common(1)[0]
        st.subheader(f"Closest class: {pred} · {generator(pred)}")
        st.write(f"**{count}/{K} votes** · mean embedding distance **{dist.mean():.2f}**")
        st.caption("Closest known class, not proof of origin. A new generator is still assigned a known class here; "
                   "this dashboard does not apply the CLI's unknown-generator heuristic. Votes are not probabilities.")
        st.write("Votes: " + " · ".join(f"{label}: {n}" for label, n in votes.most_common()))
        table = pool.iloc[idx][["uid", "label", "speaker"]].reset_index(drop=True)
        table.insert(0, "rank", np.arange(1, K + 1))
        table["generator"] = table.label.map(generator)
        table["distance (1024D)"] = dist
        table["PC1"], table["PC2"] = xy[idx, 0], xy[idx, 1]
        st.subheader("Nearest neighbors")
        st.dataframe(table, hide_index=True, width="stretch")
        point = pca.transform(q)[0]
        st.caption(f"Selected audio: PC1 {point[0]:.3f}, PC2 {point[1]:.3f}")
    with st.expander("Reference data · table equivalent of the map"):
        table = pool[["uid", "label", "speaker"]].copy()
        table["PC1"], table["PC2"] = xy[:, 0], xy[:, 1]
        st.dataframe(table, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
