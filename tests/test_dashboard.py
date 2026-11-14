"""Dashboard geometry and audio checks; no model downloads or real inference."""

from io import BytesIO

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from streamlit.testing.v1 import AppTest

import dashboard


def bank():
    X = np.random.default_rng(4).normal(size=(40, 4)).astype(np.float32)
    pool = pd.DataFrame({"uid": [f"clip{i}" for i in range(len(X))],
                         "label": "A07", "speaker": "speaker"})
    return pool, X, NearestNeighbors(n_neighbors=22).fit(X)


def wav(seconds=1, sr=8000):
    buf = BytesIO()
    sf.write(buf, np.sin(np.arange(int(seconds * sr)) * 0.1), sr, format="WAV")
    return buf.getvalue()


def test_neighbors_exclude_reference_uid():
    pool, X, nn = bank()
    dist, idx = dashboard.neighbors(nn, pool, X[:1], "clip0")
    assert len(idx) == 21 and 0 not in idx
    expected = np.argsort(np.linalg.norm(X - X[0], axis=1))[1:22]
    np.testing.assert_array_equal(idx, expected)
    np.testing.assert_allclose(dist, np.linalg.norm(X[idx] - X[0], axis=1), atol=1e-5)


def test_custom_query_keeps_all_references():
    pool, X, nn = bank()
    dist, idx = dashboard.neighbors(nn, pool, X[:1])
    assert len(idx) == 21 and idx[0] == 0 and dist[0] == 0


def test_map_transforms_without_refitting_and_uses_supplied_neighbors():
    pool, X, nn = bank()
    pca = PCA(n_components=2).fit(X)
    xy = pca.transform(X)
    original_components = pca.components_.copy()
    q = X[:1] + 3
    _, idx = dashboard.neighbors(nn, pool, q)
    fig = dashboard.map_figure(pool, xy, pca, q, idx)
    np.testing.assert_array_equal(pca.components_, original_components)
    np.testing.assert_allclose(fig.data[-1].x, pca.transform(q)[:, 0])
    np.testing.assert_allclose(fig.data[-1].y, pca.transform(q)[:, 1])
    np.testing.assert_allclose(fig.data[-2].x, xy[idx, 0])
    assert fig.data[-1].marker.symbol == "diamond"
    assert len(fig.data[1].x) == 21 * 3


def test_decode_resamples_to_mono_16khz():
    y = dashboard.decode(wav())
    assert y.shape == (16000,) and np.isfinite(y).all()


@pytest.mark.parametrize("seconds", [0.1, 30.1])
def test_decode_rejects_duration_limits(seconds):
    with pytest.raises(ValueError, match="between 0.2 and 30"):
        dashboard.decode(wav(seconds))


def test_decode_rejects_corrupt_and_large_files():
    with pytest.raises((RuntimeError, ValueError)):
        dashboard.decode(b"not audio")
    with pytest.raises(ValueError, match="20 MB"):
        dashboard.decode(b"x" * (20 * 1024 * 1024 + 1))


def test_browser_demo_uses_cached_reference_and_excludes_it():
    app = AppTest.from_file(str(dashboard.ROOT / "dashboard.py"), default_timeout=60).run()
    assert not app.exception
    assert app.selectbox[0].value.name == "A07.flac"
    app.button[0].click().run()
    assert not app.exception
    digest, q, distances, indices = app.session_state["located"]
    pool, X, pca, xy, nn, hashes = dashboard.reference_bank()
    assert hashes[digest] not in pool.iloc[indices].uid.values
    assert len(indices) == 21
    reference = pool.index[pool.uid == hashes[digest]][0]
    np.testing.assert_array_equal(q, X[reference:reference + 1])
    app.selectbox[0].select_index(1).run()
    assert not app.exception
    assert not any(h.value.startswith("Closest class:") for h in app.subheader)
