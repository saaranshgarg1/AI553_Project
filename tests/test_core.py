"""Checks of the signal-processing core against textbook results.  Run: python -m pytest -q tests"""

import csv
import hashlib
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demo  # noqa: E402
from tada import audio, features, models  # noqa: E402

rng = np.random.default_rng(0)
SR = 16000


def test_levinson_equals_direct_solution():
    """Levinson-Durbin must solve the Yule-Walker equations R a = -r and recover the AR filter."""
    a_true = [1.0, -1.3, 0.8, -0.2]
    y = lfilter([1.0], a_true, rng.standard_normal(20000))
    r = np.correlate(y, y, "full")[len(y) - 1:len(y) + 4]
    a, _ = features.levinson(r[None], 3)
    np.testing.assert_allclose(a[0, 1:], solve_toeplitz(r[:3], -r[1:4]), rtol=1e-8)
    np.testing.assert_allclose(a[0], a_true, atol=0.05)


def test_lpc_cepstrum_matches_fft_cepstrum():
    a = np.array([[1.0, -1.3, 0.8, -0.2]])
    c_fft = np.fft.ifft(np.log(1 / np.fft.fft(a[0], 4096))).real[1:13]
    np.testing.assert_allclose(features.lpc_to_cepstrum(a, 12)[0], c_fft, atol=1e-6)


def test_feature_sizes():
    y = (0.3 * np.sin(2 * np.pi * 150 * np.arange(2 * SR) / SR) + 0.01 * rng.standard_normal(2 * SR)).astype(np.float32)
    sizes = {k: v.shape[0] for k, v in features.classical(y).items()}
    assert sizes == {"mfcc": 78, "lfcc": 120, "lpcc": 34, "lfcc_low": 60, "lfcc_high": 60}


def test_quantisation_rules():
    """Uniform: SQNR = 6.02 b + 1.76 dB for a full-scale sine. mu-law: much better for quiet signals."""
    sqnr = lambda x, q: 10 * np.log10(np.sum(x ** 2) / np.sum((x - q) ** 2))
    loud = 0.999 * np.sin(2 * np.pi * 1000.3 * np.arange(SR) / SR)
    assert abs(sqnr(loud, audio.quantize(loud, 8)) - (6.02 * 8 + 1.76)) < 1
    quiet = 0.01 * np.sin(2 * np.pi * 300 * np.arange(SR) / SR)
    assert sqnr(quiet, audio.mulaw(quiet)) > sqnr(quiet, audio.quantize(quiet)) + 10


def test_telephone_channel_removes_high_band():
    tone = 0.5 * np.sin(2 * np.pi * 6000 * np.arange(SR) / SR)    # 6 kHz > 4 kHz Nyquist of 8 kHz sampling
    assert np.sqrt(np.mean(audio.DEGRADATIONS["telephone"](tone) ** 2)) < 0.01


def test_eer_two_gaussians():
    """Two unit-variance Gaussians d apart: EER = Phi(-d/2)."""
    s = np.r_[rng.standard_normal(20000), 2 + rng.standard_normal(20000)]
    y = np.r_[np.zeros(20000), np.ones(20000)]
    assert abs(models.eer(y, s)[0] - norm.cdf(-1)) < 0.01


# ---------------------------------------------------------------- demo reference clips
REF = Path(__file__).resolve().parents[1] / "demo_clips" / "reference_clips.csv"


def test_reference_clips_are_the_bank_clips():
    """Every curated clip must be a byte copy of a bank clip, so its hash really identifies it."""
    m = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "manifest.csv").set_index("uid")
    rows = list(csv.DictReader(open(REF)))
    assert sorted(r["label"] for r in rows) == sorted([f"A{i:02d}" for i in range(7, 20)] + ["bonafide"])
    for r in rows:
        p = REF.parent / r["filename"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"]
        assert m.loc[r["uid"], "label"] == r["label"]
        assert m.loc[r["uid"], "path"] == r["original_path"]


def test_identify_known_and_renamed_clip(tmp_path):
    orig = REF.parent / "A07.flac"
    copy = tmp_path / "my_recording.flac"                  # renamed copy must still be recognised
    shutil.copyfile(orig, copy)
    assert demo.identify(orig) == "LA_E_8252737"
    assert demo.identify(copy) == "LA_E_8252737"


def test_identify_unknown_file_is_none(tmp_path):
    other = tmp_path / "other.flac"
    other.write_bytes(b"not a reference clip")
    assert demo.identify(other) is None
    assert demo.identify(tmp_path / "missing.flac") is None


def test_identify_without_reference_csv(tmp_path):
    """A checkout without demo_clips must not crash on --file."""
    orig = REF.parent / "A07.flac"
    assert demo.identify(orig, refs=tmp_path / "absent.csv") is None


def test_main_excludes_recognised_clip_from_bank(monkeypatch, capsys):
    """--file on a shipped clip: truth comes from the manifest and that uid never enters the kNN bank."""
    uid = "LA_E_8252737"
    others = [f"LA_E_{i:07d}" for i in range(41)]
    pool = pd.DataFrame({"uid": [uid] + others, "partition": "eval",
                         "label": ["A07"] * 21 + ["A08"] * 21,
                         "speaker": "LA_0018", "path": "x.flac", "duration": 3.0})
    seen = {}
    monkeypatch.setattr(demo, "manifest", lambda: pool)
    monkeypatch.setattr(demo.librosa, "load", lambda *a, **k: (np.zeros(16000, np.float32), 16000))
    monkeypatch.setattr(demo, "ssl", lambda layer, uids: seen.update(uids=list(uids)) or
                        np.random.default_rng(0).standard_normal((len(uids), 1024)).astype(np.float32))
    monkeypatch.setattr(demo, "W2VBert", lambda: lambda y: np.zeros((25, 1024), np.float32))
    fig = MagicMock()
    fig.subplots.return_value = (fig, [MagicMock() for _ in range(3)])
    monkeypatch.setattr(demo, "plt", fig)
    monkeypatch.setattr(sys, "argv", ["demo.py", "--file", str(REF.parent / "A07.flac")])

    demo.main()
    out = capsys.readouterr().out
    assert uid not in seen["uids"] and len(seen["uids"]) == 41   # self-match removed
    assert "[truth: A07]" in out
    assert "Reference clip LA_E_8252737: true generator = A07" in out


def test_main_unknown_file_keeps_old_behaviour(monkeypatch, capsys, tmp_path):
    pool = pd.DataFrame({"uid": [f"LA_E_{i:07d}" for i in range(42)], "partition": "eval",
                         "label": ["A07"] * 21 + ["A08"] * 21,
                         "speaker": "LA_0018", "path": "x.flac", "duration": 3.0})
    seen = {}
    monkeypatch.setattr(demo, "manifest", lambda: pool)
    monkeypatch.setattr(demo.librosa, "load", lambda *a, **k: (np.zeros(16000, np.float32), 16000))
    monkeypatch.setattr(demo, "ssl", lambda layer, uids: seen.update(uids=list(uids)) or
                        np.random.default_rng(0).standard_normal((len(uids), 1024)).astype(np.float32))
    monkeypatch.setattr(demo, "W2VBert", lambda: lambda y: np.zeros((25, 1024), np.float32))
    fig = MagicMock()
    fig.subplots.return_value = (fig, [MagicMock() for _ in range(3)])
    monkeypatch.setattr(demo, "plt", fig)
    unknown = tmp_path / "my_voice.flac"
    unknown.write_bytes(b"not in the bank")
    monkeypatch.setattr(sys, "argv", ["demo.py", "--file", str(unknown)])

    demo.main()
    out = capsys.readouterr().out
    assert len(seen["uids"]) == 42          # nothing excluded for an unknown file
    assert "[truth: unknown]" in out
