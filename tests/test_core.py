"""Checks of the signal-processing core against textbook results.  Run: python -m pytest -q tests"""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
