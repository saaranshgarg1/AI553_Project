"""Features: one fixed-length vector per clip.

Classical (course) features all use the same short-time analysis:
  pre-emphasis -> 25 ms frames every 10 ms -> window -> |DFT|^2 -> filterbank / LPC -> per-frame coefficients
  -> add delta and delta-delta -> mean and std over frames.

  mfcc   13 MFCC, 40 MEL-spaced filters (perceptual)                              -> 78 dims
  lfcc   20 LFCC, 20 LINEARLY spaced filters (equal resolution at high freq.)     -> 120 dims
         lfcc_low / lfcc_high: the same on 0-4 kHz / 4-8 kHz only                 -> 60 dims each
  lpcc   16 cepstral coeffs of an all-pole (LPC) model + log residual energy      -> 34 dims

Self-supervised: w2v-BERT 2.0 hidden states (25 layers), averaged over time   -> 25 x 1024
"""

import librosa
import numpy as np
import torch
from scipy.fft import dct
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

SR, N_FFT, WIN, HOP = 16000, 512, 400, 160


def deltas(c):
    """Stack coefficients with their first and second time-derivatives."""
    w = min(9, c.shape[1] - (1 - c.shape[1] % 2))           # odd window, <= number of frames
    if w < 3:
        return np.vstack([c, 0 * c, 0 * c])
    return np.vstack([c, librosa.feature.delta(c, width=w), librosa.feature.delta(c, width=w, order=2)])


def stats(frames):
    """(n_coeffs, n_frames) -> mean and std of every coefficient."""
    return np.concatenate([frames.mean(1), frames.std(1)])


# ---------------------------------------------------------------- MFCC
def mfcc(y):
    y = librosa.effects.preemphasis(y, coef=0.97)
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_fft=WIN, hop_length=HOP, n_mels=40, fmin=80, fmax=7600)
    return stats(deltas(m))


# ---------------------------------------------------------------- LFCC
def linear_filterbank(n_filters, fmin, fmax):
    """Triangular filters with centres equally spaced in Hz (MFCC spaces them on the mel scale)."""
    f = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    e = np.linspace(fmin, fmax, n_filters + 2)                # filter edges
    return np.array([np.clip(np.minimum((f - e[i]) / (e[i + 1] - e[i]), (e[i + 2] - f) / (e[i + 2] - e[i + 1])), 0, None)
                     for i in range(n_filters)])


def lfcc_frames(y, n_filters=20, fmin=0, fmax=8000):
    y = librosa.effects.preemphasis(y, coef=0.97)
    P = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=WIN, window="hamming")) ** 2
    logE = np.log(linear_filterbank(n_filters, fmin, fmax) @ P + 1e-10)
    return deltas(dct(logE, type=2, axis=0, norm="ortho"))   # keep all n_filters cepstra


# ---------------------------------------------------------------- LPCC (source-filter model)
def levinson(r, p):
    """Solve the Yule-Walker equations R a = -r for every frame (rows of r). Returns a (a0=1) and error."""
    a = np.zeros((len(r), p + 1)); a[:, 0] = 1
    err = r[:, 0].copy()
    for i in range(1, p + 1):
        k = -(r[:, i] + np.sum(a[:, 1:i] * r[:, i - 1:0:-1], axis=1)) / np.maximum(err, 1e-12)  # reflection coef.
        a[:, 1:i] = a[:, 1:i] + k[:, None] * a[:, i - 1:0:-1]
        a[:, i] = k
        err *= 1 - k ** 2
    return a, err


def lpc_to_cepstrum(a, n):
    """Cepstrum of the all-pole filter 1/A(z): c_m = alpha_m + sum_k (k/m) c_k alpha_(m-k)."""
    alpha, p = -a[:, 1:], a.shape[1] - 1
    c = np.zeros((len(a), n))
    for m in range(1, n + 1):
        c[:, m - 1] = (alpha[:, m - 1] if m <= p else 0) + sum(
            (k / m) * c[:, k - 1] * alpha[:, m - k - 1] for k in range(max(1, m - p), m))
    return c


def lpcc(y, p=16):
    y = librosa.effects.preemphasis(y, coef=0.97)
    frames = librosa.util.frame(np.pad(y, WIN // 2), frame_length=WIN, hop_length=HOP).T * np.hamming(WIN)
    frames = frames[(frames ** 2).sum(1) > 1e-8] if (frames ** 2).sum(1).max() > 1e-8 else frames
    r = np.fft.irfft(np.abs(np.fft.rfft(frames, 2 * N_FFT)) ** 2)[:, : p + 1]   # autocorrelation via FFT
    r[:, 0] *= 1 + 1e-9                                                         # numerical safety
    a, err = levinson(r, p)
    log_residual = np.log(err / np.maximum(r[:, 0], 1e-12) + 1e-12)            # the "source" part
    return stats(np.vstack([lpc_to_cepstrum(a, p).T, log_residual]))


def classical(y) -> dict:
    return {"mfcc": mfcc(y), "lfcc": stats(lfcc_frames(y)), "lpcc": lpcc(y),
            "lfcc_low": stats(lfcc_frames(y, 10, 0, 4000)), "lfcc_high": stats(lfcc_frames(y, 10, 4000, 8000))}


# ---------------------------------------------------------------- w2v-BERT 2.0
MODEL = "facebook/w2v-bert-2.0"


class W2VBert:
    """Frozen w2v-BERT 2.0. hidden_states[0] = input projection, hidden_states[l] = output of block l (1..24).
    TADA uses hidden_states[4]. fp16 on GPU (checked against fp32: cosine >= 0.9998 in every layer)."""

    def __init__(self):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.dev == "cuda" else torch.float32
        self.fe = AutoFeatureExtractor.from_pretrained(MODEL)            # 80-band log-mel filterbank
        self.model = Wav2Vec2BertModel.from_pretrained(MODEL, torch_dtype=dtype).to(self.dev).eval()

    def prepare(self, y):                                    # CPU part (filterbank)
        return self.fe(y, sampling_rate=SR, return_tensors="np")["input_features"]

    @torch.inference_mode()
    def embed(self, feats):                                  # GPU part -> (25, 1024)
        x = torch.from_numpy(feats).to(self.dev, self.model.dtype)
        hidden = self.model(input_features=x, output_hidden_states=True).hidden_states
        return torch.stack(hidden)[:, 0].float().mean(dim=1).cpu().numpy()   # average over time

    def __call__(self, y):
        return self.embed(self.prepare(y))
