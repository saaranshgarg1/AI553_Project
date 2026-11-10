"""Audio loading and the channel degradations used in the robustness test.

Each degradation is a course topic:
  telephone  resample 16 kHz -> 8 kHz -> 16 kHz   (sampling theorem: everything above 4 kHz is lost)
  quant8     uniform 8-bit quantisation            (SQNR ~ 6.02*b + 1.76 dB)
  mulaw8     mu-law 8-bit companding (G.711)       (non-uniform quantisation)
  noise20/10 white noise at 20 / 10 dB SNR
  lp4k/hp4k  Butterworth low-/high-pass at 4 kHz   (LTI filtering: keep only one band)
  mp3        MP3 at 32 kbit/s                      (perceptual coding / masking)
"""

import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

SR = 16000


def load(path) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, f"expected 16 kHz audio, got {sr}"
    return y.mean(axis=1)


def quantize(y, bits=8):
    step = 2.0 / 2 ** bits                               # 2^bits levels over [-1, 1)
    return (np.floor(np.clip(y, -1, 1 - 1e-9) / step) + 0.5) * step


def mulaw(y, bits=8, mu=255.0):
    c = np.sign(y) * np.log1p(mu * np.abs(np.clip(y, -1, 1))) / np.log1p(mu)   # compress
    c = quantize(c, bits)                                                        # quantise
    return np.sign(c) * np.expm1(np.abs(c) * np.log1p(mu)) / mu                 # expand


def add_noise(y, snr_db, seed=0):
    noise = np.random.default_rng(seed).standard_normal(len(y))
    return y + noise * np.sqrt(np.mean(y ** 2) / 10 ** (snr_db / 10))


def bandpass(y, kind, cutoff=4000):
    return sosfiltfilt(butter(8, cutoff, btype=kind, fs=SR, output="sos"), y)


def mp3(y, bitrate="32k"):
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    enc = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                          "-b:a", bitrate, "-f", "mp3", "-"], input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "-", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         input=enc, capture_output=True, check=True).stdout
    out = np.zeros_like(y)
    dec = np.frombuffer(dec, dtype=np.float32)[: len(y)]
    out[: len(dec)] = dec
    return out


DEGRADATIONS = {
    "telephone": lambda y: resample_poly(resample_poly(y, 1, 2), 2, 1)[: len(y)],
    "quant8": quantize,
    "mulaw8": mulaw,
    "noise20": lambda y: add_noise(y, 20),
    "noise10": lambda y: add_noise(y, 10),
    "lp4k": lambda y: bandpass(y, "lowpass"),
    "hp4k": lambda y: bandpass(y, "highpass"),
    "mp3_32k": mp3,
}
LABELS = {"clean": "Clean", "telephone": "8 kHz resample", "quant8": "Uniform 8-bit", "mulaw8": "μ-law 8-bit",
          "noise20": "Noise 20 dB", "noise10": "Noise 10 dB", "lp4k": "Low-pass 4 kHz", "hp4k": "High-pass 4 kHz",
          "mp3_32k": "MP3 32 kbps"}
