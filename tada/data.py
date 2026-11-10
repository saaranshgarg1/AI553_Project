"""Paths, the generator table, the clip list and the train/test splits."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"          # audio + manifest.csv (one row per clip)
FEATS = ROOT / "features"     # extracted feature matrices
RESULTS = ROOT / "results"    # json / tables / figures
for d in (FEATS, RESULTS / "figures", RESULTS / "tables"):
    d.mkdir(parents=True, exist_ok=True)

# ASVspoof 2019 LA generators: (type, waveform generator = "vocoder").
# From Wang et al. 2020, Table 1. A16 and A19 re-use the A04 and A06 algorithms.
SYSTEMS = {
    "A01": ("TTS", "WaveNet"),            "A02": ("TTS", "WORLD"),
    "A03": ("TTS", "WORLD"),              "A04": ("TTS", "Waveform concat."),
    "A05": ("VC", "WORLD"),               "A06": ("VC", "Spectral filtering"),
    "A07": ("TTS", "WORLD + GAN"),        "A08": ("TTS", "Neural source-filter"),
    "A09": ("TTS", "Vocaine"),            "A10": ("TTS", "WaveRNN"),
    "A11": ("TTS", "Griffin-Lim"),        "A12": ("TTS", "WaveNet"),
    "A13": ("TTS+VC", "Waveform filtering"), "A14": ("TTS+VC", "STRAIGHT"),
    "A15": ("TTS+VC", "WaveNet"),         "A16": ("TTS", "Waveform concat."),
    "A17": ("VC", "Waveform filtering"),  "A18": ("VC", "MFCC vocoder"),
    "A19": ("VC", "Spectral filtering"),
}
KNOWN = ["A01", "A02", "A03", "A04", "A05", "A06"]     # train/dev partitions
UNSEEN = [f"A{i:02d}" for i in range(7, 20)]           # eval partition only
SAME_AS = {"A16": "A04", "A19": "A06"}                 # "unseen" in name only


def manifest() -> pd.DataFrame:
    """uid, partition (train/dev/eval), label (system or 'bonafide'), speaker, path, duration."""
    return pd.read_csv(DATA / "manifest.csv")


def eval_pool() -> pd.DataFrame:
    """Main attribution task: the 13 eval generators A07-A19 (400 clips each)."""
    m = manifest()
    return m[(m.partition == "eval") & (m.label != "bonafide")].reset_index(drop=True)


def random_split(labels, seed):
    """80:20 split, stratified by class (as in the TADA paper)."""
    return next(StratifiedShuffleSplit(1, test_size=0.2, random_state=seed).split(labels, labels))


def speaker_split(labels, speakers, seed):
    """80:20 split by SPEAKER: no test speaker is ever in the support set."""
    return next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(labels, labels, speakers))


# ---- feature storage: features/<name>.npy rows match features/<name>.uids.txt ----------
def save(name: str, X: np.ndarray, uids):
    np.save(FEATS / f"{name}.npy", X)
    (FEATS / f"{name}.uids.txt").write_text("\n".join(uids))


_cache = {}


def load(name: str, uids) -> np.ndarray:
    """Load a feature matrix and return the rows for `uids`, in that order."""
    if name not in _cache:
        X = np.load(FEATS / f"{name}.npy")
        order = (FEATS / f"{name}.uids.txt").read_text().split("\n")
        _cache[name] = (X, {u: i for i, u in enumerate(order)})
    X, pos = _cache[name]
    return X[[pos[u] for u in uids]].astype(np.float32)


def ssl(layer: int, uids, cond: str = "") -> np.ndarray:
    """One w2v-BERT layer for the given clips (the store holds all 25 layers)."""
    return load("ssl" + (f"_{cond}" if cond else ""), uids)[:, layer]
