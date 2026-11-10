#!/usr/bin/env python3
"""Figures for the report and slides (results/figures/*.png and *.pdf)."""

import json

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

import _path  # noqa: F401
from tada import audio
from tada.data import KNOWN, RESULTS, ROOT, SAME_AS, SYSTEMS, UNSEEN, eval_pool, load, manifest, ssl

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "axes.axisbelow": True, "savefig.dpi": 220, "savefig.bbox": "tight"})
R = json.loads((RESULTS / "results.json").read_text())


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / "figures" / f"{name}.{ext}")
    plt.close(fig)


def heatmap(ax, M, xt, yt, fmt="{:.0f}"):
    ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > 55 else "black")
    ax.set_xticks(range(len(xt)), xt)
    ax.set_yticks(range(len(yt)), yt)
    ax.grid(False)


def spectrograms():
    ev = manifest().query("partition == 'eval' and duration > 3")
    picks = [("bonafide", "Bona fide (human)"), ("A10", "A10: Tacotron + WaveRNN"), ("A11", "A11: Tacotron + Griffin-Lim"),
             ("A16", "A16: unit-selection concatenation"), ("A17", "A17: VC + waveform filtering"),
             ("A08", "A08: neural source-filter")]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 4.8), sharey=True, gridspec_kw={"hspace": 0.45})
    for ax, (lab, title) in zip(axes.flat, picks):
        y = audio.load(ROOT / ev[ev.label == lab].sort_values("uid").path.iloc[0])[:40000]
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=128)), ref=np.max)
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=-80, extent=[0, len(y) / 16000, 0, 8])
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("kHz")
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    save(fig, "spectrograms")


def feature_bars():
    at = R["attribution"]
    rows = [("MFCC", "mfcc"), ("LPCC", "lpcc"), ("LFCC 0-4 kHz", "lfcc_low"), ("LFCC", "lfcc"),
            ("LFCC 4-8 kHz", "lfcc_high"), ("w2v-BERT L4", "ssl")]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    x = np.arange(len(rows))
    for i, (key, lab, col) in enumerate([("random", "random split", BLUE), ("speaker", "unseen speakers", ORANGE)]):
        v = [np.mean(at["features"][f][key]) * 100 for _, f in rows]
        ax.bar(x + (i - 0.5) * 0.38, v, 0.36, color=col, label=lab)
        for xi, vi in zip(x, v):
            ax.text(xi + (i - 0.5) * 0.38, vi + 1, f"{vi:.0f}", ha="center", fontsize=7)
    ax.axhline(100 / 13, color=GREY, ls="--", lw=1)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 108); ax.legend(loc="upper left", frameon=False)
    ax.set_title("Attribution of 13 generators (kNN, k = 21); dashed line = chance", loc="left")
    save(fig, "feature_comparison")


def layer_sweep():
    L = R["attribution"]["layers"]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.plot(np.array(L["random"]) * 100, "o-", ms=3.5, color=BLUE, label="random split")
    ax.plot(np.array(L["speaker"]) * 100, "s-", ms=3.5, color=ORANGE, label="unseen speakers")
    ax.axvline(4, color=GREY, ls="--", lw=1)
    ax.text(4.3, 60, "TADA uses layer 4", fontsize=7.5, color="#52514e")
    ax.set_xlabel("w2v-BERT 2.0 layer (0 = input)"); ax.set_ylabel("accuracy (%)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Generator identity is available from layer 3 on (only the last layer loses it)", loc="left")
    save(fig, "layer_sweep")


def confusion():
    at = R["attribution"]
    cls = at["classes"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for ax, f, title in zip(axes, ("ssl", "lfcc"), ("w2v-BERT layer 4", "LFCC")):
        M = np.array(at["confusion"][f]) * 100
        heatmap(ax, M, cls, [f"{c} {SYSTEMS[c][1]}" for c in cls])
        ax.tick_params(axis="x", rotation=90)
        ax.set_title(f"{title} (accuracy {np.trace(M) / len(cls):.1f}%)", loc="left")
        ax.set_xlabel("predicted")
    axes[1].set_yticklabels([])
    save(fig, "confusion")


def ood():
    od = R["ood"]
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    x = np.arange(len(UNSEEN))
    for i, (f, lab, col) in enumerate([("ssl", "w2v-BERT L4", BLUE), ("lfcc", "LFCC", ORANGE)]):
        ax.bar(x + (i - 0.5) * 0.38, [od[f]["per_system"][s] * 100 for s in UNSEEN], 0.36, color=col, label=lab)
    ax.axhline(50, color=GREY, ls="--", lw=1)
    ax.set_xticks(x, [s + (f"\n= {SAME_AS[s]}" if s in SAME_AS else "") for s in UNSEEN])
    ax.set_ylabel("AUROC (%)"); ax.set_ylim(0, 105); ax.legend(frameon=False, loc="lower left", ncol=2)
    ax.set_title("Is this generator unknown? (AUROC per unseen system; dashed = chance)", loc="left")
    save(fig, "ood_per_system")

    M = np.array([[od["forced_choice"][s][t] * 100 for t in KNOWN] for s in UNSEEN])
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    heatmap(ax, M, [f"{t}\n{SYSTEMS[t][1].split()[0]}" for t in KNOWN], [f"{s} {SYSTEMS[s][1]}" for s in UNSEEN])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("nearest KNOWN generator (% of clips)")
    ax.set_title("Where do unseen generators land?", loc="left")
    save(fig, "ood_forced_choice")


def robustness():
    rb = R["robustness"]
    conds = ["clean"] + list(audio.DEGRADATIONS)
    feats = [("mfcc", "MFCC"), ("lfcc", "LFCC"), ("lpcc", "LPCC"), ("ssl", "w2v-BERT L4")]
    M = np.array([[rb[f][c] * 100 for c in conds] for f, _ in feats])
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    heatmap(ax, M, [audio.LABELS[c] for c in conds], [n for _, n in feats])
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Accuracy (%) when the test audio goes through an unseen channel", loc="left")
    save(fig, "robustness")


def tsne():
    pool = eval_pool()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for ax, (X, title) in zip(axes, [(ssl(4, pool.uid), "w2v-BERT layer 4"), (load("mfcc", pool.uid), "MFCC")]):
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        Z = TSNE(perplexity=40, init="pca", random_state=0).fit_transform(X)
        for i, s in enumerate(UNSEEN):
            k = pool.label.values == s
            ax.scatter(*Z[k].T, s=3, alpha=0.5, color=plt.cm.tab20(i / 13))
            ax.text(*np.median(Z[k], 0), s, fontsize=7.5, weight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title(title, loc="left")
    save(fig, "tsne")


if __name__ == "__main__":
    for f in (spectrograms, feature_bars, layer_sweep, confusion, ood, robustness, tsne):
        f()
        print("done", f.__name__)
