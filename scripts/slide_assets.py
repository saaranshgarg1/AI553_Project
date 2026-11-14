#!/usr/bin/env python3
"""Large, slide-ready chart assets in slides/assets/.

Reads only results/results.json and results/figures/spectrograms.png.
No titles are baked in: slide titles belong to the deck.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slides" / "assets"
R = json.loads((ROOT / "results" / "results.json").read_text())

# palette: light surface, ink, secondary ink, hairline grid, fixed series slots
BG, INK, SEC, GRID = "#FBFAF6", "#182B3A", "#51616C", "#DCE1E2"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # random/SSL, speaker/LFCC

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": SEC, "axes.edgecolor": GRID,
    "xtick.color": SEC, "ytick.color": SEC, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 12, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11.5, "lines.linewidth": 2.4, "savefig.dpi": 230,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def layers():
    """Accuracy vs w2v-BERT layer; layer 4 (used by TADA) marked."""
    L = R["attribution"]["layers"]
    x = np.arange(len(L["random"]))
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.axvspan(3.55, 4.45, color="#E7EDF3", lw=0, zorder=0)
    rnd, spk = np.array(L["random"]) * 100, np.array(L["speaker"]) * 100
    ax.plot(x, rnd, "-o", color=BLUE, ms=4.5, label="Random split")
    ax.plot(x, spk, "-s", color=ORANGE, ms=4.5, label="Unseen speakers")
    ax.text(4.5, 43.5, "TADA uses layer 4", fontsize=10.5, color=SEC, va="bottom")
    for y, txt, col in [(rnd[-1], f"Random split · {rnd[-1]:.1f}%", BLUE),
                        (spk[-1], f"Unseen speakers · {spk[-1]:.1f}%", ORANGE)]:
        ax.annotate(txt, xy=(24, y), xytext=(24.9, y + (2.6 if col == BLUE else -2.6)),
                    ha="left", va="center", fontsize=12, color=col, annotation_clip=False)
    ax.set_xlim(-0.4, 25.6)
    ax.set_ylim(40, 100)
    ax.set_xticks([0, 4, 24])
    ax.set_xlabel("w2v-BERT 2.0 layer (0 = input)")
    ax.set_ylabel("Generator-ID accuracy (%)")
    ax.legend(loc="lower right", frameon=False)
    return save(fig, "layers")


def ood_distance():
    """Per-group normalised density of the kNN OOD score (no threshold drawn)."""
    s = R["ood"]["scores"]
    known, unseen = np.array(s["known"]), np.array(s["unknown"])
    bins = np.linspace(min(known.min(), unseen.min()), max(known.max(), unseen.max()), 46)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for v, col, lab in [(known, BLUE, "Known generators (A01–A06)"),
                        (unseen, ORANGE, "Unseen generators (A07–A19)")]:
        ax.hist(v, bins=bins, density=True, color=col, alpha=0.30, edgecolor=col,
                linewidth=2.2, label=lab)
    ax.set_xlabel("Mean distance to the 21 nearest known clips  (higher = less familiar)")
    ax.set_ylabel("Density (normalised within each group)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(bottom=0)
    return save(fig, "ood_distance")


def confusion():
    """Two row-normalised 13-class confusion matrices on one 0-100 blue ramp."""
    at = R["attribution"]
    cls = at["classes"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.7), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    im = None
    for ax, key, head in [(axes[0], "ssl", "w2v-BERT L4"), (axes[1], "lfcc", "LFCC")]:
        M = np.array(at["confusion"][key]) * 100
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100)
        acc = np.trace(M) / len(cls)
        ax.set_title(f"{head} · {acc:.1f}% correct", loc="left", fontsize=12.5, color=INK, pad=8)
        ax.set_xticks(range(len(cls)), cls, rotation=90, fontsize=10.5)
        ax.set_yticks(range(len(cls)), cls, fontsize=10.5)
        ax.set_xticks(np.arange(-.5, len(cls), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(cls), 1), minor=True)
        ax.grid(which="minor", color=BG, linewidth=1.6)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        ax.set_xlabel("predicted", fontsize=12)
        if key == "lfcc":                        # only the large, readable errors get numbers
            for i, j in zip(*np.where((M >= 15) & ~np.eye(len(cls), dtype=bool))):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                        fontsize=9.5, color="white", weight="bold")
    axes[0].set_ylabel("true", fontsize=12)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.036, pad=0.02, ticks=[0, 25, 50, 75, 100])
    cb.set_label("clips of the true class (%)", fontsize=11)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=10.5)
    return save(fig, "confusion")


def feature_dumbbell():
    """Mean accuracy per feature, random vs unseen speakers, on one 0-100 axis."""
    at = R["attribution"]["features"]
    rows = [("LPCC", "lpcc"), ("MFCC", "mfcc"), ("LFCC 0–4 kHz", "lfcc_low"),
            ("LFCC", "lfcc"), ("LFCC 4–8 kHz", "lfcc_high"), ("w2v-BERT L4", "ssl")]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.axvline(100 / 13, color=SEC, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(100 / 13 + 1.6, 5.42, "chance 7.7%", fontsize=10.5, color=SEC, va="center")
    for yi, (lab, key) in zip(y, rows):
        r = np.mean(at[key]["random"]) * 100
        s = np.mean(at[key]["speaker"]) * 100
        ax.plot([s, r], [yi, yi], color="#C7CED2", lw=2.2, zorder=2, solid_capstyle="round")
        ax.plot(r, yi, "o", ms=9, color=BLUE, zorder=3)
        ax.plot(s, yi, "s", ms=8.5, color=ORANGE, zorder=3)
        if key in ("lfcc_high", "ssl"):          # exact values, last two rows only
            ax.annotate(f"{r:.1f}", (r, yi), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=10.5, color=BLUE)
            ax.annotate(f"{s:.1f}", (s, yi), xytext=(0, -16), textcoords="offset points",
                        ha="center", fontsize=10.5, color=ORANGE)
    ax.plot([], "o", color=BLUE, label="Random split")
    ax.plot([], "s", color=ORANGE, label="Unseen speakers")
    ax.set_yticks(y, [lab for lab, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(5.75, -0.65)
    ax.set_xlim(0, 100)
    ax.set_xlabel("kNN attribution accuracy (%)")
    ax.legend(loc="upper right", frameon=False)
    return save(fig, "feature_dumbbell")


# plotting rectangles of spectrograms.png (image read at 1714x984; axis/title margins excluded)
SPECTRO = {"spectro_human": (100, 60, 563, 393),
           "spectro_wavernn": (654, 60, 1116, 393),
           "spectro_griffinlim": (1207, 60, 1670, 393)}


def spectrograms():
    """Pure crops of the existing spectrogram panels, upscaled to 1800 px wide."""
    src = Image.open(ROOT / "results" / "figures" / "spectrograms.png").convert("RGB")
    assert src.size == (1714, 984), src.size
    paths = []
    for name, box in SPECTRO.items():
        im = src.crop(box)
        w, h = im.size
        im = im.resize((1800, round(h * 1800 / w)), Image.LANCZOS)
        p = OUT / f"{name}.png"
        im.save(p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    made = [layers(), ood_distance(), confusion(), feature_dumbbell(), *spectrograms()]
    for p in made:
        with Image.open(p) as im:
            print(f"{p.relative_to(ROOT)}  {im.size[0]}x{im.size[1]}  {p.stat().st_size // 1024} KB")
