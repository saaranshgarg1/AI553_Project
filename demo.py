#!/usr/bin/env python3
"""Live demo: "which generator made this clip?"

    python demo.py                 # a random eval clip (it is left out of the reference bank)
    python demo.py --file x.wav    # any audio file, e.g. your own voice (16 kHz mono recommended)
    python demo.py --file demo_clips/A07.flac   # a shipped reference clip: recognised by file hash and
                                                # treated as the eval clip it was copied from (left out of the bank)

Steps: load audio -> w2v-BERT 2.0 layer 4 (frozen) -> average over time -> 21 nearest
neighbours among 5600 labelled eval clips (13 generators A07-A19 + human "bonafide") -> majority vote.
If the clip is farther from the bank than 95 % of the bank's own clips are (5 % false-alarm rate),
we say "unknown generator".
Saves demo_output.png (waveform, spectrogram, votes).
"""

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from tada import audio
from tada.data import ROOT, SYSTEMS, manifest, ssl
from tada.features import W2VBert

REFS = ROOT / "demo_clips" / "reference_clips.csv"   # filename, uid, label, speaker, sha256


def identify(path, refs=REFS):
    """uid of the reference clip this file is a byte-identical copy of, else None."""
    path = Path(path)
    if not path.exists() or not Path(refs).exists():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with open(refs, newline="") as f:
        return next((r["uid"] for r in csv.DictReader(f) if r["sha256"] == digest), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="audio file to analyse")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    pool = manifest().query("partition == 'eval'").reset_index(drop=True)   # A07-A19 + bona fide
    name = lambda s: "human speech" if s == "bonafide" else SYSTEMS[s][1]
    if args.file:
        y, sr = librosa.load(args.file, sr=16000, mono=True)
        uid = identify(args.file)                           # a copy of a bank clip is still that clip
        row = pool[pool.uid == uid].iloc[0] if uid else None
        truth, query_uid = (row.label, row.uid) if row is not None else ("unknown", None)
        if row is not None:
            print(f"Reference clip {row.uid}: true generator = {truth} ({name(truth)}), speaker {row.speaker}")
    else:
        row = pool.sample(1, random_state=args.seed).iloc[0]
        y, truth, query_uid = audio.load(ROOT / row.path), row.label, row.uid
        print(f"Clip {row.uid}: true generator = {truth} ({name(truth)}), speaker {row.speaker}")

    bank = pool[pool.uid != query_uid]                      # never compare a clip with itself
    X = ssl(4, bank.uid)                                    # precomputed layer-4 embeddings of the bank
    print("Embedding with w2v-BERT 2.0 (frozen) ...")
    q = W2VBert()(y)[4][None]                               # (1, 1024)

    nn = NearestNeighbors(n_neighbors=21).fit(X)
    dist, idx = nn.kneighbors(q)
    votes = Counter(bank.label.values[idx[0]])
    pred, n = votes.most_common(1)[0]
    bank_d = nn.kneighbors(X, n_neighbors=22)[0][:, 1:].mean(1)   # each bank clip vs its own neighbours
    pct = 100 * np.mean(bank_d < dist.mean())                  # how unusual is this distance?

    print(f"\nPrediction: {pred} ({name(pred)})  - {n}/21 votes   [truth: {truth}]")
    print("Votes:", ", ".join(f"{s}:{c}" for s, c in votes.most_common()))
    print("Nearest neighbours:")
    for d, i in zip(dist[0][:5], idx[0][:5]):
        r = bank.iloc[i]
        print(f"   {r.uid}  {r.label} ({name(r.label):<18}) speaker {r.speaker}  distance {d:.2f}")
    verdict = "UNKNOWN generator (too far from everything known)" if pct > 95 else "known generator"
    print(f"Mean distance to 21 neighbours: {dist.mean():.2f} - farther than {pct:.0f}% of known clips -> {verdict}")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3), gridspec_kw={"width_ratios": [1.2, 1.4, 1]})
    ax[0].plot(np.arange(len(y)) / 16000, y, lw=0.4); ax[0].set_title("waveform"); ax[0].set_xlabel("s")
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=128)), ref=np.max)
    ax[1].imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=-80, extent=[0, len(y) / 16000, 0, 8])
    ax[1].set_title("spectrogram (kHz)")
    labels, counts = zip(*votes.most_common())
    ax[2].barh(labels[::-1], counts[::-1], color="#2a78d6"); ax[2].set_title(f"kNN votes -> {pred}")
    fig.tight_layout(); fig.savefig("demo_output.png", dpi=130)
    print("Saved demo_output.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
