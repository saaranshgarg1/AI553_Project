#!/usr/bin/env python3
"""Extract features for every clip (and for degraded copies of the robustness test clips).

  features/<feat>.npy            classical features (mfcc, lfcc, lfcc_low, lfcc_high, lpcc), all clips
  features/lfcc_frames.npy       frame-level LFCC of the eval generators (for the GMM)
  features/ssl.npy               w2v-BERT 2.0, all 25 layers, all clips          (N, 25, 1024) float16
  features/<feat>_<cond>.npy     the same for the 1040 robustness test clips under each degradation

The GPU runs w2v-BERT; the CPU (worker processes) does decoding and filterbanks in parallel.
Takes ~15 min on a laptop with an RTX 2050. Existing files are skipped.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")          # one BLAS thread per worker process

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

import _path  # noqa: F401,E402
from tada import audio, features  # noqa: E402
from tada.data import FEATS, ROOT, eval_pool, manifest, random_split, save  # noqa: E402

_fe = None


def load_degraded(args):
    path, cond = args
    y = audio.load(ROOT / path)
    return y if cond == "clean" else audio.DEGRADATIONS[cond](y).astype(np.float32)


def classical(y):
    return features.classical(y)


def filterbank(y):                       # runs in worker processes (it is slow, pure-Python code)
    global _fe
    if _fe is None:
        from transformers import AutoFeatureExtractor
        _fe = AutoFeatureExtractor.from_pretrained(features.MODEL)
    return _fe(y, sampling_rate=16000, return_tensors="np")["input_features"]


def extract(paths, uids, cond, model, pool):
    suffix = "" if cond == "clean" else f"_{cond}"
    need_classical = not (FEATS / f"mfcc{suffix}.npy").exists()
    need_ssl = not (FEATS / f"ssl{suffix}.npy").exists()
    if not (need_classical or need_ssl):
        return
    ys = list(tqdm(pool.imap(load_degraded, [(p, cond) for p in paths], chunksize=8), total=len(paths), desc=cond))
    if need_classical:
        feats = pool.map(classical, ys, chunksize=8)
        for name in feats[0]:
            save(name + suffix, np.stack([f[name] for f in feats]), uids)
    if need_ssl:
        X = np.stack([model.embed(f).astype(np.float16) for f in tqdm(pool.imap(filterbank, ys, chunksize=4),
                                                                      total=len(ys), desc=f"{cond} w2v-BERT")])
        save("ssl" + suffix, X, uids)


def main():
    m, pool_df = manifest(), eval_pool()
    model = features.W2VBert()
    with Pool(8) as pool:
        extract(list(m.path), list(m.uid), "clean", model, pool)
        if not (FEATS / "lfcc_frames.npy").exists():       # frame-level LFCC for the GMM baseline
            ys = pool.map(load_degraded, [(p, "clean") for p in pool_df.path], chunksize=8)
            frames = pool.map(features.lfcc_frames, ys, chunksize=8)
            np.save(FEATS / "lfcc_frames.npy", np.concatenate([f.T for f in frames]).astype(np.float16))
            np.save(FEATS / "lfcc_frames.offsets.npy", np.cumsum([0] + [f.shape[1] for f in frames]))
        _, test = random_split(pool_df.label, seed=0)       # robustness test clips = split-0 test set
        sub = pool_df.iloc[test]
        for cond in audio.DEGRADATIONS:
            extract(list(sub.path), list(sub.uid), cond, model, pool)


if __name__ == "__main__":
    main()
