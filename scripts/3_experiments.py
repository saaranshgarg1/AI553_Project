#!/usr/bin/env python3
"""All experiments. Writes results/results.json and results/tables/*.tex.

1. Attribution   13 eval generators (A07-A19), kNN k=21, 5 random splits and 5 speaker-disjoint splits.
                 Classical features vs w2v-BERT layer 4 (TADA); GMM on LFCC frames; sweep over all 25 layers.
2. Unseen generators (OOD)
                 support = A01-A06 (train partition), known test = A01-A06 (dev partition, new speakers),
                 unknown = A07-A19. Score = mean distance to 21 nearest support vectors.
                 Control: A16/A19 use the same algorithms as A04/A06.
3. Robustness    clean support set, test clips passed through each channel degradation.
"""

import json

import numpy as np

import _path  # noqa: F401
from tada import audio
from tada.data import FEATS, KNOWN, RESULTS, SAME_AS, UNSEEN, eval_pool, load, manifest, random_split, speaker_split, ssl
from tada.models import GMMClassifier, auroc, eer, knn_predict, ood_score, zscore

LAYER, SEEDS = 4, range(5)
CLASSICAL = ["mfcc", "lfcc", "lfcc_low", "lfcc_high", "lpcc"]
NAMES = {"mfcc": "MFCC", "lfcc": "LFCC", "lfcc_low": "LFCC 0--4 kHz", "lfcc_high": "LFCC 4--8 kHz",
         "lpcc": "LPCC", "ssl": "w2v-BERT layer 4 (TADA)"}


def get(feat, uids, cond=""):
    return ssl(LAYER, uids, cond) if feat == "ssl" else load(feat + (f"_{cond}" if cond else ""), uids)


def accuracy(X, y, splits, standardize):
    return [float(np.mean(knn_predict(X[a], y[a], X[b], standardize=standardize) == y[b])) for a, b in splits]


def attribution():
    pool = eval_pool()
    y, uids = pool.label.values, pool.uid
    splits = {"random": [random_split(y, s) for s in SEEDS],
              "speaker": [speaker_split(y, pool.speaker, s) for s in SEEDS]}
    out = {"features": {}, "layers": {"random": [], "speaker": []}}
    for f in CLASSICAL + ["ssl"]:
        X = get(f, uids)
        out["features"][f] = {k: accuracy(X, y, s, standardize=f != "ssl") for k, s in splits.items()}
        print(f, {k: round(np.mean(v), 3) for k, v in out["features"][f].items()})

    # GMM on frame-level LFCC (classical statistical model, trained with EM)
    F = np.load(FEATS / "lfcc_frames.npy")
    off = np.load(FEATS / "lfcc_frames.offsets.npy")
    frames = [F[off[i]:off[i + 1]].astype(np.float32) for i in range(len(pool))]
    out["gmm_lfcc"] = []
    for a, b in splits["random"]:
        g = GMMClassifier().fit([frames[i] for i in a], y[a])
        out["gmm_lfcc"].append(float(np.mean(g.predict([frames[i] for i in b]) == y[b])))
    print("GMM", np.mean(out["gmm_lfcc"]))

    # which layer carries the fingerprint?
    for layer in range(25):
        X = ssl(layer, uids)
        for k, s in splits.items():
            out["layers"][k].append(float(np.mean(accuracy(X, y, s, False))))

    # confusion matrices (split 0)
    a, b = splits["random"][0]
    classes = sorted(set(y))
    out["classes"] = classes
    out["confusion"] = {}
    for f in ("ssl", "lfcc"):
        X = get(f, uids)
        pred = knn_predict(X[a], y[a], X[b], standardize=f != "ssl")
        out["confusion"][f] = [[float(np.mean(pred[y[b] == t] == p)) for p in classes] for t in classes]
    return out


def unseen_generators():
    m = manifest()
    sup = m[(m.partition == "train") & m.label.isin(KNOWN)]
    known = m[(m.partition == "dev") & m.label.isin(KNOWN)]
    unk = m[(m.partition == "eval") & m.label.isin(UNSEEN)]
    new = ~unk.label.isin(list(SAME_AS)).values                 # the 11 genuinely new systems
    out = {}
    for f in ("mfcc", "lfcc", "ssl"):
        Xs, Xk, Xu = get(f, sup.uid), get(f, known.uid), get(f, unk.uid)
        if f != "ssl":
            Xs, Xk, Xu = zscore(Xs, Xk, Xu)
        sk, su = ood_score(Xs, Xk), ood_score(Xs, Xu)
        y_all = np.r_[np.zeros(len(sk)), np.ones(new.sum())]
        out[f] = {"auroc": auroc(y_all, np.r_[sk, su[new]]), "eer": eer(y_all, np.r_[sk, su[new]])[0],
                  "per_system": {s: auroc(np.r_[np.zeros(len(sk)), np.ones((unk.label == s).sum())],
                                          np.r_[sk, su[unk.label.values == s]]) for s in UNSEEN},
                  "closed_set_acc_known": float(np.mean(knn_predict(Xs, sup.label.values, Xk) == known.label.values))}
        if f == "ssl":
            out["scores"] = {"known": sk.tolist(), "unknown": su.tolist(), "labels": unk.label.tolist()}
            pred = knn_predict(Xs, sup.label.values, Xu)        # forced choice: nearest KNOWN generator
            out["forced_choice"] = {s: {t: float(np.mean(pred[unk.label.values == s] == t)) for t in KNOWN}
                                    for s in UNSEEN}
        print(f, "AUROC", round(out[f]["auroc"], 3), "EER", round(out[f]["eer"], 3),
              "A16", round(out[f]["per_system"]["A16"], 2), "A19", round(out[f]["per_system"]["A19"], 2))
    return out


def robustness():
    pool = eval_pool()
    y, uids = pool.label.values, pool.uid.values
    a, b = random_split(y, seed=0)
    out = {}
    for f in ("mfcc", "lfcc", "lpcc", "ssl"):
        Xc = get(f, uids)
        out[f] = {"clean": float(np.mean(knn_predict(Xc[a], y[a], Xc[b], standardize=f != "ssl") == y[b]))}
        for cond in audio.DEGRADATIONS:
            Xt = get(f, uids[b], cond)
            out[f][cond] = float(np.mean(knn_predict(Xc[a], y[a], Xt, standardize=f != "ssl") == y[b]))
        print(f, {k: round(v, 2) for k, v in out[f].items()})
    return out


def latex_table(path, header, rows, caption, label):
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\toprule", " & ".join(header) + r" \\",
             r"\midrule"] + [" & ".join(r) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (RESULTS / "tables" / path).write_text("\n".join(lines))


def pm(v):
    return f"{np.mean(v) * 100:.1f} $\\pm$ {np.std(v) * 100:.1f}"


def main():
    res = {"attribution": attribution(), "ood": unseen_generators(), "robustness": robustness()}
    (RESULTS / "results.json").write_text(json.dumps(res, indent=1))

    at = res["attribution"]
    rows = [[NAMES[f], pm(at["features"][f]["random"]), pm(at["features"][f]["speaker"])] for f in CLASSICAL]
    rows.append(["LFCC frames + GMM (32 comp.)", pm(at["gmm_lfcc"]), "--"])
    rows.append([r"\textbf{" + NAMES["ssl"] + "}", pm(at["features"]["ssl"]["random"]),
                 pm(at["features"]["ssl"]["speaker"])])
    latex_table("attribution.tex", ["Feature", "Random split", "Unseen speakers"], rows,
                "Attribution accuracy (\\%) over the 13 ASVspoof 2019 eval generators (kNN, $k=21$; mean $\\pm$ std "
                "over 5 splits). ``Unseen speakers'': test speakers never appear in the support set.", "tab:attr")

    od = res["ood"]
    rows = [[NAMES[f], f"{od[f]['auroc'] * 100:.1f}", f"{od[f]['eer'] * 100:.1f}",
             f"{od[f]['per_system']['A16'] * 100:.0f}", f"{od[f]['per_system']['A19'] * 100:.0f}"]
            for f in ("mfcc", "lfcc", "ssl")]
    latex_table("ood.tex", ["Feature", "AUROC", "EER", "A16 AUROC", "A19 AUROC"], rows,
                "Detecting unseen generators (\\%). Known: A01--A06; unknown: the 11 genuinely new eval generators. "
                "A16/A19 re-use the A04/A06 algorithms, so an AUROC near 50 means they are correctly seen as known.",
                "tab:ood")

    rb = res["robustness"]
    conds = ["clean"] + list(audio.DEGRADATIONS)
    rows = [[audio.LABELS[c].replace("\u03bc", "$\\mu$")] + [f"{rb[f][c] * 100:.1f}" for f in ("mfcc", "lfcc", "lpcc", "ssl")] for c in conds]
    latex_table("robustness.tex", ["Test channel", "MFCC", "LFCC", "LPCC", "w2v-BERT L4"], rows,
                "Attribution accuracy (\\%) when test audio passes through a channel the (clean) support set "
                "never saw. 13 generators, 1040 test clips.", "tab:robust")


if __name__ == "__main__":
    main()
