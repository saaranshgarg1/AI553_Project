"""Classifiers and metrics.

kNN (TADA, training-free): keep the support vectors; label a query by majority vote of its
    k=21 nearest neighbours (Euclidean). Adding a new generator = adding its vectors.
GMM (classical): one Gaussian mixture per generator over frame-level LFCCs; pick the
    generator with the highest average log-likelihood (the ASVspoof 2019 baseline idea).
OOD score: mean distance to the k nearest support vectors (large = unknown generator).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def zscore(train, *others):
    """Standardise with support-set statistics (classical features have very different scales)."""
    mu, sd = train.mean(0), train.std(0) + 1e-8
    return [(x - mu) / sd for x in (train, *others)]


def knn_predict(Xtr, ytr, Xte, k=21, standardize=False):
    if standardize:
        Xtr, Xte = zscore(Xtr, Xte)
    return KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr).predict(Xte)


class GMMClassifier:
    def __init__(self, n_components=32, max_frames=20000, seed=0):
        self.n, self.max_frames, self.seed = n_components, max_frames, seed

    def fit(self, frames, y):                     # frames: list of (n_frames, d) arrays
        rng = np.random.default_rng(self.seed)
        allf = np.concatenate(frames)
        self.mu, self.sd = allf.mean(0), allf.std(0) + 1e-8
        self.classes = sorted(set(y))
        self.gmms = {}
        for c in self.classes:
            F = np.concatenate([f for f, l in zip(frames, y) if l == c])
            F = F[rng.choice(len(F), min(len(F), self.max_frames), replace=False)]
            self.gmms[c] = GaussianMixture(self.n, covariance_type="diag", max_iter=100, reg_covar=1e-3,
                                           random_state=self.seed).fit((F - self.mu) / self.sd)
        return self

    def predict(self, frames):
        scores = [[self.gmms[c].score((f - self.mu) / self.sd) for c in self.classes] for f in frames]
        return np.array(self.classes)[np.argmax(scores, axis=1)]


def ood_score(support, X, k=21):
    return NearestNeighbors(n_neighbors=k).fit(support).kneighbors(X)[0].mean(1)


def eer(y_true, scores):
    """Equal error rate: where false-alarm rate = miss rate. Returns (eer, threshold)."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    i = np.argmin(np.abs((1 - tpr) - fpr))
    return float((fpr[i] + 1 - tpr[i]) / 2), float(thr[i])


def auroc(y_true, scores):
    return float(roc_auc_score(y_true, scores))
