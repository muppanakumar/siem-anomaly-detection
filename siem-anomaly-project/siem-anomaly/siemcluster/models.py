"""Stage 5 - unsupervised clustering-based anomaly scorers (higher score = more anomalous).

Shared principle ("cluster-mass prior"): over-cluster the unlabelled training window, then call a cluster a
*normal mode* only if it holds at least `min_share` of the traffic.  Small clusters are the training
data's own rare behaviours - exactly where faults and attacks live - so they are *not* allowed to define
normal.  A session is scored by how far it sits from the nearest normal mode.  No labels are used.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


class KMeansScorer:
    name = "kmeans"

    def __init__(self, k=20, min_share=0.01, seed=0):
        self.k, self.min_share, self.seed = k, min_share, seed

    def fit(self, X):
        self.km = KMeans(self.k, n_init=10, random_state=self.seed).fit(X)
        lab = self.km.labels_
        self.share_ = np.bincount(lab, minlength=self.k) / len(X)
        self.normal_ = self.share_ >= self.min_share
        if not self.normal_.any():                               # degenerate: fall back to the biggest cluster
            self.normal_[self.share_.argmax()] = True
        d = self.km.transform(X)[:, self.normal_]
        self.scale_ = float(np.percentile(d.min(1)[self.normal_[lab]], 75)) + 1e-3   # robust global spread
        self.labels_ = lab
        self.train_score_ = self.score(X)
        return self

    def assign(self, X):
        """Nearest cluster of ANY size (for display) and distance to it."""
        d = self.km.transform(X)
        return d.argmin(1), d.min(1)

    def assign_normal(self, X):
        """Nearest NORMAL mode (global cluster index) and distance to it."""
        d = self.km.transform(X)
        idx = np.where(self.normal_)[0]
        j = d[:, idx].argmin(1)
        return idx[j], d[np.arange(len(X)), idx[j]]

    def score(self, X):
        d = self.km.transform(X)[:, self.normal_].min(1)
        return np.log1p(d / self.scale_)


class GMMScorer:
    name = "gmm"

    def __init__(self, k=20, min_share=0.01, seed=0):
        self.k, self.min_share, self.seed = k, min_share, seed

    def fit(self, X):
        g = GaussianMixture(self.k, covariance_type="diag", reg_covar=1e-2, n_init=3, random_state=self.seed).fit(X)
        keep = g.weights_ >= self.min_share
        if not keep.any():
            keep[g.weights_.argmax()] = True
        self.gmm = g
        self.keep = keep
        self.w = g.weights_[keep] / g.weights_[keep].sum()
        return self

    def score(self, X):
        g, k = self.gmm, self.keep
        # log-density under the retained (normal) components only
        logp = np.stack([
            -0.5 * (np.log(2 * np.pi * g.covariances_[j]).sum() + (((X - g.means_[j]) ** 2) / g.covariances_[j]).sum(1))
            for j in np.where(k)[0]], 1) + np.log(self.w)
        m = logp.max(1, keepdims=True)
        return -(m[:, 0] + np.log(np.exp(logp - m).sum(1)))


class HDBSCANKnnScorer:
    """Density-based variant: HDBSCAN finds dense behavioural clusters in a training subsample; only clusters
    with >= min_share of the subsample are normal.  Sessions are scored by k-NN distance to those core points."""
    name = "hdbscan"

    def __init__(self, min_cluster_size=15, min_share=0.01, k=5, max_fit=4000, seed=0):
        self.mcs, self.min_share, self.k, self.max_fit, self.seed = min_cluster_size, min_share, k, max_fit, seed

    def fit(self, X):
        rng = np.random.RandomState(self.seed)
        sub = X[rng.choice(len(X), min(len(X), self.max_fit), replace=False)]
        hd = HDBSCAN(min_cluster_size=self.mcs, min_samples=5, copy=True).fit(sub)
        lab = hd.labels_
        sizes = np.bincount(lab[lab >= 0], minlength=lab.max() + 1) / len(sub) if (lab >= 0).any() else np.array([])
        good = np.where(sizes >= self.min_share)[0]
        keep = np.isin(lab, good)
        if not keep.any():
            keep = lab >= 0
        self.core, self.core_labels = sub[keep], lab[keep]
        self.n_clusters_, self.n_normal_clusters_ = int(lab.max() + 1), int(len(good))
        self.noise_frac_ = float((lab < 0).mean())
        self.nn = NearestNeighbors(n_neighbors=self.k + 1).fit(self.core)
        d, _ = self.nn.kneighbors(self.core)
        self.ref = float(np.percentile(d[:, 1:].mean(1), 75)) + 1e-3
        return self

    def score(self, X):
        d, _ = self.nn.kneighbors(X, n_neighbors=self.k)
        return np.log1p(d.mean(1) / self.ref)


class IForestScorer:
    name = "isolation_forest"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X):
        self.m = IsolationForest(n_estimators=300, random_state=self.seed).fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


class PCAReconScorer:
    """Classic baseline (Xu et al., SOSP'09): squared prediction error in the residual PCA subspace."""
    name = "pca_spe"

    def __init__(self, var=0.8):
        self.var = var

    def fit(self, X):
        self.p = PCA(self.var, svd_solver="full").fit(X)
        return self

    def score(self, X):
        r = X - self.p.inverse_transform(self.p.transform(X))
        return np.log1p((r ** 2).sum(1))
