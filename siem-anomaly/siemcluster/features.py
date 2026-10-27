"""Stage 4 - sequence -> numeric features.

Three blocks, all fitted on the training window only:
  1. event-count block   : log(1+count) of each template, IDF-weighted (rare events weigh more)
  2. transition block    : log(1+count) of ordered event bigrams incl. BOS/EOS markers, IDF-weighted,
                           plus a counter of transitions never seen in training
  3. shape block         : #events, duration, #distinct hosts, longest silent gap, #OOV events
OOV (unseen) templates are never dropped: they get their own column and a count in the shape block.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .sessions import Sessions

BOS, EOS = -10, -11


def _bigrams(t: np.ndarray):
    seq = [BOS, *t.tolist(), EOS]
    return list(zip(seq[:-1], seq[1:]))


class SessionFeaturizer:
    def __init__(self, n_templates: int, min_bigram_df: int = 3, w_bigram: float = 0.6,
                 w_shape: float = 0.5, one_sided_timing: bool = False):
        self.V, self.min_bigram_df = n_templates, min_bigram_df
        self.w_bigram, self.w_shape = w_bigram, w_shape
        # one_sided_timing: only *slower-than-usual* lifecycles / longer gaps count as deviations (z clipped at 0)
        self.one_sided_timing = one_sided_timing

    # -- raw blocks -----------------------------------------------------------------------
    def _counts(self, S: Sessions):
        X = np.zeros((len(S), self.V + 1), np.float32)          # last col = OOV
        for i, t in enumerate(S.tids):
            for v in t:
                X[i, v if 0 <= v < self.V else self.V] += 1
        return X

    def _bigram_counts(self, S: Sessions):
        B = len(self.bigram_index)
        X = np.zeros((len(S), B + 1), np.float32)               # last col = unseen transitions
        for i, t in enumerate(S.tids):
            for bg in _bigrams(t):
                X[i, self.bigram_index.get(bg, B)] += 1
        return X

    def _shape(self, S: Sessions):
        rows = []
        for t, d, h in zip(S.tids, S.dts, S.hosts):
            gaps = np.diff(d) if len(d) > 1 else np.zeros(1)
            rows.append([np.log1p(len(t)), np.log1p(d[-1] if len(d) else 0), len({x for x in h if x}),
                         np.log1p(gaps.max()), float((t < 0).sum())])
        return np.asarray(rows, np.float32)

    # -- sklearn-style API ----------------------------------------------------------------
    def fit(self, S: Sessions):
        n = len(S)
        c = self._counts(S)
        self.idf_c = np.log((1 + n) / (1 + (c > 0).sum(0))) + 1
        df = {}
        for t in S.tids:
            for bg in set(_bigrams(t)):
                df[bg] = df.get(bg, 0) + 1
        keep = sorted(bg for bg, k in df.items() if k >= self.min_bigram_df)
        self.bigram_index = {bg: i for i, bg in enumerate(keep)}
        self.bigram_names = keep
        b = self._bigram_counts(S)
        self.idf_b = np.log((1 + n) / (1 + (b > 0).sum(0))) + 1
        sh = self._shape(S)
        self.sh_mu, self.sh_sd = sh.mean(0), sh.std(0) + 1e-3
        X = self._assemble(c, b, sh)
        self.mu = X.mean(0)
        # PCA is *only* used to draw the 2-D map; clustering runs in the full weighted space, because
        # rare events live in low-variance directions that a variance-truncated PCA would discard.
        self.map_pca = PCA(n_components=2, svd_solver="full", random_state=0).fit(X - self.mu)
        self.feature_names = ([f"count[T{i}]" for i in range(self.V)] + ["count[OOV]"]
                              + [f"trans[{a}>{b_}]" for a, b_ in self.bigram_names] + ["trans[unseen]"]
                              + ["n_events", "duration", "n_hosts", "max_gap", "n_oov_events"])
        return self

    def _assemble(self, c, b, sh):
        z = (sh - self.sh_mu) / self.sh_sd
        if self.one_sided_timing:
            z[:, [1, 3]] = np.maximum(z[:, [1, 3]], 0.0)         # duration, longest gap
        return np.hstack([np.log1p(c) * self.idf_c,
                          self.w_bigram * np.log1p(b) * self.idf_b,
                          self.w_shape * z]).astype(np.float32)

    def raw_parts(self, S: Sessions):
        """Unweighted human-unit quantities: (event counts, transition counts, shape block)."""
        return self._counts(S), self._bigram_counts(S), self._shape(S)

    def transform(self, S: Sessions):
        """Weighted feature matrix (n_sessions x n_features) used for clustering."""
        return self._assemble(self._counts(S), self._bigram_counts(S), self._shape(S))

    def project2d(self, X):
        return self.map_pca.transform(X - self.mu).astype(np.float32)
