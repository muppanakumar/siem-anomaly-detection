"""Stage 6 - "why was this flagged?" explanations an analyst can read.

For a session x and its nearest *normal mode* c, each feature's share of the squared deviation
||x - c||^2 ranks the reasons.  Reasons are then phrased in human units (counts, seconds) against what is
typical for that mode (the median over the mode's training members).
"""
from __future__ import annotations

import numpy as np

from .features import BOS, EOS


def _name(v, tmpl):
    return {BOS: "session start", EOS: "session end"}.get(v) or (f"OOV event" if v < 0 else f"T{v}")


class Explainer:
    def __init__(self, ft, scorer, S_train, X_train, tmpl_text):
        self.ft, self.sc, self.tmpl = ft, scorer, tmpl_text
        c, b, sh = ft.raw_parts(S_train)
        lab = scorer.km.predict(X_train)
        k = scorer.k
        med = lambda A: np.stack([np.median(A[lab == j], 0) if (lab == j).any() else np.zeros(A.shape[1]) for j in range(k)])
        self.med_c, self.med_b, self.med_sh = med(c), med(b), med(sh)
        self.sh_p99 = np.percentile(sh, 99, axis=0)
        self.nc, self.nb = c.shape[1], b.shape[1]

    def reasons(self, x, raw_c, raw_b, raw_sh, cluster, top=3):
        """x: weighted feature row; raw_*: the same session's unweighted parts; cluster: nearest NORMAL mode."""
        cen = self.sc.km.cluster_centers_[cluster]
        dev = (x - cen) ** 2
        tot = dev.sum() + 1e-12
        out = []
        for j in np.argsort(-dev)[: top + 2]:
            share = float(dev[j] / tot)
            if share < 0.05 and len(out) >= 1:
                break
            out.append(self._phrase(j, raw_c, raw_b, raw_sh, cluster, share))
            if len(out) == top:
                break
        return out

    def _phrase(self, j, rc, rb, rsh, cl, share):
        nc, nb, ft = self.nc, self.nb, self.ft
        if j < nc:                                              # event counts
            n, typ = int(rc[j]), self.med_c[cl, j]
            if j == nc - 1:
                return ["oov", f"{n} event(s) with a message shape never seen in training", share]
            t = self.tmpl(j)
            return ["count", f"“{t}” ×{n} (typical for this behaviour: ×{typ:.0f})", share]
        j2 = j - nc
        if j2 < nb:                                             # transitions
            n, typ = int(rb[j2]), self.med_b[cl, j2]
            if j2 == nb - 1:
                return ["transition", f"{n} event-to-event transition(s) never observed in training", share]
            a, b = ft.bigram_names[j2]
            nm = lambda v: _name(v, None) if v < 0 and v in (BOS, EOS) else f"“{self.tmpl(v)}”"
            return ["transition", f"{nm(a)} → {nm(b)} ×{n} (typical: ×{typ:.0f})", share]
        k = j2 - nb                                             # shape
        v, typ, p99 = rsh[k], self.med_sh[cl, k], self.sh_p99[k]
        if k == 0:
            return ["shape", f"{np.expm1(v):.0f} events in the lifecycle (typical: {np.expm1(typ):.0f})", share]
        if k == 1:
            return ["shape", f"lifecycle took {np.expm1(v):.0f}s (typical: {np.expm1(typ):.0f}s; 99th pct of training: {np.expm1(p99):.0f}s)", share]
        if k == 2:
            return ["shape", f"{v:.0f} distinct hosts involved (typical: {typ:.0f})", share]
        if k == 3:
            return ["shape", f"longest silent gap {np.expm1(v):.0f}s (typical: {np.expm1(typ):.0f}s)", share]
        return ["oov", f"{v:.0f} unseen-shape event(s)", share]
