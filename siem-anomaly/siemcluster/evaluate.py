"""Held-out evaluation. Labels are used here and nowhere else."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def evaluate(y, s, thr=None, budgets=(0.01, 0.02, 0.05)):
    y, s = np.asarray(y), np.asarray(s)
    out = {"n": int(len(y)), "n_anomalies": int(y.sum()), "prevalence": float(y.mean()),
           "pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s))}
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / np.clip(p + r, 1e-12, None)
    j = int(np.nanargmax(f[:-1]))
    out["best_f1_oracle"] = float(f[j])
    out["best_f1_threshold"] = float(t[j])
    order = np.argsort(-s)
    for b in budgets:                                            # analyst alert-budget view
        k = max(1, int(round(b * len(y))))
        hit = int(y[order[:k]].sum())
        out[f"precision@{int(b*100)}%"] = hit / k
        out[f"recall@{int(b*100)}%"] = hit / max(1, y.sum())
    if thr is not None:
        pred = s >= thr
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        out.update({"threshold": float(thr), "tp": tp, "fp": fp, "fn": fn, "tn": int(len(y) - tp - fp - fn),
                    "precision": pr, "recall": rc, "f1": 2 * pr * rc / max(1e-12, pr + rc),
                    "alert_rate": float(pred.mean())})
    return out


def bootstrap_ci(y, s, thr, n_boot=1000, seed=0):
    """Percentile 95% CIs from resampling held-out sessions."""
    rng = np.random.RandomState(seed)
    y, s = np.asarray(y), np.asarray(s)
    keys = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        e = evaluate(y[i], s[i], thr)
        for k in keys:
            acc[k].append(e[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items()}
