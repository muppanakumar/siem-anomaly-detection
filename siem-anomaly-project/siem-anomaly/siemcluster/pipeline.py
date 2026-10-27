"""End-to-end pipeline: raw log -> events -> templates -> sessions -> features -> clusters -> scores.

Leakage discipline
  * sessions are ordered by first-event time and split forward in time: train (60%) | validation (15%) | test (25%)
  * templates, IDF weights, the transition vocabulary, scalers, clusters and the alert threshold are all fitted on
    the TRAIN window only; validation/test sessions are only ever transformed
  * ground-truth labels are read for evaluation only (and for the workbench's optional "reveal" toggle)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .drain import Drain
from .evaluate import bootstrap_ci, evaluate
from .explain import Explainer
from .features import SessionFeaturizer
from .ingest import read_log
from .models import GMMScorer, HDBSCANKnnScorer, IForestScorer, KMeansScorer, PCAReconScorer
from .sessions import Sessions, build_sessions, explode_entities, session_order, temporal_split


@dataclass
class Config:
    k: int = 20                 # over-cluster on purpose; the cluster-mass prior decides which clusters are "normal"
    min_share: float = 0.01     # a cluster must hold >= 1% of training traffic to count as a normal mode
    envelope_q: float = 0.99    # alert threshold = this quantile of normal-mode members' training scores
    train: float = 0.60
    val: float = 0.15
    w_bigram: float = 0.6
    w_shape: float = 0.5
    min_bigram_df: int = 3
    seed: int = 0


class Timer(dict):
    def __call__(self, name):
        t = time.time()
        class _C:
            def __enter__(s): return s
            def __exit__(s, *a): self[name] = round(time.time() - t, 2)
        return _C()


def fit_score(cfg: Config, S: Sessions, drain: Drain, idx_train):
    ft = SessionFeaturizer(len(drain.templates), cfg.min_bigram_df, cfg.w_bigram, cfg.w_shape).fit(S.subset(idx_train))
    X = ft.transform(S)
    sc = KMeansScorer(cfg.k, cfg.min_share, cfg.seed).fit(X[idx_train])
    return ft, X, sc


def run(log_path, labels_path=None, cfg: Config | None = None, verbose=True):
    cfg = cfg or Config()
    T = Timer()
    say = print if verbose else (lambda *a, **k: None)

    with T("ingest"):
        events = read_log(log_path)
    labels = None
    if labels_path:
        lab = pd.read_csv(labels_path)
        labels = dict(zip(lab.BlockId, lab.Label))
    with T("sessionize"):
        ev = explode_entities(events)
        order = session_order(ev)
    tr, va, te = temporal_split(len(order), cfg.train, cfg.val)
    train_ids = set(order[i] for i in tr)
    with T("template_mining"):
        drain = Drain().fit(ev[ev.entity.isin(train_ids)].message)   # TRAIN window only
        uniq = ev.message.unique()
        lut = dict(zip(uniq, drain.transform(uniq)))
        tids = ev.message.map(lut).to_numpy()
    S = build_sessions(ev, tids, labels, order)
    say(f"[ingest] {len(events):,} log lines -> {len(S):,} sessions | split train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    say(f"[templates] {len(drain.templates)} mined on train window; OOV events later: "
        f"{int(sum((t < 0).sum() for t in S.tids)):,}")

    with T("features_and_clustering"):
        ft, X, sc = fit_score(cfg, S, drain, tr)
    score = sc.score(X)
    near_any, _ = sc.assign(X)
    near_norm, d_norm = sc.assign_normal(X)

    # label-free alert threshold: outer envelope of the normal modes' own training members
    in_mode = sc.normal_[sc.labels_]
    thr = float(np.quantile(score[tr][in_mode], cfg.envelope_q))
    flagged = score >= thr
    say(f"[model] k={cfg.k} -> {int(sc.normal_.sum())} normal modes ({sc.share_[sc.normal_].sum():.1%} of train traffic), "
        f"{int((~sc.normal_).sum())} small clusters distrusted; threshold={thr:.3f}")

    res = dict(cfg=cfg, S=S, events=events, drain=drain, ft=ft, X=X, sc=sc, score=score, near_any=near_any,
               near_norm=near_norm, flagged=flagged, thr=thr, split=dict(train=tr, val=va, test=te), timings=T)

    if S.labels is not None:
        y = S.labels
        with T("evaluation"):
            res["metrics"] = evaluate_all(cfg, S, X, tr, va, te, score, thr, drain, ft, sc)
        m = res["metrics"]["test"]["primary"]
        say(f"[test] PR-AUC {m['pr_auc']:.3f} | ROC-AUC {m['roc_auc']:.3f} | at threshold: P {m['precision']:.3f} "
            f"R {m['recall']:.3f} F1 {m['f1']:.3f} (alert rate {m['alert_rate']:.1%})")

    with T("explanations"):
        res["explainer"] = Explainer(ft, sc, S.subset(tr), X[tr], lambda t: short(drain.template_text(t)))
    return res


def short(t: str, n=64):
    t = t.replace("<*>", "…").replace("……", "…")
    return t if len(t) <= n else t[: n - 1] + "…"


def evaluate_all(cfg, S, X, tr, va, te, score, thr, drain, ft, sc):
    y = S.labels
    out = {"split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
           "anomalies": {"train": int(y[tr].sum()), "val": int(y[va].sum()), "test": int(y[te].sum())}}
    # ---- primary model on VALIDATION and TEST -------------------------------------------------------------
    out["val"] = {"primary": evaluate(y[va], score[va], thr)}
    out["test"] = {"primary": evaluate(y[te], score[te], thr)}
    out["test"]["primary_ci95"] = bootstrap_ci(y[te], score[te], thr)
    # ---- comparison: same features, same train window, no labels ------------------------------------------
    comp = {}
    for name, mk in [("KMeans + cluster-mass prior (primary)", None),
                     ("GMM + cluster-mass prior", lambda: GMMScorer(cfg.k, cfg.min_share, cfg.seed)),
                     ("HDBSCAN k-NN + cluster-mass prior", lambda: HDBSCANKnnScorer(min_share=cfg.min_share, seed=cfg.seed)),
                     ("Isolation Forest (baseline)", lambda: IForestScorer(cfg.seed)),
                     ("PCA residual / SPE (Xu et al. baseline)", lambda: PCAReconScorer(0.8)),
                     ("Plain KMeans, silhouette-k, no mass prior", None)]:
        if name.startswith("KMeans + cluster"):
            s_te = score[te]
        elif name.startswith("Plain"):
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            rng = np.random.RandomState(cfg.seed); sub = rng.choice(len(tr), min(len(tr), 3000), replace=False)
            best = max((silhouette_score(X[tr][sub], (km := KMeans(k, n_init=5, random_state=0).fit(X[tr])).labels_[sub]), k) for k in range(2, 13))
            km = KMeans(best[1], n_init=5, random_state=0).fit(X[tr])
            s_te = km.transform(X[te]).min(1)
        else:
            s_te = mk().fit(X[tr]).score(X[te])
        e = evaluate(y[te], s_te)
        comp[name] = {k: e[k] for k in ("pr_auc", "roc_auc", "best_f1_oracle", "precision@5%", "recall@5%")}
    kw = np.array([any("xception" in m for m in S.msgs[i]) for i in te])
    tp = int((kw & (y[te] == 1)).sum())
    comp["Analyst grep rule: any message contains 'exception'"] = {
        "precision": tp / max(1, kw.sum()), "recall": tp / max(1, y[te].sum()),
        "f1": 2 * tp / max(1, kw.sum() + y[te].sum())}
    out["test"]["comparison"] = comp
    # ---- post-hoc diagnostics (looked at AFTER seeing test errors: descriptive, not used to pick the model) -----
    post = {}
    for wb in (0.6, 0.3, 0.0):
        f2 = SessionFeaturizer(len(drain.templates), cfg.min_bigram_df, wb, cfg.w_shape).fit(S.subset(tr))
        A, B = f2.transform(S.subset(tr)), f2.transform(S.subset(te))
        m2 = KMeansScorer(cfg.k, cfg.min_share, cfg.seed).fit(A)
        t2 = float(np.quantile(m2.score(A)[m2.normal_[m2.labels_]], cfg.envelope_q))
        e = evaluate(y[te], m2.score(B), t2)
        post[f"w_transition={wb}"] = {"pr_auc": e["pr_auc"], "roc_auc": e["roc_auc"], "alert_rate_at_envelope_threshold": e["alert_rate"],
                                      "precision": e["precision"], "recall": e["recall"]}
    out["test"]["posthoc_transition_weight"] = post
    # ---- sensitivity & ablation: VALIDATION window only (never tuned on test) ---------------------------------
    grid = {}
    for k in (10, 20, 30, 50):
        for ms in (0.005, 0.01, 0.02, 0.05):
            m = KMeansScorer(k, ms, cfg.seed).fit(X[tr]); grid[f"k={k}|tau={ms}"] = evaluate(y[va], m.score(X[va]))["pr_auc"]
    out["val"]["sensitivity_pr_auc"] = grid
    abl = {}
    for label, wb, ws in [("event counts only", 0.0, 0.0), ("+ transitions", cfg.w_bigram, 0.0),
                          ("+ shape/timing", 0.0, cfg.w_shape), ("all blocks (default)", cfg.w_bigram, cfg.w_shape)]:
        f2 = SessionFeaturizer(len(drain.templates), cfg.min_bigram_df, wb, ws).fit(S.subset(tr))
        A, B = f2.transform(S.subset(tr)), f2.transform(S.subset(va))
        e = evaluate(y[va], KMeansScorer(cfg.k, cfg.min_share, cfg.seed).fit(A).score(B))
        abl[label] = {"pr_auc": e["pr_auc"], "roc_auc": e["roc_auc"]}
    out["val"]["ablation"] = abl
    # ---- detectability ceiling: anomalies whose event multiset is identical to routine traffic -----------------
    import collections
    sig = lambda t: tuple(sorted(collections.Counter(t.tolist()).items()))
    sigs = [sig(t) for t in S.tids]
    common = {k for k, v in collections.Counter(sigs[i] for i in tr if y[i] == 0).items() if v >= 50}
    out["ceiling"] = {}
    for nm, idx in (("train", tr), ("val", va), ("test", te)):
        a = [i for i in idx if y[i] == 1]
        out["ceiling"][nm] = {"anomalies": len(a), "same_event_counts_as_routine": int(sum(sigs[i] in common for i in a))}
    fl = score >= thr
    a_te = [i for i in te if y[i] == 1]
    out["ceiling"]["test"].update({
        "caught_with_same_counts": int(sum(fl[i] and sigs[i] in common for i in a_te)),
        "missed_with_same_counts": int(sum((not fl[i]) and sigs[i] in common for i in a_te)),
        "caught_with_different_counts": int(sum(fl[i] and sigs[i] not in common for i in a_te)),
        "missed_with_different_counts": int(sum((not fl[i]) and sigs[i] not in common for i in a_te))})
    return out
