"""Stage 7 - export model outputs for the analyst workbench and for offline review."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .features import BOS, EOS
from .pipeline import short

BLK = re.compile(r"blk_-?\d+")


def _signature(med_counts, tmpl_short, top=4):
    order = np.argsort(-med_counts)
    parts = [f"{tmpl_short(j)} ×{med_counts[j]:.0f}" for j in order[:top] if med_counts[j] >= 0.5]
    return parts


def build_workbench_data(res: dict, source_name="HDFS (LogHub / loglizer sample, 100k lines)") -> dict:
    S, X, sc, ft, drain, cfg = res["S"], res["X"], res["sc"], res["ft"], res["drain"], res["cfg"]
    score, thr, flagged = res["score"], res["thr"], res["flagged"]
    tr, va, te = res["split"]["train"], res["split"]["val"], res["split"]["test"]
    ex = res["explainer"]
    tshort = lambda t: short(drain.template_text(t))
    n = len(S)
    coords = ft.project2d(X)
    lo, hi = np.percentile(coords, 0.5, axis=0), np.percentile(coords, 99.5, axis=0)
    coords = np.clip((coords - lo) / (hi - lo + 1e-9), -0.05, 1.05)

    # ---- hosts ----------------------------------------------------------------------------------------
    host_ix = {}
    for i in te:
        for h in S.hosts[i]:
            if h and h not in host_ix:
                host_ix[h] = len(host_ix)

    # ---- templates --------------------------------------------------------------------------------------
    all_t = np.concatenate([t for t in S.tids]) if n else np.array([])
    templates = [{"id": t.tid, "text": drain.template_text(t.tid), "short": tshort(t.tid),
                  "n_train": int(t.count), "n_all": int((all_t == t.tid).sum())} for t in drain.templates]
    templates.append({"id": -1, "text": "<unseen message shape>", "short": "UNSEEN EVENT SHAPE",
                      "n_train": 0, "n_all": int((all_t == -1).sum())})

    # ---- clusters ---------------------------------------------------------------------------------------
    c_all = res["near_any"]
    craw, _, shraw = ft.raw_parts(S.subset(tr))
    clusters = []
    for j in range(sc.k):
        mem = np.where(sc.labels_ == j)[0]
        med = np.median(craw[mem], 0) if len(mem) else np.zeros(craw.shape[1])
        clusters.append({"id": j, "share": float(sc.share_[j]), "normal": bool(sc.normal_[j]), "n_train": int(len(mem)),
                         "n_hunt": int((c_all[te] == j).sum()), "mean_score_hunt": float(score[te][c_all[te] == j].mean()) if (c_all[te] == j).any() else None,
                         "signature": _signature(med[:-1], tshort),
                         "typ": {str(t): int(round(v)) for t, v in enumerate(med[:-1]) if v >= 0.5}, "typ_events": float(np.expm1(np.median(shraw[mem, 0])) if len(mem) else 0),
                         "typ_duration": float(np.expm1(np.median(shraw[mem, 1])) if len(mem) else 0)})

    # ---- per-session data -------------------------------------------------------------------------------
    c, b, sh = ft.raw_parts(S.subset(te))
    te_pos = {int(i): j for j, i in enumerate(te)}
    Xte = X[te]
    # nearest neighbours (feature space) among hunt-window sessions, for "apply verdict to similar sessions"
    from sklearn.neighbors import NearestNeighbors
    nnm = NearestNeighbors(n_neighbors=min(41, len(te))).fit(Xte)
    interesting = np.where(score[te] >= 1.0)[0]
    dist, ind = nnm.kneighbors(Xte[interesting])
    nn = {}
    for r_, j in enumerate(interesting):
        keep = [int(q) for q, d in zip(ind[r_], dist[r_]) if q != j and d <= 0.6 * sc.scale_]
        if keep:
            nn[int(j)] = keep
    hunt = []
    for j, i in enumerate(te):
        rs = ex.reasons(X[i], c[j], b[j], sh[j], int(res["near_norm"][i])) if score[i] >= 0.5 else []
        hunt.append({
            "id": str(S.ids[i]), "t": int(S.start[i].timestamp()), "s": round(float(score[i]), 3),
            "c": int(c_all[i]), "nm": int(res["near_norm"][i]), "n": int(len(S.tids[i])),
            "d": int(S.dts[i][-1]) if len(S.dts[i]) else 0,
            "e": S.tids[i].tolist(), "dt": [int(x) for x in S.dts[i]],
            "h": [host_ix.get(x, -1) if x else -1 for x in S.hosts[i]],
            "m": [BLK.sub("~", m) for m in S.msgs[i]][:60],
            "lv": [("W" if v != "INFO" else "") for v in S.sev[i]][:60],
            "why": [[k, t, round(sh_, 2)] for k, t, sh_ in rs],
            "y": int(S.labels[i]) if S.labels is not None else None,
        })

    meta = {
        "dataset": source_name, "n_lines": int(len(res["events"])), "n_sessions": int(n),
        "t0": int(S.start.min().timestamp()), "t1": int(S.start.max().timestamp()),
        "split": {"train": [int(S.start[tr[0]].timestamp()), int(S.start[tr[-1]].timestamp()), len(tr)],
                  "val": [int(S.start[va[0]].timestamp()), int(S.start[va[-1]].timestamp()), len(va)],
                  "test": [int(S.start[te[0]].timestamp()), int(S.start[te[-1]].timestamp()), len(te)]},
        "threshold": float(thr), "k": int(sc.k), "n_normal_modes": int(sc.normal_.sum()), "min_share": cfg.min_share,
        "envelope_q": cfg.envelope_q, "n_templates": len(drain.templates), "n_features": int(X.shape[1]),
        "timings": dict(res["timings"]),
        "dur_p99": float(np.expm1(ex.sh_p99[1])), "scale": float(sc.scale_),
    }
    data = {"meta": meta, "templates": templates, "clusters": clusters, "hosts": list(host_ix),
            "map": {"x": np.round(coords[:, 0], 3).tolist(), "y": np.round(coords[:, 1], 3).tolist(),
                    "s": np.round(score, 2).tolist(), "c": c_all.tolist(),
                    "sp": ([0] * len(tr)) + ([1] * len(va)) + ([2] * len(te)),
                    "t": [int(x.timestamp()) for x in S.start]},
            "hunt_offset": int(te[0]), "hunt": hunt, "nn": nn}
    if "metrics" in res:
        m = res["metrics"]
        y, s_te = S.labels[te], score[te]
        p, r, t_ = precision_recall_curve(y, s_te)
        pick = np.unique(np.linspace(0, len(p) - 1, 80).astype(int))
        order = np.argsort(-s_te)
        cum = np.cumsum(y[order]) / max(1, y.sum())
        gains = [[round((k + 1) / len(y), 4), round(float(cum[k]), 4)] for k in range(len(y))]
        budgets = []
        for bud in (0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15):
            k = max(1, int(round(bud * len(y)))); hit = int(y[order[:k]].sum())
            budgets.append({"budget": bud, "reviewed": k, "found": hit, "precision": hit / k, "recall": hit / max(1, y.sum()),
                            "lift": (hit / k) / max(1e-9, y.mean())})
        # reason mix of flagged sessions by ground truth (used in the written analysis)
        mix = {"TP": {}, "FP": {}}
        for j, i in enumerate(te):
            if flagged[i]:
                bucket = mix["TP" if S.labels[i] == 1 else "FP"]
                for k_, _, _ in hunt[j]["why"]:
                    bucket[k_] = bucket.get(k_, 0) + 1
        data["eval"] = {"metrics": m, "pr_curve": [[float(r[i]), float(p[i])] for i in pick if i < len(p)], "gains": gains,
                        "budgets": budgets, "prevalence": float(y.mean()), "reason_mix": mix}
    return data


def write_artifacts(res: dict, out_dir: str | Path):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    S, score = res["S"], res["score"]
    split = np.empty(len(S), dtype=object)
    for k, v in res["split"].items():
        split[v] = k
    df = pd.DataFrame({"session_id": S.ids, "split": split, "start": S.start, "n_events": [len(t) for t in S.tids],
                       "duration_s": [float(d[-1]) if len(d) else 0 for d in S.dts], "anomaly_score": np.round(score, 4),
                       "nearest_cluster": res["near_any"], "nearest_normal_mode": res["near_norm"], "flagged": res["flagged"]})
    if S.labels is not None:
        df["ground_truth"] = np.where(S.labels == 1, "Anomaly", "Normal")
    df.to_csv(out / "session_scores.csv", index=False)
    pd.DataFrame([{"template_id": t.tid, "template": t.text(), "train_count": t.count} for t in res["drain"].templates]
                 ).to_csv(out / "templates.csv", index=False)
    data = build_workbench_data(res)
    (out / "workbench_data.json").write_text(json.dumps(data, separators=(",", ":")))
    if "metrics" in res:
        (out / "metrics.json").write_text(json.dumps({"config": res["cfg"].__dict__, "threshold": res["thr"],
                                                       "timings_s": dict(res["timings"]), **res["metrics"]}, indent=2))
    return data
