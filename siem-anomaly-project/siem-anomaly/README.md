# Unsupervised clustering for SIEM log-anomaly detection

An unsupervised pipeline that turns a raw server-log stream into per-session event sequences, learns what
"normal" looks like by clustering the earlier traffic, and ranks later sessions by how far they sit from every
normal mode. **Outlier Desk** (`workbench/`) is the analyst-facing triage tool built on its output.

Data: the LogHub **HDFS** sample (Xu et al., SOSP'09), 104,815 log lines, 7,940 sessions (one per HDFS block),
with block-level labels used **only for evaluation**.

## Result (held-out test window: the last 25% of sessions by start time, 1,985 sessions, 105 anomalies)

| | PR-AUC | note |
|---|---|---|
| **KMeans + cluster-mass prior (this pipeline)** | **0.646** (95% CI 0.55–0.73) | prevalence is 0.053, so chance is 0.05 |
| GMM + cluster-mass prior | 0.649 | same features and prior |
| HDBSCAN k-NN + cluster-mass prior | 0.646 | same features and prior |
| PCA residual / SPE (Xu et al. 2009) | 0.514 | no clusters |
| Plain KMeans, silhouette-chosen k | 0.479 | no cluster-mass prior |
| Isolation Forest | 0.469 | |

At the label-free alert threshold (4.9% of sessions flagged): precision 0.66, recall 0.62, F1 0.64.
Reviewing the top 3% of the queue finds 56 of 60 anomalies (93% precision, 18x the base rate). A one-line
"message contains *exception*" rule gets precision 1.0 but recall 0.21.

**Detectability ceiling.** 40 of the 105 test anomalies have exactly the same event counts as routine
sessions. Count-based features cannot separate those, and the model flags none of them. Of the other 65, it
flags all 65. Telling the 40 apart would need features other than event counts; the post-hoc transition-weight
check in `metrics.json` suggests event order alone does not do it on this log.

## How it works (`siemcluster/`)

| Stage | Module | What it does |
|---|---|---|
| 1 Ingest | `ingest.py` | Parses raw lines to a SIEM-style schema (`ts, pid, severity, source, message`); counts unparseable lines instead of dropping them |
| 2 Templates | `drain.py` | Drain-style log parsing into event templates, **fitted on the training window only** |
| 3 Sessions | `sessions.py` | Groups events by entity (HDFS block id; swap `ENTITY_RX` for user/host/flow in other SIEM sources); forward-in-time split 60/15/25 |
| 4 Features | `features.py` | TF-IDF event counts + event-transition bigrams + lifecycle shape (length, duration, hosts, longest gap) |
| 5 Model | `models.py` | KMeans with k=20, deliberately over-clustered. **Cluster-mass prior:** only clusters holding at least 1% of training traffic count as normal modes, so a small cluster of failures is never trusted as normal. Score = distance to the nearest normal mode over its typical spread |
| 6 Threshold | `pipeline.py` | 99th percentile of normal-mode members' training scores. No labels involved |
| 7 Explain | `explain.py` | Per-session reasons in analyst units ("×7 unseen-shape events", "lifecycle took 523 s, typical 40 s") |
| 8 Evaluate | `evaluate.py` | PR/ROC-AUC, precision/recall at budgets, bootstrap CIs, baselines, ablation, sensitivity (validation window only) |

Leakage controls: templates, IDF weights, transition vocabulary, scalers, clusters and threshold are all fit on
train only; configuration choices use the validation slice; the test window is scored once. `tests/` asserts
that scores are bit-identical with and without labels, that the split runs forward in time, and that the
threshold is label-free.

## Run it

```bash
pip install numpy pandas scikit-learn scipy
python scripts/run_pipeline.py          # -> artifacts/{metrics.json, session_scores.csv, templates.csv, workbench_data.json}
python workbench/build_workbench.py     # -> workbench/outlier_desk.html (single self-contained page, ~3.3 MB)
python -m pytest -q                     # 8 tests, ~30 s
```

`run_pipeline.py [log] [labels]` takes any HDFS-format log. The full 575k-block HDFS dataset was not reachable
from the build environment, so results above are for the 100k-line sample. The same entry point accepts the
full set, but it was not run on it.

## Outlier Desk

* **Triage**: ranked queue with a "review top N" budget slider, per-session reasons, event-by-event lifecycle,
  behaviour map and score-over-time view, similar-session lookup, confirm / watch / benign verdicts with notes
  (keys `j` `k` `c` `w` `b`), CSV export. Labels stay hidden until you tick "Show benchmark labels".
* **Behaviour modes**: the normal modes, their share of training traffic, and the small clusters the model refuses to trust.
* **Hosts**: where flagged sessions concentrate.
* **Held-out evaluation**: everything above, measured on the test window only.

## Limits

* The labels are HDFS system faults, not attacker activity. The method and the workbench are source-agnostic,
  but the numbers say nothing about intrusion detection on other log types.
* 105 test anomalies is a small sample; use the confidence interval, not the point estimate.
* Transition features add almost nothing on this log (see the ablation); event counts do most of the work.
* `archive/abandoned_synthetic_attempt/` holds an earlier, unused synthetic-event generator.
