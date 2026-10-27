"""Guards for the properties the write-up relies on: no leakage, forward-in-time split, label-free threshold,
reproducibility, and that the held-out numbers shown in the workbench are the numbers the pipeline produces.

Run:  python -m pytest -q          (needs data/raw/HDFS_100k.log and data/raw/anomaly_label.csv; ~45 s)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from siemcluster.ingest import parse_lines  # noqa: E402
from siemcluster.pipeline import run  # noqa: E402

LOG, LAB = ROOT / "data/raw/HDFS_100k.log", ROOT / "data/raw/anomaly_label.csv"
pytestmark = pytest.mark.skipif(not (LOG.exists() and LAB.exists()), reason="HDFS sample not downloaded")


@pytest.fixture(scope="module")
def blind():
    return run(LOG, None, verbose=False)            # no labels at all


@pytest.fixture(scope="module")
def labelled():
    return run(LOG, LAB, verbose=False)


def test_unparseable_lines_are_counted_not_dropped_silently():
    df = parse_lines(["081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_1 src: /1.2.3.4:5 dest: /1.2.3.4:6",
                      "not a log line"])
    assert len(df) == 1 and df.attrs["unparsed_lines"] == 1


def test_split_runs_forward_in_time(labelled):
    S, sp = labelled["S"], labelled["split"]
    tr, va, te = sp["train"], sp["val"], sp["test"]
    assert S.start[tr].max() <= S.start[va].min()
    assert S.start[va].max() <= S.start[te].min()
    assert len(tr) + len(va) + len(te) == len(S)


def test_templates_come_from_the_training_window_only(labelled):
    S, sp = labelled["S"], labelled["split"]
    oov = lambda idx: int(sum((S.tids[i] < 0).sum() for i in idx))
    assert oov(sp["train"]) == 0                       # everything seen in training is known
    assert oov(sp["val"]) + oov(sp["test"]) > 0        # later windows contain shapes training never saw


def test_labels_never_influence_the_model(blind, labelled):
    assert np.allclose(blind["score"], labelled["score"])
    assert blind["thr"] == labelled["thr"]
    assert (blind["flagged"] == labelled["flagged"]).all()


def test_threshold_is_the_label_free_envelope_of_normal_modes(labelled):
    sc, cfg, tr = labelled["sc"], labelled["cfg"], labelled["split"]["train"]
    in_mode = sc.normal_[sc.labels_]
    assert labelled["thr"] == pytest.approx(float(np.quantile(labelled["score"][tr][in_mode], cfg.envelope_q)))


def test_run_is_reproducible(blind):
    again = run(LOG, None, verbose=False)
    assert np.array_equal(again["score"], blind["score"])


def test_heldout_results_beat_baselines_and_chance(labelled):
    m = labelled["metrics"]["test"]
    p, cmp = m["primary"], m["comparison"]
    assert p["pr_auc"] > 5 * p["prevalence"]
    assert p["pr_auc"] > cmp["Isolation Forest (baseline)"]["pr_auc"]
    assert p["pr_auc"] > cmp["PCA residual / SPE (Xu et al. baseline)"]["pr_auc"]
    lo, hi = m["primary_ci95"]["pr_auc"]
    assert lo <= p["pr_auc"] <= hi


def test_workbench_data_matches_the_pipeline(labelled):
    art = ROOT / "artifacts"
    if not (art / "workbench_data.json").exists():
        pytest.skip("run scripts/run_pipeline.py first")
    d = json.loads((art / "workbench_data.json").read_text())
    saved = json.loads((art / "metrics.json").read_text())
    assert d["meta"]["n_sessions"] == len(labelled["S"])
    assert saved["test"]["primary"]["pr_auc"] == pytest.approx(labelled["metrics"]["test"]["primary"]["pr_auc"])
    assert len(d["hunt"]) == len(labelled["split"]["test"])
