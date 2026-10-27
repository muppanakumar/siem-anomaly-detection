#!/usr/bin/env python3
"""Run the full pipeline and write artifacts/.   usage: python scripts/run_pipeline.py [log] [labels]"""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from siemcluster.export import write_artifacts
from siemcluster.pipeline import run

log = sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/raw/HDFS_100k.log"
lab = sys.argv[2] if len(sys.argv) > 2 else ROOT / "data/raw/anomaly_label.csv"
res = run(log, lab)
data = write_artifacts(res, ROOT / "artifacts")
print("timings (s):", dict(res["timings"]))
print("wrote", ROOT / "artifacts", "| workbench_data.json:", round((ROOT / "artifacts/workbench_data.json").stat().st_size / 1e6, 2), "MB")
