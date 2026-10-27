#!/usr/bin/env python3
"""Inject artifacts/workbench_data.json into workbench/template.html -> workbench/outlier_desk.html."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
data = (ROOT / "artifacts/workbench_data.json").read_text()
data = data.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")   # safe inside <script>
html = (ROOT / "workbench/template.html").read_text().replace("__DATA__", data)
out = ROOT / "workbench/outlier_desk.html"
out.write_text(html)
print("wrote", out, round(out.stat().st_size / 1e6, 2), "MB")
