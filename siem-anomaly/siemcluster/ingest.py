"""Stage 1 - ingest raw log lines and normalise them to a SIEM-style event schema.

Output schema (one row per event, loosely modelled on ECS / CEF fields):
    ts          datetime64   event time
    pid         int          emitting process id
    severity    str          INFO / WARN / ERROR ...
    source      str          emitting component (log source)
    message     str          free-text message body
The HDFS line format is:  ``YYMMDD HHMMSS PID LEVEL component: message``
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HDFS_LINE = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+(?P<severity>[A-Z]+)\s+"
    r"(?P<source>[\w.$]+):\s?(?P<message>.*)$"
)


def parse_lines(lines) -> pd.DataFrame:
    """Parse an iterable of raw log lines. Unparseable lines are counted, not silently dropped."""
    rows, bad = [], 0
    for ln in lines:
        m = HDFS_LINE.match(ln.rstrip("\n"))
        if not m:
            bad += 1
            continue
        rows.append(m.groupdict())
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df.pop("date") + df.pop("time"), format="%y%m%d%H%M%S")
    df["pid"] = df["pid"].astype(int)
    df.attrs["unparsed_lines"] = bad
    return df[["ts", "pid", "severity", "source", "message"]]


def read_log(path: str | Path) -> pd.DataFrame:
    with open(path, encoding="utf-8", errors="replace") as fh:
        df = parse_lines(fh)
    # Stable chronological order; ties keep file order (important for event sequences).
    df = df.reset_index(drop=True)
    df["seq"] = range(len(df))
    return df.sort_values(["ts", "seq"], kind="stable").reset_index(drop=True)
