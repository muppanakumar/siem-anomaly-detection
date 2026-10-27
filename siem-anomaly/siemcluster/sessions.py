"""Stage 3 - turn the event stream into per-entity event *sequences* (sessions).

For HDFS the entity is the block id: every log line that mentions a block belongs to that block's
lifecycle. The same idea carries to other SIEM sources by swapping `ENTITY_RX` (user, host, src_ip,
process tree, flow id ...).  Labels, when available, are attached for *evaluation only*.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

ENTITY_RX = re.compile(r"blk_-?\d+")
IP_RX = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3})")


@dataclass
class Sessions:
    ids: np.ndarray                 # entity ids (object)
    start: pd.DatetimeIndex
    end: pd.DatetimeIndex
    tids: list[np.ndarray]          # template ids per event, chronological
    dts: list[np.ndarray]           # seconds since first event of the session
    hosts: list[list[str]]          # first IPv4 mentioned per event ('' if none)
    msgs: list[list[str]]           # raw message text per event (for analyst drill-down)
    sev: list[list[str]]
    src: list[list[str]]
    labels: np.ndarray | None = None  # 1 = anomaly, 0 = normal, from ground truth (evaluation only)

    def __len__(self):
        return len(self.ids)

    def subset(self, idx) -> "Sessions":
        idx = np.asarray(idx)
        pick = lambda xs: [xs[i] for i in idx]
        return Sessions(self.ids[idx], self.start[idx], self.end[idx], pick(self.tids), pick(self.dts),
                        pick(self.hosts), pick(self.msgs), pick(self.sev), pick(self.src),
                        None if self.labels is None else self.labels[idx])


def explode_entities(events: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, entity). A line that touches several entities is shared by all of them."""
    ev = events.copy()
    ev["entity"] = ev.message.map(lambda m: sorted(set(ENTITY_RX.findall(m))))
    ev = ev[ev.entity.str.len() > 0].explode("entity")
    ev["host"] = ev.message.map(lambda m: (IP_RX.findall(m) or [""])[0])
    return ev.sort_values(["entity", "ts", "seq"], kind="stable").reset_index(drop=True)


def session_order(ev: pd.DataFrame) -> list:
    """Entity ids ordered by first-event time (ties: file order) - the basis of the forward-in-time split."""
    first = ev.groupby("entity").agg(ts=("ts", "min"), seq=("seq", "min"))
    return first.sort_values(["ts", "seq"], kind="stable").index.tolist()


def build_sessions(ev: pd.DataFrame, tids: np.ndarray, labels: dict | None = None,
                   order: list | None = None) -> Sessions:
    """ev: output of explode_entities. tids: template id per row of ev."""
    ev = ev.assign(tid=tids)
    groups = {k: g for k, g in ev.groupby("entity", sort=False)}
    order = order or session_order(ev)
    ids, start, end, T, D, H, M, S, R = [], [], [], [], [], [], [], [], []
    for ent in order:
        g = groups[ent]
        t0 = g.ts.iloc[0]
        ids.append(ent); start.append(t0); end.append(g.ts.iloc[-1])
        T.append(g.tid.to_numpy(np.int32))
        D.append((g.ts - t0).dt.total_seconds().to_numpy(np.float32))
        H.append(g.host.tolist()); M.append(g.message.tolist())
        S.append(g.severity.tolist()); R.append(g.source.tolist())
    lab = None
    if labels is not None:
        lab = np.array([1 if labels.get(i) == "Anomaly" else 0 for i in ids], dtype=np.int8)
    return Sessions(np.array(ids, dtype=object), pd.DatetimeIndex(start), pd.DatetimeIndex(end),
                    T, D, H, M, S, R, lab)


def temporal_split(n: int, train=0.60, val=0.15):
    """Sessions are already start-time ordered, so contiguous slices are a strict forward-in-time split."""
    a, b = int(n * train), int(n * (train + val))
    idx = np.arange(n)
    return idx[:a], idx[a:b], idx[b:]
