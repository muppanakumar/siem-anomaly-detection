"""Stage 2 - online log template mining (a compact Drain-style parser).

Each message is (1) masked for obviously variable tokens, (2) routed through a fixed-depth
prefix tree keyed on token count and leading tokens, (3) merged into the most similar existing
template when similarity >= `sim`, wildcarding the tokens that differ.  Templates are mined on the
training window only and *frozen* afterwards: unseen message shapes at test time become OOV events,
which is itself a strong anomaly signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

WILD = "<*>"
_WORD = re.compile(r"[A-Za-z][A-Za-z.:]*")
_MASKS = [
    (re.compile(r"blk_-?\d+"), WILD),                       # HDFS block ids
    (re.compile(r"/?\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?"), WILD),  # IPv4[:port]
    (re.compile(r"0x[0-9a-fA-F]+"), WILD),                  # hex
    (re.compile(r"(?<![\w.])/[\w.\-/$]+"), WILD),            # absolute paths
    (re.compile(r"(?<![\w])-?\d+(?![\w])"), WILD),           # bare numbers
]


def mask(msg: str) -> str:
    for rx, rep in _MASKS:
        msg = rx.sub(rep, msg)
    return msg


def _tokens(msg: str) -> list[str]:
    return mask(msg).split()


@dataclass
class _Template:
    tid: int
    tokens: list[str]
    count: int = 1

    def text(self) -> str:
        return " ".join(self.tokens)


@dataclass
class Drain:
    sim: float = 0.5
    depth: int = 2          # number of leading tokens used for routing
    max_children: int = 100
    strict_words: bool = True   # set False for logs whose free-text params are plain words (user names, hosts)
    templates: list[_Template] = field(default_factory=list)
    _tree: dict = field(default_factory=dict)
    frozen: bool = False

    # -- routing -------------------------------------------------------------------------
    def _route(self, toks: list[str], create: bool):
        node = self._tree.setdefault(len(toks), {})
        for t in toks[: self.depth]:
            key = WILD if any(c.isdigit() for c in t) or t == WILD else t
            if key not in node:
                if not create:
                    key = WILD
                elif len(node) >= self.max_children:
                    key = WILD
                if key not in node:
                    if not create:
                        return None
                    node[key] = {}
            node = node[key]
        return node

    @staticmethod
    def _similarity(a: list[str], b: list[str]) -> float:
        same = sum(1 for x, y in zip(a, b) if x == y and x != WILD)
        return same / max(1, len(a))

    @staticmethod
    def _compatible(tpl_toks: list[str], toks: list[str]) -> bool:
        """True if `toks` fits the template: every non-wildcard template token must be equal."""
        return all(x == WILD or x == y for x, y in zip(tpl_toks, toks))

    def _mergeable(self, tpl_toks: list[str], toks: list[str]) -> bool:
        """Merge guard. Two *plain words* that differ (e.g. 'terminating' vs 'Interrupted.') mean a
        different event, not a different parameter, so they must never be wildcarded together."""
        if not self.strict_words:
            return True
        for x, y in zip(tpl_toks, toks):
            if x != y and x != WILD and _WORD.fullmatch(x) and _WORD.fullmatch(y):
                return False
        return True

    def _best(self, node, toks, learn: bool):
        best, best_s = None, -1.0
        for tpl in node.get("_t", []):
            s = self._similarity(tpl.tokens, toks)
            ok = self._mergeable(tpl.tokens, toks) if learn else self._compatible(tpl.tokens, toks)
            if ok and s > best_s:
                best, best_s = tpl, s
        return best, best_s

    # -- public API ----------------------------------------------------------------------
    def add(self, msg: str) -> int:
        """Learn from a message (training). Returns its template id."""
        toks = _tokens(msg)
        node = self._route(toks, create=True)
        tpl, s = self._best(node, toks, learn=True)
        if tpl is not None and s >= self.sim:
            tpl.tokens = [x if x == y else WILD for x, y in zip(tpl.tokens, toks)]
            tpl.count += 1
            return tpl.tid
        tpl = _Template(len(self.templates), toks)
        self.templates.append(tpl)
        node.setdefault("_t", []).append(tpl)
        return tpl.tid

    def match(self, msg: str) -> int:
        """Match without learning. Returns -1 for an unseen message shape (OOV)."""
        toks = _tokens(msg)
        node = self._route(toks, create=False)
        if node is None:
            return -1
        tpl, s = self._best(node, toks, learn=False)
        if tpl is not None and s >= self.sim:
            return tpl.tid
        return -1

    def fit(self, messages) -> "Drain":
        for m in messages:
            self.add(m)
        self.frozen = True
        return self

    def transform(self, messages) -> list[int]:
        return [self.match(m) for m in messages]

    def template_text(self, tid: int) -> str:
        return "<OOV: unseen message shape>" if tid < 0 else self.templates[tid].text()
