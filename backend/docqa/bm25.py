"""Okapi BM25, implemented from scratch with an inverted index.

score(q, d) = sum over query terms t of
    idf(t) * f(t, d) * (k1 + 1) / (f(t, d) + k1 * (1 - b + b * |d| / avgdl))
idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))      (the always-positive variant)
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

TOKEN = re.compile(r"[a-z0-9_]+")
STOP = frozenset(
    """a an and are as at be but by can do does for from how i if in is it its of on or
so that the their then there these this to was we what when where which who why will with you your""".split()
)


def stem(t: str) -> str:
    """Tiny plural stripper (errors -> error, queries -> query). Enough to match common inflections
    without pulling in a stemming library; heavier stemmers hurt identifiers like `status`."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def tokenize(text: str, stemmed: bool = False) -> list[str]:
    # BM25 indexes unstemmed tokens: on the 40-question eval, plural stemming lowered
    # BM25 recall@1 from 0.925 to 0.850, so it is only used for answer-sentence matching.
    toks = [t for t in TOKEN.findall(text.lower()) if t not in STOP]
    return [stem(t) for t in toks] if stemmed else toks


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(doc, tf)]
        self.lengths: list[int] = []
        self.avgdl = 0.0

    def fit(self, texts: list[str]) -> BM25:
        self.postings.clear()
        self.lengths = []
        for i, text in enumerate(texts):
            toks = tokenize(text)
            self.lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        return self

    def idf(self, term: str) -> float:
        n = len(self.postings.get(term, ()))
        return math.log(1 + (len(self.lengths) - n + 0.5) / (n + 0.5))

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf(term)
            for doc, tf in self.postings.get(term, ()):
                norm = self.k1 * (1 - self.b + self.b * self.lengths[doc] / self.avgdl)
                out[doc] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        s = self.scores(query)
        return sorted(s.items(), key=lambda x: -x[1])[:k]
