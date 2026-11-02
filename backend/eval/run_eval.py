"""Retrieval evaluation: Recall@k and MRR for BM25, dense and hybrid retrieval.

A question counts as a hit at k if any of its top-k chunks comes from one of the
documents labeled relevant for it. MRR is the mean of 1/rank of the first
relevant chunk (0 if none appears in the top 10).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODES = ("bm25", "dense", "hybrid")


def evaluate(index, rows: list[dict], ks=(1, 3, 5), depth: int = 10) -> dict:
    out = {}
    for mode in MODES:
        hits = {k: 0 for k in ks}
        rr = 0.0
        for row in rows:
            rel = set(row["relevant"])
            docs = [h.chunk.doc for h in index.search(row["q"], k=depth, mode=mode)]
            first = next((r for r, d in enumerate(docs, 1) if d in rel), None)
            rr += 1.0 / first if first else 0.0
            for k in ks:
                hits[k] += first is not None and first <= k
        n = max(1, len(rows))
        out[mode] = {**{f"recall@{k}": round(hits[k] / n, 3) for k in ks}, "mrr": round(rr / n, 3)}
    return out


def evaluate_answers(pipeline, in_scope: list[dict], out_of_scope: list[dict]) -> dict:
    """End-to-end checks on the answers themselves (not just retrieval):
    - answered_from_relevant_doc: in-scope questions answered with a citation to a labeled document
    - grounded_rate: answers that pass the citation checker
    - refusal_rate_out_of_scope: off-topic questions correctly refused
    - false_refusal_rate: in-scope questions wrongly refused"""
    refusal = "does not answer"
    good = grounded = false_ref = 0
    for row in in_scope:
        a = pipeline.ask(row["q"])
        grounded += a.grounded
        if refusal in a.answer:
            false_ref += 1
            continue
        cited_docs = {s.doc for s in a.sources if f"[{s.n}]" in a.answer}
        good += bool(cited_docs & set(row["relevant"]))
    refused = sum(refusal in pipeline.ask(r["q"]).answer for r in out_of_scope)
    n, m = max(1, len(in_scope)), max(1, len(out_of_scope))
    return {
        "answered_from_relevant_doc": round(good / n, 3),
        "grounded_rate": round(grounded / n, 3),
        "false_refusal_rate": round(false_ref / n, 3),
        "refusal_rate_out_of_scope": round(refused / m, 3),
    }


def print_table(results: dict, n: int) -> None:
    cols = list(next(iter(results.values())).keys())
    print(f"Retrieval quality on {n} labeled questions")
    print(f"{'mode':<8}" + "".join(f"{c:>11}" for c in cols))
    for mode, r in results.items():
        print(f"{mode:<8}" + "".join(f"{r[c]:>11.3f}" for c in cols))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from docqa.index import HybridIndex

    idx = HybridIndex.load(sys.argv[2] if len(sys.argv) > 2 else "data/index.pkl")
    with open(sys.argv[1]) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    res = evaluate(idx, rows)
    print_table(res, len(rows))
