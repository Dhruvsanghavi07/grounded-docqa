"""Command line: build an index, ask a question, or run the retrieval evaluation.

python -m docqa.cli ingest data/corpus --out data/index.pkl
python -m docqa.cli ask "How do I declare an optional query parameter?"
python -m docqa.cli eval eval/questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .chunking import load_corpus
from .index import HybridIndex
from .llm import make_generator
from .pipeline import QAPipeline


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_ingest(a):
    t0 = time.perf_counter()
    chunks = load_corpus(a.corpus, max_words=a.max_words, overlap=a.overlap)
    idx = HybridIndex(embedder=a.embedder).build(chunks)
    idx.save(a.out)
    docs = len({c.doc for c in chunks})
    print(f"indexed {len(chunks)} chunks from {docs} documents in {time.perf_counter() - t0:.1f}s -> {a.out}")


def cmd_ask(a):
    idx = HybridIndex.load(a.index)
    ans = QAPipeline(idx, make_generator(a.llm), k=a.k, mode=a.mode).ask(a.question)
    print(ans.answer)
    print()
    for s in ans.sources:
        print(f"[{s.n}] {s.doc} — {s.heading}")
    print(f"\ngrounded={ans.grounded} generator={ans.generator} retrieval={ans.retrieval_ms}ms")


def cmd_eval(a):
    from eval.run_eval import evaluate, evaluate_answers, print_table  # eval/ lives next to the package

    idx = HybridIndex.load(a.index)
    rows = read_jsonl(a.questions)
    results = {"retrieval": evaluate(idx, rows, ks=(1, 3, 5))}
    print_table(results["retrieval"], n=len(rows))
    if a.out_of_scope:
        ood = read_jsonl(a.out_of_scope)
        results["answers"] = evaluate_answers(QAPipeline(idx, make_generator(a.llm)), rows, ood)
        print(f"\nAnswer checks ({len(rows)} in-scope, {len(ood)} out-of-scope questions)")
        for k, v in results["answers"].items():
            print(f"  {k:<28}{v:.3f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2)


def main(argv=None):
    p = argparse.ArgumentParser(prog="docqa")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest", help="chunk a folder of markdown and build the index")
    i.add_argument("corpus")
    i.add_argument("--out", default="data/index.pkl")
    i.add_argument("--embedder", default="lsa", help="lsa, or st:<sentence-transformers model>")
    i.add_argument("--max-words", type=int, default=180)
    i.add_argument("--overlap", type=int, default=30)
    i.set_defaults(f=cmd_ingest)

    q = sub.add_parser("ask", help="answer one question")
    q.add_argument("question")
    q.add_argument("--index", default="data/index.pkl")
    q.add_argument("--llm", default=None, help="extractive | anthropic | openai")
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "dense"])
    q.set_defaults(f=cmd_ask)

    e = sub.add_parser("eval", help="retrieval and answer evaluation")
    e.add_argument("questions")
    e.add_argument("--index", default="data/index.pkl")
    e.add_argument("--out-of-scope", default=None)
    e.add_argument("--llm", default=None)
    e.add_argument("--json", default=None)
    e.set_defaults(f=cmd_eval)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    sys.exit(main())
