"""Retrieve -> generate -> verify -> (retry | fall back)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .grounding import check
from .index import HybridIndex
from .llm import ExtractiveGenerator, Generator

RETRY_NOTE = (
    "\n\nYour previous answer broke the citation rules: {problems}. "
    "Answer again, citing a passage for every sentence and copying numbers exactly."
)


@dataclass
class Source:
    n: int
    doc: str
    heading: str
    text: str
    score: float


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source]
    grounded: bool
    generator: str
    attempts: int
    fell_back: bool
    violations: list[str] = field(default_factory=list)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class QAPipeline:
    def __init__(
        self,
        index: HybridIndex,
        generator: Generator | None = None,
        k: int = 5,
        mode: str = "hybrid",
        max_attempts: int = 2,
    ):
        self.index, self.k, self.mode, self.max_attempts = index, k, mode, max_attempts
        self.generator = generator or ExtractiveGenerator()
        self.fallback = ExtractiveGenerator()

    def ask(self, question: str, k: int | None = None, mode: str | None = None) -> Answer:
        """k and mode override the defaults for this call only, so one shared pipeline
        can serve concurrent API requests with different settings."""
        question = question.strip()
        if not question:
            raise ValueError("question is empty")
        t0 = time.perf_counter()
        hits = self.index.search(question, k=k or self.k, mode=mode or self.mode)
        t1 = time.perf_counter()
        passages = [f"{h.chunk.heading}\n{h.chunk.text}" for h in hits]
        sources = [
            Source(i, h.chunk.doc, h.chunk.heading, h.chunk.text, round(h.score, 5)) for i, h in enumerate(hits, 1)
        ]

        prompt_q, text, report, attempts = question, "", None, 0
        violations: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            attempts = attempt
            try:
                text = self.generator.generate(prompt_q, passages)
            except Exception as e:  # network/model errors fall through to the fallback
                report = None
                text = ""
                violations = [f"generator error: {type(e).__name__}: {e}"]
                break
            report = check(text, passages)
            if report.grounded:
                break
            prompt_q = question + RETRY_NOTE.format(problems="; ".join(report.violations[:3]))

        fell_back = report is None or not report.grounded
        violations = (report.violations if report else violations) if fell_back else []
        if fell_back:
            text = self.fallback.generate(question, passages)
            report = check(text, passages)
        t2 = time.perf_counter()
        return Answer(
            question=question,
            answer=text,
            sources=sources,
            grounded=report.grounded,
            generator=self.fallback.name if fell_back else self.generator.name,
            attempts=attempts,
            fell_back=fell_back,
            violations=violations,
            retrieval_ms=round((t1 - t0) * 1000, 2),
            generation_ms=round((t2 - t1) * 1000, 2),
        )
