"""Grounding checks for generated answers.

An answer passes only if:
  1. every sentence that states something carries at least one citation like [2];
  2. every citation points at a passage that was actually retrieved;
  3. every number and every `code identifier` in a sentence appears in the
     passages that sentence cites.
Anything else is reported as a violation, and the pipeline falls back to an
extractive answer instead of showing an unsupported claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITE = re.compile(r"\[(\d+)\]")
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?(?![\w.])")
CODE = re.compile(r"`([^`]+)`")
SENTENCE = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z`])")
EXEMPT = re.compile(
    r"^(i (don't|do not) know|the (provided )?(docs|documentation|passages)( provided)? (do|does) not)", re.IGNORECASE
)


@dataclass
class GroundingReport:
    grounded: bool
    cited: list[int] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


def sentences(answer: str) -> list[str]:
    return [s.strip() for s in SENTENCE.split(answer.strip()) if s.strip()]


def check(answer: str, passages: list[str]) -> GroundingReport:
    """`passages` are the retrieved texts; citation [n] refers to passages[n-1]."""
    violations: list[str] = []
    cited: set[int] = set()
    lowered = [p.lower() for p in passages]
    for s in sentences(answer):
        refs = [int(n) for n in CITE.findall(s)]
        body = CITE.sub("", s).strip()
        if not body or EXEMPT.match(body):
            continue
        if not refs:
            violations.append(f"uncited claim: {body[:80]!r}")
            continue
        bad = [r for r in refs if not 1 <= r <= len(passages)]
        if bad:
            violations.append(f"citation to a passage that was not retrieved: {bad}")
            continue
        cited.update(refs)
        support = " ".join(lowered[r - 1] for r in refs)
        for num in NUMBER.findall(body):
            if num.lower() not in support:
                violations.append(f"number {num} not found in cited passages {refs}")
        for code in CODE.findall(body):
            if code.lower() not in support:
                violations.append(f"identifier `{code}` not found in cited passages {refs}")
    return GroundingReport(grounded=not violations, cited=sorted(cited), violations=violations)
