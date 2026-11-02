"""Answer generators.

- ExtractiveGenerator: no model. Picks the passage sentences that best overlap the
  question and cites them. Always grounded; used offline, in CI, and as the fallback.
- AnthropicGenerator / OpenAICompatibleGenerator: call an LLM with a prompt that
  requires [n] citations. OpenAI-compatible also covers local Ollama servers.
Pick one with DOCQA_LLM = extractive | anthropic | openai.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Protocol

from .bm25 import tokenize

SYSTEM_PROMPT = """You answer questions using ONLY the numbered passages provided.
Rules:
- End every sentence with the citation(s) that support it, like [1] or [2][3].
- Copy numbers and code identifiers exactly as they appear in the passages.
- If the passages do not contain the answer, reply exactly: "The documentation provided does not answer this."
- Keep the answer under 120 words."""


def build_prompt(question: str, passages: list[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    return f"Passages:\n{numbered}\n\nQuestion: {question}"


class Generator(Protocol):
    name: str

    def generate(self, question: str, passages: list[str]) -> str: ...


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_sentences: int = 3, min_overlap: float = 0.5):
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap  # share of question terms a sentence must contain

    @staticmethod
    def prose_sentences(passage: str) -> list[str]:
        """Sentences from the prose of a passage: the heading line, code blocks and list fragments are skipped."""
        body = passage.split("\n", 1)[1] if "\n" in passage else passage
        body = re.sub(r"```.*?```", " ", body, flags=re.S)
        out = []
        for para in re.split(r"\n\s*\n", body):
            lines = [ln.strip() for ln in para.splitlines()]
            para = " ".join(ln for ln in lines if not ln.startswith(("*", "-", "|", "{")))
            for s in re.split(r"(?<=[.!?])\s+", para):
                s = s.strip()
                if 6 <= len(s.split()) <= 45 and s[-1:] in ".!" and not s.startswith(("**", "[")):
                    out.append(s)
        return out

    def generate(self, question: str, passages: list[str]) -> str:
        """Quote the lead prose of the best passage that covers enough of the question.

        Retrieval puts the right document first ~97% of the time on the eval set, so quoting
        the top passage is a safe fallback; it never adds words that are not in the docs."""
        q = set(tokenize(question, stemmed=True))
        for pi, p in enumerate(passages[:3], 1):
            coverage = len(q & set(tokenize(p, stemmed=True))) / (len(q) or 1)
            sents = self.prose_sentences(p)
            if coverage >= self.min_overlap and sents:
                heading = p.split("\n", 1)[0].split(" > ")[-1]
                lead = " ".join(f"{s} [{pi}]" for s in sents[: self.max_sentences])
                return f'From the section "{heading}" [{pi}]: {lead}'
        return "The documentation provided does not answer this."


class AnthropicGenerator:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("DOCQA_MODEL", "claude-haiku-4-5")
        self.name = f"anthropic:{self.model}"
        self.key = os.environ["ANTHROPIC_API_KEY"]

    def generate(self, question: str, passages: list[str]) -> str:
        body = {
            "model": self.model,
            "max_tokens": 400,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_prompt(question, passages)}],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode(),
            headers={"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        return "".join(b.get("text", "") for b in data["content"]).strip()


class OpenAICompatibleGenerator:
    """Works with OpenAI or any compatible server, e.g. Ollama at http://localhost:11434/v1."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.environ.get("DOCQA_MODEL", "llama3.1")
        self.base = (base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.key = os.environ.get("OPENAI_API_KEY", "ollama")
        self.name = f"openai:{self.model}"

    def generate(self, question: str, passages: list[str]) -> str:
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, passages)},
            ],
        }
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"authorization": f"Bearer {self.key}", "content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
        return data["choices"][0]["message"]["content"].strip()


def make_generator(spec: str | None = None) -> Generator:
    spec = spec or os.environ.get("DOCQA_LLM", "extractive")
    if spec == "anthropic":
        return AnthropicGenerator()
    if spec == "openai":
        return OpenAICompatibleGenerator()
    return ExtractiveGenerator()
