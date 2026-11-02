"""Markdown-aware chunking.

Documents are split on headings first, so a chunk never mixes two sections.
Each section is then packed paragraph by paragraph (code blocks stay whole) into
chunks of at most `max_words` words; a short trailing paragraph is repeated at
the start of the next chunk as overlap. Only a single paragraph longer than
`max_words` is cut by a sliding word window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*(\{\s*#[^}]*\})?\s*$")
INCLUDE = re.compile(r"^\{\*.*\*\}\s*$")  # mkdocs include directives carry no text
HTML_TAG = re.compile(r"<[^>]+>")


@dataclass
class Chunk:
    id: str
    doc: str  # source path relative to the corpus root
    heading: str  # "Page title > Section"
    text: str
    meta: dict = field(default_factory=dict)


ADMONITION = re.compile(r"^///")  # mkdocs "/// tip" boxes: keep the text, drop the marker


def clean_line(line: str) -> str:
    if INCLUDE.match(line.strip()) or ADMONITION.match(line.strip()):
        return ""
    return HTML_TAG.sub("", line)


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return (heading_path, body) pairs; heading_path joins the page title and section."""
    title, current, buf, out = "", "", [], []
    in_code = False
    for raw in markdown.splitlines():
        if raw.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(raw)
        if m:
            if buf and "".join(buf).strip():
                out.append((current, "\n".join(buf).strip()))
            buf = []
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1:
                title, current = text, text
            else:
                current = f"{title} > {text}" if title else text
            continue
        buf.append(clean_line(raw))
    if buf and "".join(buf).strip():
        out.append((current, "\n".join(buf).strip()))
    return out


def window(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    if max_words <= overlap:
        raise ValueError("max_words must be larger than overlap")
    if len(words) <= max_words:
        return [words]
    step = max_words - overlap
    return [words[i : i + max_words] for i in range(0, len(words) - overlap, step)]


def blocks(body: str) -> list[str]:
    """Paragraphs and fenced code blocks, each kept whole (a code block is never cut in half by a blank line)."""
    out, cur, in_code = [], [], False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if not line.strip() and not in_code:
            if cur:
                out.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return out


def pack(body: str, max_words: int, overlap: int) -> list[str]:
    """Greedily pack whole blocks into chunks of <= max_words; split only blocks that are too long by themselves.
    The last block of a chunk is repeated at the start of the next when it is short enough (overlap)."""
    parts: list[str] = []
    for b in blocks(body):
        n = len(b.split())
        if n > max_words:
            parts.extend(" ".join(w) for w in window(b.split(), max_words, overlap))
        else:
            parts.append(b)
    chunks, cur, size = [], [], 0
    for b in parts:
        n = len(b.split())
        if cur and size + n > max_words:
            chunks.append("\n\n".join(cur))
            tail = cur[-1] if len(cur[-1].split()) <= overlap else None
            cur, size = ([tail], len(tail.split())) if tail else ([], 0)
        cur.append(b)
        size += n
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_markdown(doc: str, markdown: str, max_words: int = 180, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in split_sections(markdown):
        for part in pack(body, max_words, overlap):
            n = len(chunks)
            chunks.append(Chunk(id=f"{doc}#{n}", doc=doc, heading=heading, text=part))
    return chunks


def load_corpus(root: str | Path, pattern: str = "**/*.md", **kw) -> list[Chunk]:
    root = Path(root)
    out: list[Chunk] = []
    for path in sorted(root.glob(pattern)):
        rel = path.relative_to(root).as_posix()
        out.extend(chunk_markdown(rel, path.read_text(encoding="utf-8", errors="ignore"), **kw))
    return out
