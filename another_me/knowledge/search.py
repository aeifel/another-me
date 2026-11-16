"""Knowledge mode 'search': retrieve, rank, return passages.

v0 ranks with BM25 in pure Python: no vector database, no server, no model download.
A personal knowledge base is a few hundred chunks, so brute force is instant.
Planned next (see the roadmap in the README): add an embedding model, fuse both rankings,
then rerank the top results.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .files import iter_docs

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def chunk_markdown(text: str, max_chars: int) -> list[tuple[str, str]]:
    """Split into (heading, passage) chunks: by heading first, then by paragraph."""
    chunks: list[tuple[str, str]] = []
    heading, buf = "", []

    def flush():
        body = "\n\n".join(buf).strip()
        if body:
            chunks.append((heading, body))
        buf.clear()

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if para.startswith("#"):
            flush()
            first, _, rest = para.partition("\n")
            heading = first.lstrip("# ").strip()
            if rest.strip():
                buf.append(rest.strip())
            continue
        if buf and sum(len(b) for b in buf) + len(para) > max_chars:
            flush()
        buf.append(para)
    flush()
    return chunks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg_len = (sum(self.lengths) / len(docs)) if docs else 0.0
        df = Counter(term for d in docs for term in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.tf, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
            out.append(
                sum(
                    self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                    for t in query
                    if t in tf
                )
            )
        return out


class SearchTool:
    specs = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "Search the knowledge base and return the most relevant passages.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    def __init__(self, roots: list[Path], chunk_chars: int = 900, top_k: int = 4):
        self.top_k = top_k
        self.chunks: list[dict] = []
        for name, path in iter_docs(roots):
            for heading, body in chunk_markdown(path.read_text(errors="replace"), chunk_chars):
                self.chunks.append({"path": name, "heading": heading, "text": body})
        self.index = BM25([tokenize(c["heading"] + " " + c["text"]) for c in self.chunks])

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        scores = self.index.scores(tokenize(query))
        ranked = sorted(zip(scores, self.chunks), key=lambda x: x[0], reverse=True)
        return [
            {**chunk, "score": round(score, 3)}
            for score, chunk in ranked[: top_k or self.top_k]
            if score > 0
        ]

    def call(self, name: str, arguments: dict) -> str:
        if name != "search_knowledge":
            return f"error: unknown tool '{name}'"
        hits = self.search(str(arguments.get("query", "")))
        if not hits:
            return "No relevant passages found."
        return json.dumps(hits, ensure_ascii=False)
