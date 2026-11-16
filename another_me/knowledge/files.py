"""Knowledge mode 'files': the model browses documents itself."""
from __future__ import annotations

import json
from pathlib import Path

DOC_SUFFIXES = {".md", ".txt"}
SKIP_NAMES = {"README.md"}


def iter_docs(roots: list[Path]):
    """Yield (display_path, absolute_path) for every knowledge document."""
    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in DOC_SUFFIXES and p.name not in SKIP_NAMES:
                yield f"{root.name}/{p.relative_to(root).as_posix()}", p


def _title(path: Path) -> str:
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            return line.lstrip("# ").strip()[:100]
    return path.stem


class FileTools:
    specs = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List the knowledge documents with their titles.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read one knowledge document by the path list_files returned.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    ]

    def __init__(self, roots: list[Path], max_chars: int = 6000):
        self.docs = dict(iter_docs(roots))
        self.max_chars = max_chars

    def list_files(self) -> str:
        rows = [
            {"path": name, "title": _title(p), "chars": p.stat().st_size}
            for name, p in self.docs.items()
        ]
        return json.dumps(rows, ensure_ascii=False)

    def read_file(self, path: str) -> str:
        # Only paths that list_files returned are readable, so the model cannot
        # reach outside the knowledge folders with ../ or absolute paths.
        doc = self.docs.get(path)
        if doc is None:
            return f"error: no such document '{path}'. Call list_files for valid paths."
        text = doc.read_text(errors="replace")
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n[truncated]"
        return text

    def call(self, name: str, arguments: dict) -> str:
        if name == "list_files":
            return self.list_files()
        if name == "read_file":
            return self.read_file(str(arguments.get("path", "")))
        return f"error: unknown tool '{name}'"
