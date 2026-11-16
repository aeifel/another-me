"""Two interchangeable ways to give the model facts. Only one is attached per run.

files  -> the model lists documents and reads the ones it picks (FileTools)
search -> the model sends a query and gets ranked passages back (SearchTool)
"""
from __future__ import annotations

from ..config import resolve
from .files import FileTools
from .search import SearchTool

MODES = ("none", "files", "search")


def make_toolbox(mode: str, cfg: dict):
    if mode == "none":
        return None
    roots = [resolve(r) for r in cfg["knowledge"]["roots"]]
    roots = [r for r in roots if r.is_dir()]
    if mode == "files":
        return FileTools(roots)
    if mode == "search":
        k = cfg["knowledge"]
        return SearchTool(roots, chunk_chars=k["chunk_chars"], top_k=k["top_k"])
    raise ValueError(f"knowledge mode must be one of {MODES}")
