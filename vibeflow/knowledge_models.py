from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TextChunk:
    content: str
    source_file: str
    chunk_index: int


@dataclass(slots=True)
class SearchResult:
    content: str
    source_file: str
    chunk_index: int
    score: float
