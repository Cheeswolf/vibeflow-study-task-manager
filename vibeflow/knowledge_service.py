from __future__ import annotations

from pathlib import Path

from vibeflow.knowledge_chunker import TextChunker
from vibeflow.knowledge_loader import KnowledgeLoader
from vibeflow.knowledge_retriever import KnowledgeRetriever


class KnowledgeService:
    def __init__(
        self,
        knowledge_dir: str | Path = "knowledge",
        loader: KnowledgeLoader | None = None,
        chunker: TextChunker | None = None,
        retriever: KnowledgeRetriever | None = None,
    ) -> None:
        self._loader = loader or KnowledgeLoader(knowledge_dir)
        self._chunker = chunker or TextChunker()
        self._retriever = retriever or KnowledgeRetriever()
        self._chunks = self._build_index()

    def _build_index(self) -> list:
        from vibeflow.knowledge_models import TextChunk

        documents = self._loader.load()
        all_chunks: list[TextChunk] = []

        for doc in documents:
            chunks = self._chunker.chunk(doc["content"], doc["source_file"])
            all_chunks.extend(chunks)

        return all_chunks

    def search(self, query: str) -> list:
        return self._retriever.search(self._chunks, query)
