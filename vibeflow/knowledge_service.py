from __future__ import annotations

from pathlib import Path

from vibeflow.knowledge_chunker import TextChunker
from vibeflow.knowledge_loader import KnowledgeLoader
from vibeflow.knowledge_retriever import KeywordRetriever, KnowledgeRetriever

# 延迟导入以避免在 keyword 模式加载 sentence-transformers
# VectorRetriever / HybridRetriever 仅在对应模式下导入


class KnowledgeService:
    """知识检索服务。

    组合 Loader → Chunker → Retriever 流水线，支持三种检索模式：

    - keyword：关键词检索（默认，无需额外依赖）
    - vector：向量语义检索（需要 sentence-transformers）
    - hybrid：混合检索（综合关键词和向量得分）
    """

    _VALID_MODES = {"keyword", "vector", "hybrid"}

    def __init__(
        self,
        knowledge_dir: str | Path = "knowledge",
        loader: KnowledgeLoader | None = None,
        chunker: TextChunker | None = None,
        retriever: KnowledgeRetriever | None = None,
        mode: str = "keyword",
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"未知检索模式：{mode}，可选值：{', '.join(sorted(self._VALID_MODES))}"
            )

        self._loader = loader or KnowledgeLoader(knowledge_dir)
        self._chunker = chunker or TextChunker()

        if retriever is not None:
            # 显式传入 retriever 时直接使用（便于测试注入）
            self._retriever = retriever
        elif mode == "keyword":
            self._retriever = KeywordRetriever()
        elif mode == "vector":
            from vibeflow.knowledge_embedder import Embedder
            from vibeflow.knowledge_vector_retriever import VectorRetriever

            self._retriever = VectorRetriever(Embedder())
        elif mode == "hybrid":
            from vibeflow.knowledge_embedder import Embedder
            from vibeflow.knowledge_hybrid_retriever import HybridRetriever
            from vibeflow.knowledge_vector_retriever import VectorRetriever

            self._retriever = HybridRetriever(
                KeywordRetriever(), VectorRetriever(Embedder())
            )

        self._mode = mode
        self._chunks = self._build_index()

    @property
    def mode(self) -> str:
        return self._mode

    def _build_index(self) -> list:
        from vibeflow.knowledge_models import TextChunk

        documents = self._loader.load()
        all_chunks: list[TextChunk] = []

        for doc in documents:
            chunks = self._chunker.chunk(doc["content"], doc["source_file"])
            all_chunks.extend(chunks)

        return all_chunks

    def search(self, query: str, top_k: int | None = None) -> list:
        return self._retriever.search(self._chunks, query, top_k=top_k)
