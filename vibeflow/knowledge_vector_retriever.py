from __future__ import annotations

import numpy as np

from vibeflow.knowledge_embedder import Embedder
from vibeflow.knowledge_models import SearchResult, TextChunk


class VectorRetriever:
    """向量检索器。

    使用嵌入模型将查询和文本块编码为向量，通过余弦相似度计算相关性，
    按相似度从高到低排序，默认最多返回 3 条结果。

    可通过 search() 的 top_k 参数显式控制返回数量。
    """

    _DEFAULT_TOP_K = 3

    def __init__(
        self,
        embedder: Embedder,
        top_k: int | None = None,
    ) -> None:
        self._embedder = embedder
        self._top_k = top_k if top_k is not None else self._DEFAULT_TOP_K

    def search(
        self,
        chunks: list[TextChunk],
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        limit = top_k if top_k is not None else self._top_k

        # 空文本块集合 → 空结果
        if not chunks:
            return []

        # 空查询 → 空结果
        query = query.strip()
        if not query:
            return []

        # 过滤空内容文本块，但不应因为个别空块导致整体失败
        valid_entries: list[tuple[int, TextChunk]] = [
            (i, c) for i, c in enumerate(chunks) if c.content.strip()
        ]
        if not valid_entries:
            return []

        # 编码
        chunk_texts = [c.content for _, c in valid_entries]
        query_vec = self._embedder.encode([query])[0]
        chunk_vecs = self._embedder.encode(chunk_texts)

        # 余弦相似度
        similarities = self._cosine_similarity(query_vec, chunk_vecs)

        # 构建结果
        results: list[SearchResult] = []
        for idx, (orig_index, chunk) in enumerate(valid_entries):
            results.append(
                SearchResult(
                    content=chunk.content,
                    source_file=chunk.source_file,
                    chunk_index=chunk.chunk_index,
                    score=round(float(similarities[idx]), 4),
                )
            )

        results.sort(key=lambda r: (-r.score, r.chunk_index))
        return results[:limit]

    @staticmethod
    def _cosine_similarity(
        query_vec: np.ndarray, chunk_vecs: np.ndarray
    ) -> np.ndarray:
        """计算查询向量与每个文本块向量之间的余弦相似度。"""
        query_norm = np.linalg.norm(query_vec)
        chunk_norms = np.linalg.norm(chunk_vecs, axis=1)

        dot = np.dot(chunk_vecs, query_vec)
        denom = query_norm * chunk_norms
        # 避免除零：零向量时相似度定义为 0
        denom[denom == 0] = 1e-10

        return dot / denom
