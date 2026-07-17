from __future__ import annotations

from vibeflow.knowledge_models import SearchResult, TextChunk
from vibeflow.knowledge_retriever import KeywordRetriever
from vibeflow.knowledge_vector_retriever import VectorRetriever


class HybridRetriever:
    """混合检索器。

    综合关键词检索得分和向量相似度得分，使用加权融合：

        hybrid_score = keyword_weight × norm_keyword_score
                     + vector_weight × norm_vector_score

    不同得分先进行 min-max 归一化到 [0, 1] 区间，
    避免量纲差异导致一方完全失效。
    相同文本块（相同来源文件 + 相同块编号）不会重复返回。

    默认权重：keyword=0.3, vector=0.7。
    向量检索占主导，因为其语义理解能力更强；
    关键词检索提供精确术语匹配的补充信号。
    """

    _DEFAULT_KEYWORD_WEIGHT = 0.3
    _DEFAULT_VECTOR_WEIGHT = 0.7
    _DEFAULT_TOP_K = 3

    def __init__(
        self,
        keyword_retriever: KeywordRetriever,
        vector_retriever: VectorRetriever,
        keyword_weight: float | None = None,
        vector_weight: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self._keyword = keyword_retriever
        self._vector = vector_retriever
        self._kw_weight = (
            keyword_weight
            if keyword_weight is not None
            else self._DEFAULT_KEYWORD_WEIGHT
        )
        self._vec_weight = (
            vector_weight
            if vector_weight is not None
            else self._DEFAULT_VECTOR_WEIGHT
        )
        self._top_k = top_k if top_k is not None else self._DEFAULT_TOP_K

    @property
    def keyword_weight(self) -> float:
        return self._kw_weight

    @property
    def vector_weight(self) -> float:
        return self._vec_weight

    def search(
        self,
        chunks: list[TextChunk],
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        limit = top_k if top_k is not None else self._top_k

        if not chunks:
            return []

        query = query.strip()
        if not query:
            return []

        # 获取全部评分结果（不限数量，用于准确归一化）
        total = len(chunks)
        kw_results = self._keyword.search(chunks, query, top_k=total)
        vec_results = self._vector.search(chunks, query, top_k=total)

        # 归一化到 [0, 1]
        kw_map = self._normalize(kw_results)
        vec_map = self._normalize(vec_results)

        # 去重合并：以 (source_file, chunk_index) 为 key
        combined: dict[tuple[str, int], SearchResult] = {}
        all_keys = set(kw_map.keys()) | set(vec_map.keys())

        for key in all_keys:
            kw_score = kw_map.get(key, 0.0)
            vec_score = vec_map.get(key, 0.0)
            hybrid = round(
                self._kw_weight * kw_score + self._vec_weight * vec_score, 4
            )

            # 优先取两个来源中任一个的元数据（保留原始 content）
            source = kw_results[0] if kw_results else vec_results[0]
            for r in kw_results:
                if (r.source_file, r.chunk_index) == key:
                    source = r
                    break
            else:
                for r in vec_results:
                    if (r.source_file, r.chunk_index) == key:
                        source = r
                        break

            combined[key] = SearchResult(
                content=source.content,
                source_file=source.source_file,
                chunk_index=source.chunk_index,
                score=hybrid,
            )

        results = list(combined.values())
        results.sort(key=lambda r: (-r.score, r.chunk_index))
        return results[:limit]

    @staticmethod
    def _normalize(
        results: list[SearchResult],
    ) -> dict[tuple[str, int], float]:
        """Min-max 归一化到 [0, 1]，返回 {key: normalized_score}。"""
        if not results:
            return {}

        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)

        if max_s == min_s:
            # 所有得分相同，全部归一化为 1.0
            return {
                (r.source_file, r.chunk_index): 1.0 for r in results
            }

        return {
            (r.source_file, r.chunk_index): (r.score - min_s) / (max_s - min_s)
            for r in results
        }
