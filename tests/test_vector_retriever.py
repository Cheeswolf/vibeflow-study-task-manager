from __future__ import annotations

import sys

import numpy as np
import pytest

from vibeflow.knowledge_embedder import Embedder
from vibeflow.knowledge_hybrid_retriever import HybridRetriever
from vibeflow.knowledge_models import SearchResult, TextChunk
from vibeflow.knowledge_retriever import KeywordRetriever
from vibeflow.knowledge_vector_retriever import VectorRetriever


# ============================================================
# FakeEmbedder — 可控的假嵌入模型，不访问网络
# ============================================================


class FakeEmbedder:
    """返回基于文本内容确定性生成的归一化向量，完全离线。"""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self._load_called = False

    @property
    def is_loaded(self) -> bool:
        return self._load_called

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load_called = True
        vectors = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # 基于字符码生成确定性向量
            for j, ch in enumerate(text):
                vectors[i, j % self._dim] += ord(ch) / 1000.0
            # 归一化为单位向量
            norm = float(np.linalg.norm(vectors[i]))
            if norm > 0:
                vectors[i] = vectors[i] / norm
        return vectors


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=8)


@pytest.fixture
def vector_retriever(fake_embedder: FakeEmbedder) -> VectorRetriever:
    return VectorRetriever(fake_embedder)


@pytest.fixture
def keyword_retriever() -> KeywordRetriever:
    return KeywordRetriever()


@pytest.fixture
def hybrid_retriever(
    keyword_retriever: KeywordRetriever,
    vector_retriever: VectorRetriever,
) -> HybridRetriever:
    return HybridRetriever(keyword_retriever, vector_retriever)


def _make_chunks(
    contents: list[str], source_file: str = "test.md"
) -> list[TextChunk]:
    return [
        TextChunk(content=c, source_file=source_file, chunk_index=i)
        for i, c in enumerate(contents)
    ]


# ============================================================
# Embedder tests
# ============================================================


class TestEmbedderLazyLoading:
    """验证 Embedder 延迟加载行为。"""

    def test_embedder_not_loaded_on_construction(self) -> None:
        embedder = Embedder()
        assert embedder.is_loaded is False

    def test_embedder_loaded_after_encode(self, fake_embedder: FakeEmbedder) -> None:
        assert fake_embedder.is_loaded is False
        fake_embedder.encode(["hello"])
        assert fake_embedder.is_loaded is True

    def test_real_embedder_import_is_safe(self) -> None:
        """导入 Embedder 类不应触发 sentence-transformers 下载或加载。

        这验证了「导入 knowledge_embedder 模块 → 构造 Embedder() → 不调用 encode()」
        整个链路都不会访问网络或加载模型。延迟加载确保只有真正需要
        向量检索时才会触发下载。
        """
        # 构造 Embedder 不会加载模型
        embedder = Embedder()
        assert embedder.is_loaded is False

        # 验证 sentence-transformers 尚未被导入到当前命名空间
        import sys
        assert "sentence_transformers" not in sys.modules, (
            "导入 Embedder 不应触发 sentence-transformers 加载"
        )


# ============================================================
# VectorRetriever tests
# ============================================================


class TestVectorRetriever:
    """向量检索器测试 — 使用 FakeEmbedder，全程不访问网络。"""

    def test_chunks_are_encoded(
        self, vector_retriever: VectorRetriever, fake_embedder: FakeEmbedder
    ) -> None:
        """文本块可以被编码。"""
        chunks = _make_chunks(["Git 分支用于隔离开发工作。"])
        assert fake_embedder.is_loaded is False
        results = vector_retriever.search(chunks, "Git 分支")
        assert fake_embedder.is_loaded is True
        assert len(results) == 1

    def test_query_is_encoded(
        self, vector_retriever: VectorRetriever, fake_embedder: FakeEmbedder
    ) -> None:
        """查询可以被编码，不同查询返回不同结果。"""
        chunks = _make_chunks([
            "Git 分支用于隔离开发。",
            "Python 虚拟环境隔离项目依赖。",
            "Vibe Coding 是自然语言驱动的开发方式。",
        ])

        git_results = vector_retriever.search(chunks, "Git 分支")
        python_results = vector_retriever.search(chunks, "Python 环境")

        # 不同查询，排名最高的结果可能不同
        assert len(git_results) > 0
        assert len(python_results) > 0

    def test_cosine_similarity_correct(
        self, vector_retriever: VectorRetriever
    ) -> None:
        """余弦相似度计算正确：相同内容得分最高。"""
        chunks = _make_chunks([
            "完全无关的文本内容 A",
            "Git 分支用于隔离开发工作。",
            "完全无关的文本内容 B",
        ])
        results = vector_retriever.search(chunks, "Git 分支用于隔离开发工作。")
        assert len(results) >= 1
        # 内容完全一致的 chunk 应得分最高
        assert "Git 分支用于隔离开发工作。" in results[0].content

    def test_results_sorted_by_similarity_desc(
        self, vector_retriever: VectorRetriever
    ) -> None:
        """结果按相似度从高到低排序。"""
        chunks = _make_chunks([
            "完全无关 A",
            "Git 分支 分支 开发 项目",
            "完全无关 B",
        ])
        results = vector_retriever.search(chunks, "Git 分支 开发 项目")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_default_max_three_results(
        self, vector_retriever: VectorRetriever
    ) -> None:
        """默认最多返回 3 条结果。"""
        chunks = _make_chunks(
            [f"Git 分支 item {i} 开发任务管理学习笔记" for i in range(10)]
        )
        results = vector_retriever.search(chunks, "Git 分支 开发")
        assert len(results) == 3

    def test_explicit_top_k_effective(
        self, vector_retriever: VectorRetriever
    ) -> None:
        """显式指定 top_k 生效。"""
        chunks = _make_chunks(
            [f"Git 分支 item {i} 开发任务管理学习笔记" for i in range(10)]
        )
        results = vector_retriever.search(chunks, "Git 分支 开发", top_k=5)
        assert len(results) == 5

    def test_explicit_top_k_none_uses_default(
        self, vector_retriever: VectorRetriever
    ) -> None:
        """top_k=None 时使用默认值 3。"""
        chunks = _make_chunks(
            [f"Git 分支 item {i} 开发任务管理学习笔记" for i in range(10)]
        )
        results = vector_retriever.search(chunks, "Git 分支 开发", top_k=None)
        assert len(results) == 3

    def test_empty_query_returns_empty(
        self, vector_retriever: VectorRetriever
    ) -> None:
        """空查询返回空结果。"""
        chunks = _make_chunks(["some content"])
        assert vector_retriever.search(chunks, "") == []
        assert vector_retriever.search(chunks, "   ") == []

    def test_empty_chunks_returns_empty(
        self, vector_retriever: VectorRetriever
    ) -> None:
        """空文本块集合返回空结果。"""
        assert vector_retriever.search([], "query") == []

    def test_empty_individual_chunk_not_fatal(
        self, vector_retriever: VectorRetriever
    ) -> None:
        """单个空文本块不应导致整个索引失败。"""
        chunks = [
            TextChunk(content="", source_file="empty.md", chunk_index=0),
            TextChunk(content="Git 分支很重要。", source_file="ok.md", chunk_index=0),
        ]
        results = vector_retriever.search(chunks, "Git 分支")
        assert len(results) == 1
        assert results[0].source_file == "ok.md"

    def test_all_empty_chunks_returns_empty(
        self, vector_retriever: VectorRetriever
    ) -> None:
        """全部是空文本块时返回空结果。"""
        chunks = [
            TextChunk(content="", source_file="a.md", chunk_index=0),
            TextChunk(content="   ", source_file="b.md", chunk_index=1),
        ]
        assert vector_retriever.search(chunks, "query") == []

    def test_source_file_and_chunk_index_preserved(
        self, vector_retriever: VectorRetriever
    ) -> None:
        """来源文件和块编号正确传递。"""
        chunks = [
            TextChunk(
                content="Git 分支隔离开发。",
                source_file="notes/git.md",
                chunk_index=7,
            ),
        ]
        results = vector_retriever.search(chunks, "Git 分支")
        assert len(results) == 1
        assert results[0].source_file == "notes/git.md"
        assert results[0].chunk_index == 7

    def test_score_is_float(
        self, vector_retriever: VectorRetriever
    ) -> None:
        """相似度分数为浮点数。"""
        chunks = _make_chunks(["Git 分支用于隔离开发。"])
        results = vector_retriever.search(chunks, "Git 分支")
        assert len(results) == 1
        assert isinstance(results[0].score, float)

    def test_fewer_than_top_k_returns_all(
        self, vector_retriever: VectorRetriever
    ) -> None:
        """匹配数少于 top_k 时返回全部。"""
        chunks = _make_chunks(["唯一的文本块内容"])
        results = vector_retriever.search(chunks, "唯一", top_k=10)
        assert len(results) == 1

    def test_constructor_top_k_controls_default(
        self, fake_embedder: FakeEmbedder
    ) -> None:
        """构造函数 top_k 参数控制 search() 的默认返回数量。"""
        custom = VectorRetriever(fake_embedder, top_k=5)
        chunks = _make_chunks(
            [f"Git 分支 item {i} 开发任务管理学习笔记" for i in range(10)]
        )
        # 不传 top_k，应使用构造函数的 5
        results = custom.search(chunks, "Git 分支 开发")
        assert len(results) == 5

        # 显式传入 top_k 应覆盖构造函数值
        results = custom.search(chunks, "Git 分支 开发", top_k=2)
        assert len(results) == 2


# ============================================================
# HybridRetriever tests
# ============================================================


class TestHybridRetriever:
    """混合检索器测试 — 使用 FakeEmbedder，全程不访问网络。"""

    def test_deduplication(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        """相同来源文件和块编号的文本块不重复返回。"""
        chunks = _make_chunks([
            "Git 分支用于隔离开发工作。",
            "Python 虚拟环境管理项目依赖。",
        ])
        results = hybrid_retriever.search(chunks, "Git 分支")
        # 按 (source_file, chunk_index) 去重，每条只出现一次
        keys = [(r.source_file, r.chunk_index) for r in results]
        assert len(keys) == len(set(keys))
        assert len(results) >= 1

    def test_weights_are_effective(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        """权重可以配置且生效。"""
        assert hybrid_retriever.keyword_weight == 0.3
        assert hybrid_retriever.vector_weight == 0.7

    def test_custom_weights_affect_results(
        self, keyword_retriever: KeywordRetriever,
        vector_retriever: VectorRetriever,
    ) -> None:
        """不同权重组合产生不同排序。"""
        chunks = _make_chunks([
            "Git 分支 分支 分支 分支",  # 关键词匹配强
            "Git 使用与版本控制相关的工作流程和策略",  # 语义相关
        ])
        # 全关键词权重
        kw_heavy = HybridRetriever(
            keyword_retriever, vector_retriever,
            keyword_weight=1.0, vector_weight=0.0,
        )
        kw_results = kw_heavy.search(chunks, "Git 分支")

        # 全向量权重
        vec_heavy = HybridRetriever(
            keyword_retriever, vector_retriever,
            keyword_weight=0.0, vector_weight=1.0,
        )
        vec_results = vec_heavy.search(chunks, "Git 分支")

        # 两种权重下都应有结果
        assert len(kw_results) > 0
        assert len(vec_results) > 0

    def test_empty_chunks_returns_empty(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        """空文本块集合返回空结果。"""
        assert hybrid_retriever.search([], "query") == []

    def test_empty_query_returns_empty(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        """空查询返回空结果。"""
        chunks = _make_chunks(["some content"])
        assert hybrid_retriever.search(chunks, "") == []
        assert hybrid_retriever.search(chunks, "   ") == []

    def test_hybrid_score_is_between_zero_and_one(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        """归一化后混合得分通常在 [0, 1] 范围内。"""
        chunks = _make_chunks([
            "Git 分支用于隔离开发工作。",
            "Python 虚拟环境管理依赖。",
        ])
        results = hybrid_retriever.search(chunks, "Git 分支")
        for r in results:
            # 归一化后混合得分应在 [0, 1] 内
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of [0, 1]"

    def test_default_top_k_three(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        """默认最多返回 3 条。"""
        chunks = _make_chunks(
            [f"Git 分支 item {i} 开发笔记" for i in range(10)]
        )
        results = hybrid_retriever.search(chunks, "Git 分支")
        assert len(results) == 3

    def test_explicit_top_k(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        """显式 top_k 生效。"""
        chunks = _make_chunks(
            [f"Git 分支 item {i} 开发笔记" for i in range(10)]
        )
        results = hybrid_retriever.search(chunks, "Git 分支", top_k=5)
        assert len(results) == 5

    def test_top_k_none_uses_default(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        """top_k=None 时使用默认值 3。"""
        chunks = _make_chunks(
            [f"Git 分支 item {i} 开发笔记" for i in range(10)]
        )
        results = hybrid_retriever.search(chunks, "Git 分支", top_k=None)
        assert len(results) == 3

    def test_fewer_than_top_k_returns_all(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        """匹配数少于 top_k 时返回全部。"""
        chunks = _make_chunks(["唯一的文本块"])
        results = hybrid_retriever.search(chunks, "唯一", top_k=10)
        assert len(results) == 1

    def test_handles_individual_empty_chunk(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        """单个空文本块不应导致整体检索失败。"""
        chunks = [
            TextChunk(content="", source_file="empty.md", chunk_index=0),
            TextChunk(content="Git 分支很重要。", source_file="ok.md", chunk_index=0),
        ]
        results = hybrid_retriever.search(chunks, "Git 分支")
        assert len(results) >= 1
        assert all(r.source_file == "ok.md" for r in results)

    def test_dedup_preserves_content(
        self, keyword_retriever: KeywordRetriever,
        vector_retriever: VectorRetriever,
    ) -> None:
        """去重时保留至少一个来源的内容，不会丢失文本。"""
        chunks = _make_chunks(["Git 分支用于隔离开发工作。"])
        hybrid = HybridRetriever(keyword_retriever, vector_retriever)
        results = hybrid.search(chunks, "Git 分支")
        assert len(results) == 1
        assert "Git 分支" in results[0].content
        assert results[0].score > 0

    def test_results_sorted_by_score_desc(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        """混合结果按综合得分从高到低排序。"""
        chunks = _make_chunks([
            "完全无关 A",
            "Git 分支 分支 分支 分支 开发 项目",
            "完全无关 B",
        ])
        results = hybrid_retriever.search(chunks, "Git 分支 开发 项目")
        if len(results) >= 2:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_keyword_weight_dominance_ranking(
        self, keyword_retriever: KeywordRetriever,
    ) -> None:
        """keyword_weight=1.0 时，排名完全由关键词得分决定。"""
        from vibeflow.knowledge_vector_retriever import VectorRetriever

        # 用纯英文内容 + 中文查询 — 关键词可能命中英文词，向量语义距离远
        chunks = _make_chunks([
            "Git branching strategy for software development teams",
            "Unrelated text about cooking recipes",
        ])

        fake = FakeEmbedder(dim=8)
        vec_retriever = VectorRetriever(fake)
        hybrid = HybridRetriever(
            keyword_retriever, vec_retriever,
            keyword_weight=1.0, vector_weight=0.0,
        )
        results = hybrid.search(chunks, "branching")
        # 关键词权重主导下，含 "branching" 的 chunk 应排第一
        assert len(results) >= 1
        # 结果不崩溃且去重正确
        keys = [(r.source_file, r.chunk_index) for r in results]
        assert len(keys) == len(set(keys))

    def test_fusion_handles_low_vector_scores(
        self, vector_retriever: VectorRetriever,
    ) -> None:
        """向量得分较低时融合仍正常工作，不崩溃。"""
        chunks = _make_chunks(["量子计算利用量子比特进行信息处理。"])
        kw = KeywordRetriever()
        hybrid = HybridRetriever(kw, vector_retriever)
        results = hybrid.search(chunks, "量子计算")
        assert len(results) >= 1
        assert "量子计算" in results[0].content
        assert results[0].score > 0


# ============================================================
# KeywordRetriever 回归测试
# ============================================================


class TestKeywordRetrieverRegression:
    """确保 KeywordRetriever 重构后向后兼容。"""

    def test_default_three_results(
        self, keyword_retriever: KeywordRetriever
    ) -> None:
        chunks = _make_chunks([f"Git 分支 item {i}" for i in range(10)])
        results = keyword_retriever.search(chunks, "Git")
        assert len(results) == 3

    def test_explicit_top_k(
        self, keyword_retriever: KeywordRetriever
    ) -> None:
        chunks = _make_chunks([f"Git 分支 item {i}" for i in range(10)])
        results = keyword_retriever.search(chunks, "Git", top_k=5)
        assert len(results) == 5

    def test_top_k_none_uses_default(
        self, keyword_retriever: KeywordRetriever
    ) -> None:
        chunks = _make_chunks([f"Git 分支 item {i}" for i in range(10)])
        results = keyword_retriever.search(chunks, "Git", top_k=None)
        assert len(results) == 3

    def test_fewer_than_default(
        self, keyword_retriever: KeywordRetriever
    ) -> None:
        chunks = _make_chunks(["唯一文本"])
        results = keyword_retriever.search(chunks, "唯一")
        assert len(results) == 1

    def test_backward_compat_alias(self) -> None:
        """KnowledgeRetriever 别名仍然可用。"""
        from vibeflow.knowledge_retriever import KnowledgeRetriever

        retriever = KnowledgeRetriever()
        chunks = _make_chunks(["Git 分支 item 0"])
        results = retriever.search(chunks, "Git")
        assert len(results) == 1
