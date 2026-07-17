from __future__ import annotations

from pathlib import Path

import pytest

from vibeflow.knowledge_chunker import TextChunker
from vibeflow.knowledge_loader import KnowledgeLoader
from vibeflow.knowledge_models import TextChunk
from vibeflow.knowledge_retriever import KnowledgeRetriever
from vibeflow.knowledge_service import KnowledgeService


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    kd = tmp_path / "knowledge"
    kd.mkdir()
    return kd


@pytest.fixture
def loader(knowledge_dir: Path) -> KnowledgeLoader:
    return KnowledgeLoader(knowledge_dir)


@pytest.fixture
def chunker() -> TextChunker:
    return TextChunker()


@pytest.fixture
def retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever()


def _make_chunks(
    contents: list[str], source_file: str = "test.md"
) -> list[TextChunk]:
    return [
        TextChunk(content=c, source_file=source_file, chunk_index=i)
        for i, c in enumerate(contents)
    ]


# ============================================================
# KnowledgeLoader tests
# ============================================================


class TestKnowledgeLoader:
    def test_load_md_and_txt_files(self, knowledge_dir: Path) -> None:
        (knowledge_dir / "a.md").write_text("# A", encoding="utf-8")
        (knowledge_dir / "b.txt").write_text("B content", encoding="utf-8")
        (knowledge_dir / "c.pdf").write_text("C", encoding="utf-8")

        docs = KnowledgeLoader(knowledge_dir).load()

        sources = [d["source_file"] for d in docs]
        assert len(docs) == 2
        assert any(s.endswith("a.md") for s in sources)
        assert any(s.endswith("b.txt") for s in sources)

    def test_ignore_other_formats(self, knowledge_dir: Path) -> None:
        (knowledge_dir / "a.md").write_text("md")
        (knowledge_dir / "b.pdf").write_text("pdf")
        (knowledge_dir / "c.png").write_text("png")
        (knowledge_dir / "d_no_ext").write_text("none")

        docs = KnowledgeLoader(knowledge_dir).load()

        assert len(docs) == 1  # only a.md

    def test_empty_directory_returns_empty_list(
        self, knowledge_dir: Path
    ) -> None:
        docs = KnowledgeLoader(knowledge_dir).load()
        assert docs == []

    def test_directory_not_found_raises_error(self) -> None:
        with pytest.raises(FileNotFoundError, match="知识目录不存在"):
            KnowledgeLoader("/不存在的路径")

    def test_non_recursive_load(self, knowledge_dir: Path) -> None:
        (knowledge_dir / "root.md").write_text("root")
        sub = knowledge_dir / "sub"
        sub.mkdir()
        (sub / "nested.md").write_text("nested")

        docs = KnowledgeLoader(knowledge_dir).load()

        assert len(docs) == 1  # only root.md, nested is ignored

    def test_file_encoding(self, knowledge_dir: Path) -> None:
        (knowledge_dir / "cn.md").write_text("中文内容测试", encoding="utf-8")

        docs = KnowledgeLoader(knowledge_dir).load()

        assert "中文内容测试" in docs[0]["content"]


# ============================================================
# TextChunker tests
# ============================================================


class TestTextChunker:
    def test_split_by_paragraph(self, chunker: TextChunker) -> None:
        text = (
            "这是第一段内容足够长的文本用于测试切分功能。\n\n"
            "这是第二段内容足够长的文本用于测试切分功能。\n\n"
            "这是第三段内容足够长的文本用于测试切分功能。"
        )

        result = chunker.chunk(text, "test.md")

        assert len(result) == 3
        assert result[0].chunk_index == 0
        assert result[1].chunk_index == 1
        assert result[2].chunk_index == 2

    def test_merge_short_chunks(self, chunker: TextChunker) -> None:
        text = "Hi\n\nThis paragraph is long enough to stand on its own."

        result = chunker.chunk(text, "test.md")

        assert len(result) == 1
        assert "Hi" in result[0].content

    def test_merge_cascade(self, chunker: TextChunker) -> None:
        text = "A\n\nB\n\nC\n\nThis is a long enough paragraph now to stop merging."

        result = chunker.chunk(text, "test.md")

        assert len(result) == 1
        assert all(x in result[0].content for x in ["A", "B", "C"])

    def test_split_large_chunk(self, chunker: TextChunker) -> None:
        sentence = "这是一个用于生成超过八百字符的长句子内容。" * 50
        assert len(sentence) > 800

        result = chunker.chunk(sentence, "test.md")

        assert len(result) > 1
        for chunk in result:
            assert len(chunk.content) <= 800

    def test_code_block_not_split(self, chunker: TextChunker) -> None:
        text = "Before.\n\n```\nline1\n\nline2\nline3\n```\n\nAfter."

        result = chunker.chunk(text, "test.md")

        code_chunks = [c for c in result if "```" in c.content]
        assert len(code_chunks) == 1
        assert "line1" in code_chunks[0].content
        assert "line2" in code_chunks[0].content

    def test_empty_document_returns_empty(self, chunker: TextChunker) -> None:
        result = chunker.chunk("", "test.md")
        assert result == []

        result = chunker.chunk("   \n\n  ", "test.md")
        assert result == []

    def test_chunk_index_increment(self, chunker: TextChunker) -> None:
        text = "A\n\nB\n\nC\n\nD\n\nE"

        result = chunker.chunk(text, "test.md")

        indices = [c.chunk_index for c in result]
        assert indices == list(range(len(result)))
        for chunk in result:
            assert chunk.source_file == "test.md"


# ============================================================
# KnowledgeRetriever tests
# ============================================================


class TestKnowledgeRetriever:
    def test_chinese_keyword_match(self, retriever: KnowledgeRetriever) -> None:
        chunks = _make_chunks(["创建 Git 分支可以隔离开发工作。"])
        results = retriever.search(chunks, "Git分支")
        assert len(results) == 1
        assert results[0].score > 0

    def test_english_keyword_match(
        self, retriever: KnowledgeRetriever
    ) -> None:
        chunks = _make_chunks(["You should create a new branch for each feature."])
        results = retriever.search(chunks, "create branch")
        assert len(results) == 1
        assert results[0].score > 0

    def test_longer_match_scores_higher(
        self, retriever: KnowledgeRetriever
    ) -> None:
        chunks = _make_chunks(
            ["Git 分支", "分 支 分 支"]  # contains full word  # only single chars
        )
        results = retriever.search(chunks, "分支")
        assert len(results) >= 1
        assert results[0].source_file == "test.md"

    def test_short_chunk_ranks_above_long(
        self, retriever: KnowledgeRetriever
    ) -> None:
        short = "Git 分支很重要。" * 1  # ~10 chars, contains keyword
        long = "无关内容 " * 200 + "Git 分支很重要。"  # ~1000 chars, same keyword at end

        chunks = [
            TextChunk(content=long, source_file="f1.md", chunk_index=0),
            TextChunk(content=short, source_file="f2.md", chunk_index=0),
        ]
        results = retriever.search(chunks, "Git")
        assert len(results) >= 1
        # short chunk should rank higher (same term, less penalty from length)
        assert results[0].source_file == "f2.md"

    def test_empty_query_returns_empty(
        self, retriever: KnowledgeRetriever
    ) -> None:
        chunks = _make_chunks(["some content"])
        results = retriever.search(chunks, "")
        assert results == []

    def test_stopwords_only_returns_empty(
        self, retriever: KnowledgeRetriever
    ) -> None:
        chunks = _make_chunks(["some content about things"])
        results = retriever.search(chunks, "的 了 在")
        assert results == []

    def test_no_match_returns_empty(
        self, retriever: KnowledgeRetriever
    ) -> None:
        chunks = _make_chunks(["Git 分支用于隔离开发。"])
        results = retriever.search(chunks, "量子纠缠")
        assert results == []

    def test_results_sorted_by_score_desc(
        self, retriever: KnowledgeRetriever
    ) -> None:
        chunks = [
            TextChunk(content="Git", source_file="low.md", chunk_index=0),
            TextChunk(
                content="Git 分支 分支 开发 开发 项目 项目 提交 提交",
                source_file="high.md",
                chunk_index=0,
            ),
            TextChunk(content="Git 分支", source_file="mid.md", chunk_index=0),
        ]
        results = retriever.search(chunks, "Git 分支 开发 项目 提交")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_max_three_results(self, retriever: KnowledgeRetriever) -> None:
        # Create 10 chunks all containing the keyword
        chunks = _make_chunks([f"Git 分支 item {i}" for i in range(10)])
        results = retriever.search(chunks, "Git")
        assert len(results) == 3

    def test_fewer_than_three_results(
        self, retriever: KnowledgeRetriever
    ) -> None:
        chunks = _make_chunks(["Git 分支很重要。"])
        results = retriever.search(chunks, "Git")
        assert len(results) == 1


# ============================================================
# KnowledgeService tests
# ============================================================


class TestKnowledgeService:
    _GIT_NOTE = (
        "在项目开始时初始化 Git，并在关键开发节点提交代码。\n\n"
        "Git 提交相当于为项目建立一个个可恢复的存档点。\n\n"
        "当后续开发出现严重问题时，可以回到之前相对稳定的版本。\n\n"
        "开发新功能时应该创建新分支，这样才能隔离不同功能的开发工作。"
    )

    _CLAUDE_NOTE = (
        "每个项目用自己独立的虚拟环境，互不污染。\n\n"
        "进项目就激活 venv，所有 python/pip/pytest 都自动走这个项目的环境。"
    )

    _VIBE_NOTE = (
        "Vibe Coding 是一种通过自然语言对软件开发过程进行持续控制的开发方式。\n\n"
        "开发者向 Claude Code 描述需求，Claude Code 根据需求生成或修改产品。"
    )

    def test_e2e_with_real_knowledge_files(
        self, knowledge_dir: Path
    ) -> None:
        (knowledge_dir / "git-notes.md").write_text(
            self._GIT_NOTE, encoding="utf-8"
        )
        (knowledge_dir / "claude-code.md").write_text(
            self._CLAUDE_NOTE, encoding="utf-8"
        )
        (knowledge_dir / "vibe-coding.md").write_text(
            self._VIBE_NOTE, encoding="utf-8"
        )

        service = KnowledgeService(knowledge_dir)
        results = service.search("虚拟环境")

        assert len(results) > 0
        assert any("虚拟环境" in r.content for r in results)

    def test_search_result_structure(
        self, knowledge_dir: Path
    ) -> None:
        (knowledge_dir / "test.md").write_text(
            "Git 分支用于隔离开发功能。", encoding="utf-8"
        )

        service = KnowledgeService(knowledge_dir)
        results = service.search("Git")

        assert len(results) > 0
        r = results[0]
        assert isinstance(r.content, str)
        assert isinstance(r.source_file, str)
        assert isinstance(r.chunk_index, int)
        assert isinstance(r.score, float)
        assert r.score > 0

    def test_acceptance_example(self, knowledge_dir: Path) -> None:
        (knowledge_dir / "git-notes.md").write_text(
            self._GIT_NOTE, encoding="utf-8"
        )
        (knowledge_dir / "claude-code.md").write_text(
            self._CLAUDE_NOTE, encoding="utf-8"
        )
        (knowledge_dir / "vibe-coding.md").write_text(
            self._VIBE_NOTE, encoding="utf-8"
        )

        service = KnowledgeService(knowledge_dir)
        results = service.search("为什么开发新功能要创建 Git 分支？")

        assert len(results) > 0
        assert "git-notes.md" in results[0].source_file.replace("\\", "/")
