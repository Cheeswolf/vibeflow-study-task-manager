from __future__ import annotations

from vibeflow.knowledge_models import SearchResult
from vibeflow.knowledge_rag_models import SourceInfo

# --- 上下文控制常量 ---

# 最多纳入上下文的文本块数量
MAX_CONTEXT_CHUNKS = 5

# 单个文本块的最大字符数（超出部分截断）
MAX_CHUNK_CHARS = 500

# 上下文总字符数上限
MAX_TOTAL_CHARS = 3000


class ContextBuilder:
    """将检索结果整理为模型上下文。

    负责：
    1. 按得分高低选择文本块（retriever 已排序）
    2. 截断过长文本块
    3. 控制总上下文长度
    4. 确保不会因为一个超长块挤掉所有其他来源
    5. 生成带编号的 [SN] 格式文本
    6. 返回实际使用的来源列表
    """

    def __init__(
        self,
        max_chunks: int | None = None,
        max_chunk_chars: int | None = None,
        max_total_chars: int | None = None,
    ) -> None:
        self._max_chunks = max_chunks if max_chunks is not None else MAX_CONTEXT_CHUNKS
        self._max_chunk_chars = (
            max_chunk_chars if max_chunk_chars is not None else MAX_CHUNK_CHARS
        )
        self._max_total_chars = (
            max_total_chars if max_total_chars is not None else MAX_TOTAL_CHARS
        )

    @property
    def max_chunks(self) -> int:
        """公共只读属性，供 RAGService 等调用方使用。"""
        return self._max_chunks

    def build(
        self, results: list[SearchResult]
    ) -> tuple[str, list[SourceInfo]]:
        """从检索结果构建上下文文本和来源列表。

        Args:
            results: 已按得分降序排列的检索结果。

        Returns:
            (context_text, sources)：
            - context_text：格式化后的上下文字符串，可直接嵌入 prompt
            - sources：实际使用的来源信息列表
        """
        if not results:
            return "", []

        sources: list[SourceInfo] = []
        context_parts: list[str] = []
        total_chars = 0

        for i, r in enumerate(results):
            if len(sources) >= self._max_chunks:
                break

            # 截断过长文本块（在自然边界处截断）
            snippet = self._truncate(r.content, self._max_chunk_chars)

            label = f"[S{len(sources) + 1}]"

            # 构建单个上下文块
            block = (
                f"{label}\n"
                f"source: {r.source_file}\n"
                f"chunk_id: {r.chunk_index}\n"
                f"content: {snippet}\n"
            )

            # 检查总长度限制（第一个块即使超出也保留，至少有一个）
            if sources and total_chars + len(block) > self._max_total_chars:
                break

            context_parts.append(block)
            total_chars += len(block)

            sources.append(
                SourceInfo(
                    reference_label=label,
                    source_file=r.source_file,
                    chunk_index=r.chunk_index,
                    score=r.score,
                    content_snippet=snippet,
                )
            )

        return "\n".join(context_parts), sources

    @staticmethod
    def _truncate(content: str, max_chars: int) -> str:
        """截断文本到指定长度，尽量在自然边界处截断。

        优先在句末标点处截断，其次是换行，最后是空格。
        不会在字符串中间以明显不合理的方式截断。
        """
        if len(content) <= max_chars:
            return content

        # 在限制范围内寻找最佳截断点
        truncated = content[:max_chars]

        # 优先级：句末标点 > 换行 > 空格
        for boundary in ["。", "！", "？", ".", "!", "?", "\n", " "]:
            last = truncated.rfind(boundary)
            if last > max_chars // 2:
                return truncated[: last + 1]

        return truncated
