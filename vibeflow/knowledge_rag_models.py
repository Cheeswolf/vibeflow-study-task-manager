from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SourceInfo:
    """传给模型的实际引用来源信息。

    每个 SourceInfo 对应一个被纳入上下文的文本块，
    reference_label 与 prompt 中的 [SN] 编号一一对应。
    """

    reference_label: str
    source_file: str
    chunk_index: int
    score: float
    content_snippet: str


@dataclass(slots=True)
class RAGResult:
    """RAG 回答的结构化结果。

    不只是一个字符串 —— 包含回答、来源、检索模式、
    是否调用模型、是否因资料不足拒答等完整信息，
    便于后续 GUI、评估和日志扩展。
    """

    question: str
    answer: str
    sources: list[SourceInfo] = field(default_factory=list)
    retrieval_mode: str = "keyword"
    model_called: bool = False
    refused_due_to_insufficient: bool = False
    error_message: str | None = None
