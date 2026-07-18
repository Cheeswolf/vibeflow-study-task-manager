"""确定性评分规则。

所有规则基于字符串匹配和简单计数，不依赖 LLM 评判。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibeflow.knowledge_rag_models import SourceInfo
    from vibeflow.evaluation.models import CaseResult, EvaluationCase

# 匹配 [S数字] 引用标签
_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


def check_source_hit(
    expected_sources: list[str],
    actual_sources: list[str],
    require_all: bool = False,
) -> bool:
    """检查预期来源是否命中。

    Args:
        expected_sources: 案例指定的预期来源文件名列表。
        actual_sources: RAGResult 中实际返回的来源文件名列表。
        require_all: True 时要求所有预期来源都出现；
                     False 时至少命中一个即可。

    Returns:
        命中返回 True。
    """
    if not expected_sources:
        return True  # 没有设定预期来源则不评判

    actual_set = set(actual_sources)

    if require_all:
        return all(
            any(src in a for a in actual_set) for src in expected_sources
        )
    else:
        return any(
            any(src in a for a in actual_set) for src in expected_sources
        )


def check_citation_valid(
    answer: str,
    sources: list[SourceInfo],
) -> bool:
    """检查回答中所有 [SN] 引用是否对应真实来源。

    提取回答中所有 [S数字] 标签，验证每个数字
    都对应 sources 列表中的有效索引（1-based）。

    Args:
        answer: 模型回答文本。
        sources: 实际传给模型的 SourceInfo 列表。

    Returns:
        所有引用有效返回 True，否则 False。
    """
    if not answer:
        return True  # 空回答视为引用有效

    citations = _CITATION_PATTERN.findall(answer)
    if not citations:
        return True  # 没有引用视为有效

    max_index = len(sources)
    for num_str in citations:
        idx = int(num_str)
        if idx < 1 or idx > max_index:
            return False

    return True


def check_keyword_coverage(
    answer: str,
    expected_keywords: list[str],
) -> float:
    """计算预期关键词在回答中的覆盖率。

    使用简单的字符串包含匹配（大小写不敏感）。

    Args:
        answer: 模型回答文本。
        expected_keywords: 期望出现的关键词列表。

    Returns:
        覆盖率，0.0 到 1.0。
    """
    if not expected_keywords:
        return 1.0  # 没有设定关键词则不评判

    answer_lower = answer.lower()
    matched = 0
    for kw in expected_keywords:
        if kw.lower() in answer_lower:
            matched += 1

    return matched / len(expected_keywords)


def check_forbidden(
    answer: str,
    forbidden_keywords: list[str],
) -> bool:
    """检查回答是否包含禁止关键词。

    Args:
        answer: 模型回答文本。
        forbidden_keywords: 不应出现的关键词列表。

    Returns:
        命中禁止词返回 True（表示有问题）。
    """
    if not forbidden_keywords:
        return False

    answer_lower = answer.lower()
    for kw in forbidden_keywords:
        if kw.lower() in answer_lower:
            return True

    return False


def score_case(
    case: "EvaluationCase",
    answer: str,
    sources: list[SourceInfo],
    model_called: bool,
    latency_ms: float,
    error: str | None,
) -> "CaseResult":
    """对单条案例进行确定性评分。

    按以下规则逐一检查：
    1. 拒答判断 — should_answer=false 时验证模型未被调用/明确拒答
    2. 来源命中 — 检查 expected_sources 是否在结果中
    3. 关键词覆盖 — 计算 expected_keywords 命中率
    4. 引用有效 — 验证 [SN] 标签真实性
    5. 禁止词 — 检查 forbidden_keywords 是否出现

    Args:
        case: 评估案例。
        answer: RAGService 返回的回答文本。
        sources: RAGService 返回的来源列表。
        model_called: 是否实际调用了 LLM。
        latency_ms: 响应时间（毫秒）。
        error: 错误信息（如有）。

    Returns:
        完整的 CaseResult。
    """
    failure_reasons: list[str] = []
    actual_sources = [s.source_file for s in sources]

    # --- 计算各项指标 ---
    source_hit = check_source_hit(
        case.expected_sources, actual_sources, case.require_all_sources
    )
    citation_valid = check_citation_valid(answer, sources)
    keyword_coverage = check_keyword_coverage(answer, case.expected_keywords)
    forbidden_hit = check_forbidden(answer, case.forbidden_keywords)

    # 判断是否实际回答了问题
    actually_answered = model_called and not error

    # --- 拒答判断 ---
    if not case.should_answer:
        # 不应该回答：模型不应被调用，或被调用但系统明确拒答
        if model_called and error is None:
            # 模型被调用且没有错误 → 检查回答是否包含拒答标记
            refusal_markers = ["没有找到", "相关度过低", "无法支持", "请输入"]
            if not any(m in answer for m in refusal_markers):
                failure_reasons.append(
                    "资料不足但仍调用了模型且未明确拒答"
                )

    # --- 来源命中 ---
    if case.expected_sources and not source_hit:
        failure_reasons.append(
            f"预期来源 {case.expected_sources} 未命中，"
            f"实际来源：{actual_sources}"
        )

    # --- 引用有效性 ---
    if not citation_valid:
        failure_reasons.append("回答中引用了不存在的 [S数字] 标签")

    # --- 关键词覆盖 ---
    if case.expected_keywords and keyword_coverage < 0.5:
        failure_reasons.append(
            f"关键词覆盖率 {keyword_coverage:.0%} < 50%，"
            f"期望：{case.expected_keywords}"
        )

    # --- 禁止词 ---
    if forbidden_hit:
        failure_reasons.append("回答中出现了禁止关键词")

    # --- 综合判定 ---
    if error:
        failure_reasons.append(f"模型调用异常：{error}")

    passed = len(failure_reasons) == 0

    from vibeflow.evaluation.models import CaseResult

    return CaseResult(
        case_id=case.id,
        question=case.question,
        passed=passed,
        should_answer=case.should_answer,
        actually_answered=actually_answered,
        expected_sources=case.expected_sources,
        expected_keywords=case.expected_keywords,
        actual_sources=actual_sources,
        source_hit=source_hit,
        citation_valid=citation_valid,
        keyword_coverage=keyword_coverage,
        forbidden_keyword_hit=forbidden_hit,
        model_called=model_called,
        latency_ms=latency_ms,
        error=error,
        failure_reasons=failure_reasons,
    )
