"""评估运行器。

加载评估案例、执行 RAGService 评估、生成结构化报告。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from vibeflow.evaluation.models import CaseResult, EvalReport, EvaluationCase
from vibeflow.evaluation.scorer import score_case
from vibeflow.knowledge_llm_client import LLMClient


class ScriptedFakeLLM(LLMClient):
    """按问题返回预设回答的 Fake LLM。

    用于评估模式：每个案例可指定期望的 fake_answer，
    使得引用有效性、关键词覆盖等检查可以被验证。

    对于未匹配的问题返回默认回答。
    """

    def __init__(
        self,
        answers: dict[str, str] | None = None,
        default_answer: str = "基于知识库的回答。",
    ) -> None:
        self._answers = answers or {}
        self._default = default_answer
        self.call_count = 0
        self.last_question = ""

    @property
    def model_name(self) -> str:
        return "fake-eval"

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.call_count += 1
        # 从 user message 中提取问题
        user_content = ""
        for m in messages:
            if m["role"] == "user":
                user_content = m["content"]
                break
        self.last_question = user_content

        # 尝试精确匹配问题
        for q, a in self._answers.items():
            if q in user_content:
                return a

        return self._default


class ErrorFakeLLM(LLMClient):
    """按概率抛出异常的 Fake LLM。

    用于压力测试场景四：模型错误处理。
    """

    def __init__(self, error_rate: float = 0.0, error_msg: str = "模拟的 LLM 错误") -> None:
        self.error_rate = error_rate
        self.error_msg = error_msg
        self.call_count = 0
        self._counter = 0

    @property
    def model_name(self) -> str:
        return "fake-error"

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.call_count += 1
        if self.error_rate <= 0:
            return "Fake 回答内容。"
        self._counter += 1
        if self._counter % max(1, int(1.0 / max(self.error_rate, 0.01))) == 0:
            raise RuntimeError(self.error_msg)
        return "Fake 回答内容。"


def load_cases(path: str | Path) -> list[EvaluationCase]:
    """从 JSON 文件加载评估案例。

    Args:
        path: rag_cases.json 文件路径。

    Returns:
        案例列表。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: JSON 格式或必填字段缺失。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"评估案例文件不存在：{path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError("评估案例文件格式错误：缺少顶层 'cases' 字段")

    cases: list[EvaluationCase] = []
    required_fields = {"id", "category", "question"}

    for i, item in enumerate(data["cases"]):
        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(
                f"案例 #{i} ({item.get('id', 'unknown')}) 缺少必填字段：{missing}"
            )
        cases.append(EvaluationCase.from_dict(item))

    return cases


def _percentile(values: list[float], pct: float) -> float:
    """计算百分位数（线性插值）。"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_vals):
        return sorted_vals[f] + c * (sorted_vals[f + 1] - sorted_vals[f])
    return sorted_vals[f]


def run_evaluation(
    cases: list[EvaluationCase],
    *,
    mode: str = "fake",
    knowledge_dir: str | Path = "knowledge",
    limit: int | None = None,
    category_filter: str | None = None,
    llm_client: LLMClient | None = None,
) -> EvalReport:
    """执行评估。

    Args:
        cases: 评估案例列表。
        mode: "fake" 或 "ollama"。
        knowledge_dir: 知识库目录路径。
        limit: 限制执行的案例数量（None = 全部）。
        category_filter: 仅执行指定 category 的案例（None = 全部）。
        llm_client: 外部注入的 LLMClient（用于测试）。Fake 模式下
                     如果未提供则自动创建 ScriptedFakeLLM。

    Returns:
        EvalReport 包含汇总和明细。
    """
    # 筛选案例
    selected = list(cases)
    if category_filter:
        selected = [c for c in selected if c.category == category_filter]
    if limit is not None:
        selected = selected[:limit]

    # 准备 Fake LLM（如需要）
    if mode == "fake" and llm_client is None:
        answers = {
            c.question: c.fake_answer
            for c in selected
            if c.should_answer and c.fake_answer
        }
        llm_client = ScriptedFakeLLM(answers=answers)

    if llm_client is None:
        # ollama 模式且未注入客户端
        from vibeflow.knowledge_llm_client import OllamaClient
        llm_client = OllamaClient()

    # 逐案例执行
    case_results: list[CaseResult] = []
    from vibeflow.knowledge_rag_service import RAGService
    from vibeflow.knowledge_service import KnowledgeService

    for case in selected:
        error: str | None = None
        answer = ""
        sources: list = []
        model_called = False
        start = time.perf_counter()

        try:
            service = KnowledgeService(
                str(knowledge_dir),
                mode=case.retrieval_mode,
            )
            rag = RAGService(service, llm_client)
            result = rag.ask(case.question)

            answer = result.answer
            sources = result.sources
            model_called = result.model_called
            error = result.error_message
        except Exception as e:
            error = str(e)

        latency_ms = (time.perf_counter() - start) * 1000

        case_results.append(
            score_case(
                case=case,
                answer=answer,
                sources=sources,
                model_called=model_called,
                latency_ms=latency_ms,
                error=error,
            )
        )

    # --- 汇总统计 ---
    total = len(case_results)
    passed = sum(1 for r in case_results if r.passed)
    failed = total - passed

    # 各维度统计
    # 来源命中率：只统计有 expected_sources 的案例
    cases_with_sources = [
        r for r in case_results if r.expected_sources
    ]
    source_hit_rate = (
        sum(1 for r in cases_with_sources if r.source_hit) / len(cases_with_sources)
        if cases_with_sources else 1.0
    )

    # 引用有效率
    citation_valid_rate = (
        sum(1 for r in case_results if r.citation_valid) / total
        if total else 1.0
    )

    # 拒答准确率：只统计 should_answer=False 的案例
    refusal_cases = [r for r in case_results if not r.should_answer]
    refusal_accuracy = (
        sum(1 for r in refusal_cases if r.passed) / len(refusal_cases)
        if refusal_cases else 1.0
    )

    # 关键词覆盖率：只统计有 expected_keywords 的案例
    kc_cases = [r for r in case_results if r.expected_sources or r.expected_keywords]
    avg_keyword_coverage = (
        sum(r.keyword_coverage for r in kc_cases) / len(kc_cases)
        if kc_cases else 1.0
    )

    # 模型调用错误率
    model_error_rate = (
        sum(1 for r in case_results if r.error is not None) / total
        if total else 0.0
    )

    # 延迟统计
    latencies = [r.latency_ms for r in case_results if r.latency_ms > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p50_latency = _percentile(latencies, 50)
    p95_latency = _percentile(latencies, 95)

    return EvalReport(
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=passed / total if total else 0.0,
        source_hit_rate=source_hit_rate,
        citation_valid_rate=citation_valid_rate,
        refusal_accuracy=refusal_accuracy,
        avg_keyword_coverage=avg_keyword_coverage,
        model_error_rate=model_error_rate,
        avg_latency_ms=avg_latency,
        p50_latency_ms=p50_latency,
        p95_latency_ms=p95_latency,
        mode=mode,
        case_results=case_results,
    )


def format_summary(report: EvalReport) -> str:
    """生成终端友好的摘要报告。"""
    lines = [
        "=" * 60,
        "        VibeFlow RAG 评估报告",
        "=" * 60,
        f"  模式：{report.mode}",
        f"  案例总数：{report.total}",
        f"  通过：{report.passed}  |  失败：{report.failed}  |  通过率：{report.pass_rate:.1%}",
        "",
        "  --- 质量指标 ---",
        f"  来源命中率：{report.source_hit_rate:.1%}",
        f"  引用有效率：{report.citation_valid_rate:.1%}",
        f"  拒答准确率：{report.refusal_accuracy:.1%}",
        f"  平均关键词覆盖率：{report.avg_keyword_coverage:.1%}",
        f"  模型调用错误率：{report.model_error_rate:.1%}",
        "",
        "  --- 性能 ---",
        f"  平均响应时间：{report.avg_latency_ms:.1f} ms",
        f"  P50 响应时间：{report.p50_latency_ms:.1f} ms",
        f"  P95 响应时间：{report.p95_latency_ms:.1f} ms",
        "",
    ]

    # 失败案例明细
    failed_cases = [r for r in report.case_results if not r.passed]
    if failed_cases:
        lines.append(f"  --- 失败案例 ({len(failed_cases)} 条) ---")
        for r in failed_cases:
            lines.append(f"  [{r.case_id}] {r.question[:40]}")
            for reason in r.failure_reasons:
                lines.append(f"    - {reason}")
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def save_report(report: EvalReport, path: str | Path) -> None:
    """将评估报告保存为 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
