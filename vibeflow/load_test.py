"""并发压力测试工具。

不依赖外部服务框架，使用标准库 concurrent.futures 实现。

用法：
    # Fake 模式（默认）
    python -m vibeflow.load_test

    # 自定义并发参数
    python -m vibeflow.load_test --concurrency 5 --requests 50

    # 设置单个请求超时
    python -m vibeflow.load_test --timeout 10

    # 本地 Ollama 模式（小规模）
    python -m vibeflow.load_test --mode ollama --requests 5 --concurrency 1

    # 模型错误场景
    python -m vibeflow.load_test --error-rate 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vibeflow.knowledge_llm_client import LLMClient


# ============================================================
# Fake LLM 变体（用于压力测试场景）
# ============================================================


class LoadTestFakeLLM(LLMClient):
    """压力测试用 FakeLLM：返回固定回答，模拟固定延迟。"""

    def __init__(self, delay_ms: float = 0) -> None:
        self.delay_ms = delay_ms
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "load-test-fake"

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.call_count += 1
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000)
        return "这是压力测试的回答。[S1]"


class ErrorRateFakeLLM(LLMClient):
    """按概率抛出异常的 FakeLLM。"""

    def __init__(self, error_rate: float = 0.0) -> None:
        self.error_rate = error_rate
        self.call_count = 0
        self._counter = 0

    @property
    def model_name(self) -> str:
        return "load-test-error-fake"

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.call_count += 1
        if self.error_rate <= 0:
            return "这是压力测试的回答。[S1]"
        self._counter += 1
        if self._counter % max(1, round(1.0 / max(self.error_rate, 0.01))) == 0:
            raise RuntimeError("模拟的 LLM 调用失败")
        return "这是压力测试的回答。[S1]"


# ============================================================
# 数据模型
# ============================================================


@dataclass
class RequestResult:
    """单个请求的结果。"""

    index: int
    success: bool
    latency_ms: float
    error: str | None = None
    model_called: bool = False


@dataclass
class LoadTestReport:
    """压力测试报告。"""

    total: int = 0
    successful: int = 0
    failed: int = 0
    timed_out: int = 0
    success_rate: float = 0.0
    total_duration_ms: float = 0.0
    throughput_rps: float = 0.0
    min_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    error_distribution: dict[str, int] = field(default_factory=dict)
    mode: str = "fake"
    concurrency: int = 0
    request_results: list[RequestResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total": self.total,
                "successful": self.successful,
                "failed": self.failed,
                "timed_out": self.timed_out,
                "success_rate": round(self.success_rate, 4),
                "total_duration_ms": round(self.total_duration_ms, 2),
                "throughput_rps": round(self.throughput_rps, 2),
                "min_latency_ms": round(self.min_latency_ms, 2),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "p50_latency_ms": round(self.p50_latency_ms, 2),
                "p95_latency_ms": round(self.p95_latency_ms, 2),
                "p99_latency_ms": round(self.p99_latency_ms, 2),
                "max_latency_ms": round(self.max_latency_ms, 2),
                "mode": self.mode,
                "concurrency": self.concurrency,
            },
            "error_distribution": self.error_distribution,
            "details": [
                {
                    "index": r.index,
                    "success": r.success,
                    "latency_ms": round(r.latency_ms, 2),
                    "error": r.error,
                    "model_called": r.model_called,
                }
                for r in self.request_results
            ],
        }


# ============================================================
# 核心逻辑
# ============================================================


def _percentile(values: list[float], pct: float) -> float:
    """计算百分位数。"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_vals):
        return sorted_vals[f] + c * (sorted_vals[f + 1] - sorted_vals[f])
    return sorted_vals[f]


# 默认问题集（基于真实知识库内容）
_DEFAULT_QUESTIONS = [
    "什么是Git提交？",
    "为什么要使用虚拟环境？",
    "Vibe Coding的黄金循环是什么？",
    "Git分支如何与代码审查配合？",
    "Claude Code需要哪些开发环境？",
    "Agent和Skills有什么区别？",
    "Hook工具调用拦截的作用是什么？",
    "Memory记忆管理的作用是什么？",
    "RAG项目实践包含哪些阶段？",
    "如何配置Python虚拟环境？",
    "Git版本控制的基本操作有哪些？",
    "什么是上下文压缩？",
    "单元测试在项目中的作用是什么？",
    "如何管理项目依赖？",
    "Vibe Coding中如何描述需求？",
]


def _execute_single_request(
    index: int,
    question: str,
    knowledge_dir: str,
    mode: str,
    timeout: float,
    llm_client: LLMClient | None,
) -> RequestResult:
    """执行单个请求（在独立线程中运行）。

    每个请求创建独立的 KnowledgeService 和 RAGService，
    避免状态串扰。
    """
    from vibeflow.knowledge_rag_service import RAGService
    from vibeflow.knowledge_service import KnowledgeService

    start = time.perf_counter()
    error: str | None = None
    model_called = False

    try:
        service = KnowledgeService(str(knowledge_dir), mode="keyword")
        rag = RAGService(service, llm_client)
        result = rag.ask(question)

        if result.error_message:
            error = result.error_message
        model_called = result.model_called
    except Exception as e:
        error = str(e)

    latency_ms = (time.perf_counter() - start) * 1000

    return RequestResult(
        index=index,
        success=error is None,
        latency_ms=latency_ms,
        error=error,
        model_called=model_called,
    )


def run_load_test(
    questions: list[str],
    *,
    mode: str = "fake",
    concurrency: int = 2,
    total_requests: int = 20,
    timeout: float = 30.0,
    knowledge_dir: str | Path = "knowledge",
    llm_client: LLMClient | None = None,
) -> LoadTestReport:
    """执行并发压力测试。

    Args:
        questions: 问题列表（循环使用）。
        mode: "fake" 或 "ollama"。
        concurrency: 并发线程数。
        total_requests: 总请求数。
        timeout: 单个请求超时秒数（Fake 模式忽略）。
        knowledge_dir: 知识库目录。
        llm_client: 外部注入的 LLMClient（用于测试）。

    Returns:
        LoadTestReport 包含统计信息。
    """
    # 准备 LLM 客户端
    if llm_client is None:
        if mode == "fake":
            llm_client = LoadTestFakeLLM()
        else:
            from vibeflow.knowledge_llm_client import OllamaClient
            llm_client = OllamaClient()

    # 构建请求队列
    request_args = [
        (
            i,
            questions[i % len(questions)],
            str(knowledge_dir),
            mode,
            timeout,
            llm_client,
        )
        for i in range(total_requests)
    ]

    results: list[RequestResult] = []
    start_wall = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_execute_single_request, *args): args[0]
            for args in request_args
        }
        for future in as_completed(futures):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception:
                idx = futures[future]
                results.append(
                    RequestResult(
                        index=idx,
                        success=False,
                        latency_ms=timeout * 1000,
                        error="请求超时",
                    )
                )

    total_duration_ms = (time.perf_counter() - start_wall) * 1000

    # 按 index 排序
    results.sort(key=lambda r: r.index)

    # 统计
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    timed_out = sum(1 for r in results if r.error == "请求超时")

    success_latencies = [r.latency_ms for r in results if r.success]
    all_latencies = [r.latency_ms for r in results]

    # 错误分布
    error_dist: dict[str, int] = {}
    for r in results:
        if r.error:
            key = r.error[:50] if len(r.error) > 50 else r.error
            error_dist[key] = error_dist.get(key, 0) + 1

    return LoadTestReport(
        total=total,
        successful=successful,
        failed=failed,
        timed_out=timed_out,
        success_rate=successful / total if total else 0.0,
        total_duration_ms=total_duration_ms,
        throughput_rps=total / (total_duration_ms / 1000) if total_duration_ms > 0 else 0.0,
        min_latency_ms=min(all_latencies) if all_latencies else 0.0,
        avg_latency_ms=sum(all_latencies) / len(all_latencies) if all_latencies else 0.0,
        p50_latency_ms=_percentile(all_latencies, 50),
        p95_latency_ms=_percentile(all_latencies, 95),
        p99_latency_ms=_percentile(all_latencies, 99),
        max_latency_ms=max(all_latencies) if all_latencies else 0.0,
        error_distribution=error_dist,
        mode=mode,
        concurrency=concurrency,
        request_results=results,
    )


def format_load_test_summary(report: LoadTestReport) -> str:
    """生成终端友好的压力测试摘要。"""
    lines = [
        "=" * 60,
        "        VibeFlow RAG 并发压力测试报告",
        "=" * 60,
        f"  模式：{report.mode}  |  并发数：{report.concurrency}",
        f"  总请求：{report.total}  |  成功：{report.successful}  |  "
        f"失败：{report.failed}  |  超时：{report.timed_out}",
        f"  成功率：{report.success_rate:.1%}",
        f"  总耗时：{report.total_duration_ms:.0f} ms",
        f"  吞吐量：{report.throughput_rps:.1f} 请求/秒",
        "",
        "  --- 响应时间 ---",
        f"  最小：{report.min_latency_ms:.1f} ms",
        f"  平均：{report.avg_latency_ms:.1f} ms",
        f"  P50：{report.p50_latency_ms:.1f} ms",
        f"  P95：{report.p95_latency_ms:.1f} ms",
        f"  P99：{report.p99_latency_ms:.1f} ms",
        f"  最大：{report.max_latency_ms:.1f} ms",
        "",
    ]

    if report.error_distribution:
        lines.append("  --- 异常类型分布 ---")
        for err_msg, count in sorted(
            report.error_distribution.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  [{count}次] {err_msg}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def save_load_test_report(report: LoadTestReport, path: str | Path) -> None:
    """保存压力测试报告为 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)


# ============================================================
# CLI 入口
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VibeFlow RAG 并发压力测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m vibeflow.load_test                                   # 默认配置\n"
            "  python -m vibeflow.load_test --concurrency 5 --requests 50     # 高并发\n"
            "  python -m vibeflow.load_test --mode ollama --requests 5        # Ollama 小规模\n"
            "  python -m vibeflow.load_test --error-rate 0.3                  # 30% 错误率\n"
            "  python -m vibeflow.load_test --output load_results/result.json # 保存报告\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["fake", "ollama"],
        default="fake",
        help="测试模式：fake（默认）/ ollama",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="并发线程数（默认：2）",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=20,
        help="总请求数（默认：20）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="单请求超时秒数（默认：30）",
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.0,
        help="Fake LLM 的异常率 0.0-1.0（默认：0）",
    )
    parser.add_argument(
        "--knowledge-dir",
        default="knowledge",
        help="知识库目录（默认：knowledge）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="保存 JSON 报告路径（如 load_results/result.json）",
    )
    args = parser.parse_args()

    # 显示模式
    if args.mode == "ollama":
        print("[!] 本地 Ollama 模式，将连接真实模型")
        print(f"  并发数={args.concurrency}，总请求={args.requests}")
        print()
    else:
        label = f"Fake LLM 模式（不连接真实模型），异常率={args.error_rate}"
        print(f"评估模式：{label}")
        print(f"  并发数={args.concurrency}，总请求={args.requests}")
        print()

    # 准备 LLM 客户端
    llm_client: LLMClient | None = None
    if args.mode == "fake":
        if args.error_rate > 0:
            llm_client = ErrorRateFakeLLM(error_rate=args.error_rate)
        else:
            llm_client = LoadTestFakeLLM()

    # 执行
    questions = _DEFAULT_QUESTIONS
    report = run_load_test(
        questions=questions,
        mode=args.mode,
        concurrency=args.concurrency,
        total_requests=args.requests,
        timeout=args.timeout,
        knowledge_dir=args.knowledge_dir,
        llm_client=llm_client,
    )

    print(format_load_test_summary(report))

    if args.output:
        save_load_test_report(report, args.output)
        print(f"完整报告已保存至：{args.output}")


if __name__ == "__main__":
    main()
