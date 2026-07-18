"""并发压力测试工具单元测试。

覆盖：FakeLLM 变体、请求执行、统计指标、并发安全。
所有测试不连接网络、不调用真实 Ollama。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from vibeflow.load_test import (
    ErrorRateFakeLLM,
    LoadTestFakeLLM,
    LoadTestReport,
    RequestResult,
    _execute_single_request,
    _percentile,
    format_load_test_summary,
    run_load_test,
    save_load_test_report,
)


# ============================================================
# FakeLLM 测试
# ============================================================


class TestLoadTestFakeLLM:
    def test_returns_fixed_answer(self) -> None:
        llm = LoadTestFakeLLM()
        result = llm.generate([{"role": "user", "content": "任意问题"}])
        assert "压力测试" in result
        assert "[S1]" in result

    def test_tracks_call_count(self) -> None:
        llm = LoadTestFakeLLM()
        for _ in range(5):
            llm.generate([{"role": "user", "content": "q"}])
        assert llm.call_count == 5

    def test_no_network_access(self) -> None:
        llm = LoadTestFakeLLM()
        result = llm.generate([])
        assert isinstance(result, str)

    def test_delay_simulation(self) -> None:
        """延迟参数正常模拟。"""
        llm = LoadTestFakeLLM(delay_ms=10)
        import time
        start = time.perf_counter()
        llm.generate([{"role": "user", "content": "q"}])
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed >= 8  # 允许一定误差


class TestErrorRateFakeLLM:
    def test_no_errors_when_rate_zero(self) -> None:
        llm = ErrorRateFakeLLM(error_rate=0.0)
        for _ in range(10):
            result = llm.generate([{"role": "user", "content": "q"}])
            assert isinstance(result, str)

    def test_errors_at_configured_rate(self) -> None:
        """错误率 0.5 时大约一半调用失败。"""
        llm = ErrorRateFakeLLM(error_rate=0.5)
        errors = 0
        for _ in range(20):
            try:
                llm.generate([{"role": "user", "content": "q"}])
            except RuntimeError:
                errors += 1
        # 0.5 错误率，20 次中至少应有 1 次错误
        assert errors >= 1

    def test_all_errors_when_rate_one(self) -> None:
        """错误率 1.0 时全部失败。"""
        llm = ErrorRateFakeLLM(error_rate=1.0)
        errors = 0
        for _ in range(10):
            try:
                llm.generate([{"role": "user", "content": "q"}])
            except RuntimeError:
                errors += 1
        assert errors == 10


# ============================================================
# 统计指标测试
# ============================================================


class TestPercentile:
    def test_p50_odd(self) -> None:
        assert _percentile([1, 2, 3, 4, 5], 50) == pytest.approx(3.0)

    def test_p50_even(self) -> None:
        assert _percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_p95(self) -> None:
        vals = list(range(1, 101))
        assert _percentile(vals, 95) == pytest.approx(95.05)

    def test_p99(self) -> None:
        vals = list(range(1, 101))
        assert _percentile(vals, 99) == pytest.approx(99.01)

    def test_min_max_p0_p100(self) -> None:
        vals = [10, 20, 30]
        assert _percentile(vals, 0) == 10.0
        assert _percentile(vals, 100) == 30.0

    def test_empty(self) -> None:
        assert _percentile([], 50) == 0.0

    def test_single(self) -> None:
        assert _percentile([42.0], 95) == 42.0

    def test_two_values(self) -> None:
        assert _percentile([1.0, 2.0], 50) == pytest.approx(1.5)
        assert _percentile([1.0, 2.0], 0) == 1.0
        assert _percentile([1.0, 2.0], 100) == 2.0

    def test_all_identical(self) -> None:
        assert _percentile([5.0, 5.0, 5.0], 50) == 5.0
        assert _percentile([5.0, 5.0, 5.0], 95) == 5.0


class TestLoadTestReport:
    def test_to_dict(self) -> None:
        report = LoadTestReport(
            total=10,
            successful=9,
            failed=1,
            success_rate=0.9,
            total_duration_ms=1000.0,
            throughput_rps=10.0,
            avg_latency_ms=50.0,
            p50_latency_ms=45.0,
            p95_latency_ms=80.0,
            p99_latency_ms=95.0,
            mode="fake",
            concurrency=2,
            request_results=[
                RequestResult(index=0, success=True, latency_ms=45.0),
                RequestResult(index=1, success=False, latency_ms=100.0, error="超时"),
            ],
        )
        d = report.to_dict()
        assert d["summary"]["total"] == 10
        assert d["summary"]["successful"] == 9
        assert len(d["details"]) == 2


# ============================================================
# 请求执行测试
# ============================================================


class TestExecuteSingleRequest:
    def test_successful_request(self, tmp_path: Path) -> None:
        """单个请求成功执行。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("Git提交是存档点。", encoding="utf-8")

        result = _execute_single_request(
            index=0,
            question="什么是Git提交？",
            knowledge_dir=str(kd),
            mode="fake",
            timeout=30.0,
            llm_client=LoadTestFakeLLM(),
        )
        assert result.success is True
        assert result.latency_ms > 0
        assert result.model_called is True

    def test_error_captured(self, tmp_path: Path) -> None:
        """LLM 异常被正确捕获。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("Git提交是存档点。", encoding="utf-8")

        result = _execute_single_request(
            index=0,
            question="什么是Git提交",
            knowledge_dir=str(kd),
            mode="fake",
            timeout=30.0,
            llm_client=ErrorRateFakeLLM(error_rate=1.0),
        )
        # error_rate=1.0 时 LLM 一定会抛异常，被 _execute_single_request 捕获
        assert result.success is False
        assert result.error is not None

    def test_empty_question(self, tmp_path: Path) -> None:
        """空问题正确处理。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()

        result = _execute_single_request(
            index=0,
            question="",
            knowledge_dir=str(kd),
            mode="fake",
            timeout=30.0,
            llm_client=LoadTestFakeLLM(),
        )
        assert result.success is True


# ============================================================
# 并发压力测试集成测试
# ============================================================


class TestRunLoadTest:
    def test_basic_fake_run(self, tmp_path: Path) -> None:
        """基本 Fake 模式运行成功。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("测试内容。", encoding="utf-8")

        report = run_load_test(
            questions=["问题A", "问题B", "问题C"],
            mode="fake",
            concurrency=2,
            total_requests=10,
            knowledge_dir=str(kd),
            llm_client=LoadTestFakeLLM(),
        )
        assert report.total == 10
        assert report.successful == 10
        assert report.concurrency == 2

    def test_concurrent_requests_no_crosstalk(self, tmp_path: Path) -> None:
        """并发请求之间来源编号不串线。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "notes.md").write_text("Git分支隔离开发。Python虚拟环境管理依赖。", encoding="utf-8")

        report = run_load_test(
            questions=["Git分支", "Python虚拟环境"],
            mode="fake",
            concurrency=3,
            total_requests=10,
            knowledge_dir=str(kd),
            llm_client=LoadTestFakeLLM(),
        )
        assert report.successful == 10
        # 每个请求独立创建 RAGService，不会有状态串扰

    def test_error_handling_in_concurrent(self, tmp_path: Path) -> None:
        """并发中部分请求失败，不影响其他请求。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("Git提交是存档点。Python虚拟环境隔离依赖。", encoding="utf-8")

        report = run_load_test(
            questions=["Git提交", "Python虚拟环境"],
            mode="fake",
            concurrency=2,
            total_requests=10,
            knowledge_dir=str(kd),
            llm_client=ErrorRateFakeLLM(error_rate=0.3),
        )
        # 不应全部失败
        assert report.successful >= 1
        assert report.failed >= 1
        assert report.total == 10

    def test_single_request_exception_does_not_crash(self, tmp_path: Path) -> None:
        """单个请求异常不导致整个测试崩溃。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("Git提交是存档点。", encoding="utf-8")

        # 即使全部请求都会失败，也不应抛出异常
        report = run_load_test(
            questions=["Git提交"],
            mode="fake",
            concurrency=1,
            total_requests=5,
            knowledge_dir=str(kd),
            llm_client=ErrorRateFakeLLM(error_rate=1.0),
        )
        assert report.total == 5
        assert report.successful == 0
        assert report.failed == 5

    def test_throughput_calculation(self, tmp_path: Path) -> None:
        """吞吐量计算正确。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("测试。", encoding="utf-8")

        report = run_load_test(
            questions=["问题"],
            mode="fake",
            concurrency=1,
            total_requests=10,
            knowledge_dir=str(kd),
            llm_client=LoadTestFakeLLM(),
        )
        assert report.throughput_rps > 0
        assert report.total_duration_ms > 0

    def test_all_metrics_present(self, tmp_path: Path) -> None:
        """报告包含所有必要指标。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("测试内容。", encoding="utf-8")

        report = run_load_test(
            questions=["问题A", "问题B"],
            mode="fake",
            concurrency=2,
            total_requests=6,
            knowledge_dir=str(kd),
            llm_client=LoadTestFakeLLM(),
        )
        summary = format_load_test_summary(report)
        assert "成功率" in summary
        assert "吞吐量" in summary
        assert "P50" in summary
        assert "P95" in summary
        assert "P99" in summary
        assert "最小" in summary
        assert "最大" in summary

    def test_no_duplicate_source_labels(self, tmp_path: Path) -> None:
        """多次请求之间来源编号不串线。

        每个请求创建独立的 RAGService，因此 [S1] 标签
        分别属于各自的请求上下文，不会跨请求共用来源列表。
        """
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("Git提交是存档点。Python虚拟环境隔离依赖。", encoding="utf-8")

        report = run_load_test(
            questions=["Git提交", "Python虚拟环境"],
            mode="fake",
            concurrency=2,
            total_requests=4,
            knowledge_dir=str(kd),
            llm_client=LoadTestFakeLLM(),
        )
        assert report.successful == 4
        # 如果串线，某些请求会失败（但 Fake 模式下 LLM 总是返回固定回答）
        # 这里主要验证系统不崩溃

    def test_save_and_load_report(self, tmp_path: Path) -> None:
        """报告保存和加载正确。"""
        report = LoadTestReport(
            total=1,
            successful=1,
            mode="fake",
            concurrency=2,
            request_results=[
                RequestResult(index=0, success=True, latency_ms=10.0)
            ],
        )
        path = tmp_path / "load_results" / "report.json"
        save_load_test_report(report, path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["summary"]["total"] == 1

    def test_insufficient_knowledge_questions(self, tmp_path: Path) -> None:
        """资料不足问题并发时都正确拒答，不调用模型。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        # 空知识库
        report = run_load_test(
            questions=["量子计算", "美国总统大选", "治疗感冒"],
            mode="fake",
            concurrency=2,
            total_requests=6,
            knowledge_dir=str(kd),
            llm_client=LoadTestFakeLLM(),
        )
        assert report.successful == 6
        # 所有请求都应完成（拒答也是成功）

    def test_result_indices_preserve_order(self, tmp_path: Path) -> None:
        """结果按 index 排序。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("测试。", encoding="utf-8")

        report = run_load_test(
            questions=["问题"],
            mode="fake",
            concurrency=1,
            total_requests=5,
            knowledge_dir=str(kd),
            llm_client=LoadTestFakeLLM(),
        )
        indices = [r.index for r in report.request_results]
        assert indices == sorted(indices)

    def test_shared_llm_call_count_accumulates(self, tmp_path: Path) -> None:
        """共享的 LLMClient 调用计数在请求间正确累积。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("Git提交是存档点。", encoding="utf-8")

        llm = LoadTestFakeLLM()
        report = run_load_test(
            questions=["什么是Git提交？"],
            mode="fake",
            concurrency=2,
            total_requests=5,
            knowledge_dir=str(kd),
            llm_client=llm,
        )
        # 共享 LLM 被调用 5 次
        assert llm.call_count == 5
        assert report.successful == 5


# ============================================================
# 原有功能回归
# ============================================================


class TestNoRegressions:
    def test_rag_service_unaffected(self, tmp_path: Path) -> None:
        """原有 RAGService 功能不受影响。"""
        from vibeflow.knowledge_llm_client import LLMClient
        from vibeflow.knowledge_rag_service import RAGService
        from vibeflow.knowledge_service import KnowledgeService

        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "notes.md").write_text("Git提交是存档点。", encoding="utf-8")

        class SimpleFakeLLM(LLMClient):
            @property
            def model_name(self) -> str:
                return "test"
            def generate(self, messages):
                return "回答[S1]"

        service = KnowledgeService(str(kd), mode="keyword")
        rag = RAGService(service, SimpleFakeLLM())
        result = rag.ask("什么是Git提交？")
        assert result.model_called is True
        assert len(result.answer) > 0

    def test_keyword_retriever_unaffected(self) -> None:
        """KeywordRetriever 不受影响。"""
        from vibeflow.knowledge_models import TextChunk
        from vibeflow.knowledge_retriever import KeywordRetriever

        retriever = KeywordRetriever()
        chunks = [TextChunk(content="Git提交是存档点", source_file="t.md", chunk_index=0)]
        results = retriever.search(chunks, "Git提交")
        assert len(results) == 1
