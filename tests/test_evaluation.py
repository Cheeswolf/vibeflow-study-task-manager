"""评估体系单元测试。

覆盖：数据加载、各评分规则、案例评分、汇总统计、FakeLLM 评估运行。
所有测试不连接网络、不调用真实 Ollama。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from vibeflow.evaluation.models import CaseResult, EvalReport, EvaluationCase
from vibeflow.evaluation.runner import (
    ErrorFakeLLM,
    ScriptedFakeLLM,
    _percentile,
    format_summary,
    load_cases,
    run_evaluation,
    save_report,
)
from vibeflow.evaluation.scorer import (
    check_citation_valid,
    check_forbidden,
    check_keyword_coverage,
    check_source_hit,
    score_case,
)
from vibeflow.knowledge_rag_models import SourceInfo


# ============================================================
# 测试辅助
# ============================================================


def _make_sources(
    filenames: list[str],
    scores: list[float] | None = None,
) -> list[SourceInfo]:
    if scores is None:
        scores = [5.0] * len(filenames)
    return [
        SourceInfo(
            reference_label=f"[S{i + 1}]",
            source_file=fn,
            chunk_index=i,
            score=scores[i],
            content_snippet=f"content from {fn}",
        )
        for i, fn in enumerate(filenames)
    ]


def _make_case(**overrides) -> EvaluationCase:
    defaults = {
        "id": "test-001",
        "category": "测试",
        "question": "测试问题？",
        "should_answer": True,
        "expected_sources": ["test.md"],
        "expected_keywords": ["测试", "关键词"],
        "forbidden_keywords": [],
        "retrieval_mode": "keyword",
        "require_all_sources": False,
        "description": "",
        "fake_answer": "这是测试回答[S1]，包含关键词。",
    }
    defaults.update(overrides)
    return EvaluationCase(**defaults)


# ============================================================
# 数据加载测试
# ============================================================


class TestLoadCases:
    def test_load_valid_file(self, tmp_path: Path) -> None:
        """正确加载格式合法的评估案例文件。"""
        cases_json = {
            "cases": [
                {
                    "id": "c1",
                    "category": "精确关键词",
                    "question": "测试问题",
                    "should_answer": True,
                }
            ]
        }
        path = tmp_path / "test_cases.json"
        path.write_text(json.dumps(cases_json, ensure_ascii=False), encoding="utf-8")
        cases = load_cases(path)
        assert len(cases) == 1
        assert cases[0].id == "c1"
        assert cases[0].question == "测试问题"

    def test_load_with_all_fields(self, tmp_path: Path) -> None:
        """所有可选字段都被正确加载。"""
        cases_json = {
            "cases": [
                {
                    "id": "full",
                    "category": "完整案例",
                    "question": "完整测试？",
                    "should_answer": True,
                    "expected_sources": ["a.md", "b.md"],
                    "expected_keywords": ["kw1", "kw2"],
                    "forbidden_keywords": ["bad"],
                    "retrieval_mode": "hybrid",
                    "require_all_sources": True,
                    "description": "完整字段测试",
                    "fake_answer": "预设回答[S1]",
                }
            ]
        }
        path = tmp_path / "full_cases.json"
        path.write_text(json.dumps(cases_json, ensure_ascii=False), encoding="utf-8")
        cases = load_cases(path)
        c = cases[0]
        assert c.expected_sources == ["a.md", "b.md"]
        assert c.expected_keywords == ["kw1", "kw2"]
        assert c.forbidden_keywords == ["bad"]
        assert c.retrieval_mode == "hybrid"
        assert c.require_all_sources is True
        assert c.description == "完整字段测试"
        assert c.fake_answer == "预设回答[S1]"

    def test_file_not_found(self) -> None:
        """文件不存在时抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_cases("nonexistent_file.json")

    def test_missing_cases_field(self, tmp_path: Path) -> None:
        """缺少顶层 cases 字段时抛出 ValueError。"""
        path = tmp_path / "bad.json"
        path.write_text('{"not_cases": []}', encoding="utf-8")
        with pytest.raises(ValueError, match="cases"):
            load_cases(path)

    def test_missing_required_id(self, tmp_path: Path) -> None:
        """案例缺少必填 id 字段时抛出 ValueError。"""
        cases_json = {"cases": [{"category": "x", "question": "q"}]}
        path = tmp_path / "missing_id.json"
        path.write_text(json.dumps(cases_json), encoding="utf-8")
        with pytest.raises(ValueError, match="id"):
            load_cases(path)

    def test_missing_required_question(self, tmp_path: Path) -> None:
        """案例缺少必填 question 字段时抛出 ValueError。"""
        cases_json = {"cases": [{"id": "x", "category": "x"}]}
        path = tmp_path / "missing_q.json"
        path.write_text(json.dumps(cases_json), encoding="utf-8")
        with pytest.raises(ValueError, match="question"):
            load_cases(path)

    def test_load_18_default_cases(self) -> None:
        """默认评估集至少包含 15 条案例。"""
        import os
        cases_path = Path(__file__).resolve().parent.parent / "evaluation" / "rag_cases.json"
        if not cases_path.exists():
            pytest.skip("默认评估集文件不存在")
        cases = load_cases(cases_path)
        assert len(cases) >= 15, f"至少需要 15 条案例，实际 {len(cases)}"


# ============================================================
# 评分规则测试
# ============================================================


class TestSourceHit:
    def test_at_least_one_match(self) -> None:
        """命中任意一个来源。"""
        assert check_source_hit(["a.md"], ["a.md", "b.md"]) is True

    def test_no_match(self) -> None:
        """全部未命中。"""
        assert check_source_hit(["x.md"], ["a.md", "b.md"]) is False

    def test_partial_match_substring(self) -> None:
        """来源文件名包含匹配。"""
        assert check_source_hit(["notes"], ["path/notes/git.md"]) is True

    def test_require_all_true(self) -> None:
        """要求全部命中 — 全部命中。"""
        assert (
            check_source_hit(
                ["a.md", "b.md"], ["a.md", "b.md", "c.md"], require_all=True
            )
            is True
        )

    def test_require_all_false_partial(self) -> None:
        """要求全部命中 — 部分命中失败。"""
        assert (
            check_source_hit(
                ["a.md", "x.md"], ["a.md", "b.md"], require_all=True
            )
            is False
        )

    def test_empty_expected_is_always_true(self) -> None:
        """没有设定预期来源，总是通过。"""
        assert check_source_hit([], ["a.md"]) is True


class TestCitationValid:
    def test_all_citations_in_range(self) -> None:
        """所有引用在有效范围内。"""
        sources = _make_sources(["a.md", "b.md", "c.md"])
        assert check_citation_valid("参见[S1]和[S3]。", sources) is True

    def test_citation_out_of_range(self) -> None:
        """引用不存在的标签号。"""
        sources = _make_sources(["a.md"])
        assert check_citation_valid("参见[S5]。", sources) is False

    def test_citation_zero_invalid(self) -> None:
        """[S0] 是无效引用。"""
        sources = _make_sources(["a.md"])
        assert check_citation_valid("[S0] 是无效的。", sources) is False

    def test_no_citations_is_valid(self) -> None:
        """没有引用标签时视为有效。"""
        sources = _make_sources(["a.md"])
        assert check_citation_valid("没有引用的回答。", sources) is True

    def test_empty_answer_is_valid(self) -> None:
        """空回答视为引用有效。"""
        assert check_citation_valid("", []) is True

    def test_citations_with_empty_sources(self) -> None:
        """有引用但没有来源时，任何引用都无效。"""
        assert check_citation_valid("[S1] 引用了不存在的内容。", []) is False

    def test_mixed_valid_invalid_citation_fails(self) -> None:
        """同时包含有效[S1]和无效[S5]引用时失败。"""
        sources = _make_sources(["a.md", "b.md"])
        assert check_citation_valid("参见[S1]和[S5]。", sources) is False

    def test_citation_with_space_not_matched(self) -> None:
        """[S 1]中间含空格不被识别为引用，没有有效引用视为通过。"""
        sources = _make_sources(["a.md"])
        assert check_citation_valid("参见[S 1]。", sources) is True

    def test_large_citation_number_out_of_range(self) -> None:
        """极大的引用号超出范围。"""
        sources = _make_sources(["a.md"])
        assert check_citation_valid("参见[S99999]。", sources) is False


class TestKeywordCoverage:
    def test_all_matched(self) -> None:
        assert check_keyword_coverage("关键词A和关键词B", ["关键词A", "关键词B"]) == 1.0

    def test_partial_match(self) -> None:
        assert check_keyword_coverage("只有关键词A", ["关键词A", "关键词B"]) == 0.5

    def test_none_matched(self) -> None:
        assert check_keyword_coverage("无关内容", ["关键词A", "关键词B"]) == 0.0

    def test_case_insensitive(self) -> None:
        """大小写不敏感。"""
        assert check_keyword_coverage("Hello World", ["hello"]) == 1.0

    def test_empty_keywords_always_full(self) -> None:
        """没有设定关键词，返回满分。"""
        assert check_keyword_coverage("任何内容", []) == 1.0

    def test_empty_answer_zero_coverage(self) -> None:
        """空回答覆盖率为 0。"""
        assert check_keyword_coverage("", ["关键词"]) == 0.0


class TestForbidden:
    def test_no_forbidden_hit(self) -> None:
        assert check_forbidden("正常内容", ["禁止词"]) is False

    def test_forbidden_hit(self) -> None:
        assert check_forbidden("包含禁止词的内容", ["禁止词"]) is True

    def test_case_insensitive(self) -> None:
        assert check_forbidden("BAD WORD here", ["bad word"]) is True

    def test_empty_forbidden_list(self) -> None:
        assert check_forbidden("任何内容", []) is False


# ============================================================
# 拒答评分精细化测试
# ============================================================


class TestRefusalScoring:
    def test_marker_meiyouzhaodao(self) -> None:
        """拒答标记'没有找到'命中则通过。"""
        case = _make_case(should_answer=False, expected_sources=[], expected_keywords=[])
        result = score_case(
            case=case,
            answer="没有找到相关资料。",
            sources=[],
            model_called=True,
            latency_ms=10.0,
            error=None,
        )
        assert result.passed is True

    def test_marker_xiangguandugodi(self) -> None:
        """拒答标记'相关度过低'命中则通过。"""
        case = _make_case(should_answer=False, expected_sources=[], expected_keywords=[])
        result = score_case(
            case=case,
            answer="检索结果相关度过低，无法回答。",
            sources=[],
            model_called=True,
            latency_ms=10.0,
            error=None,
        )
        assert result.passed is True

    def test_marker_wufazhichi(self) -> None:
        """拒答标记'无法支持'命中则通过。"""
        case = _make_case(should_answer=False, expected_sources=[], expected_keywords=[])
        result = score_case(
            case=case,
            answer="知识库内容无法支持回答此问题。",
            sources=[],
            model_called=True,
            latency_ms=10.0,
            error=None,
        )
        assert result.passed is True

    def test_marker_qingshuru(self) -> None:
        """拒答标记'请输入'命中则通过。"""
        case = _make_case(should_answer=False, expected_sources=[], expected_keywords=[])
        result = score_case(
            case=case,
            answer="请输入有效的问题。",
            sources=[],
            model_called=True,
            latency_ms=10.0,
            error=None,
        )
        assert result.passed is True

    def test_should_not_answer_model_not_called_passes(self) -> None:
        """不应回答且模型未被调用，通过。"""
        case = _make_case(should_answer=False, expected_sources=[], expected_keywords=[])
        result = score_case(
            case=case,
            answer="",
            sources=[],
            model_called=False,
            latency_ms=5.0,
            error=None,
        )
        assert result.passed is True

    def test_should_not_answer_with_error_fails(self) -> None:
        """不应回答但模型调用出错，因有错误而失败。"""
        case = _make_case(should_answer=False, expected_sources=[], expected_keywords=[])
        result = score_case(
            case=case,
            answer="",
            sources=[],
            model_called=True,
            latency_ms=10.0,
            error="连接超时",
        )
        assert result.passed is False
        assert result.error == "连接超时"


# ============================================================
# 单案例评分测试
# ============================================================


class TestScoreCase:
    def test_perfect_pass(self) -> None:
        """所有条件满足，通过。"""
        case = _make_case()
        sources = _make_sources(["test.md"])
        result = score_case(
            case=case,
            answer="这是测试回答[S1]，包含关键词。",
            sources=sources,
            model_called=True,
            latency_ms=50.0,
            error=None,
        )
        assert result.passed is True
        assert result.source_hit is True
        assert result.citation_valid is True
        assert result.keyword_coverage == 1.0
        assert result.forbidden_keyword_hit is False

    def test_source_miss_fails(self) -> None:
        """预期来源未命中。"""
        case = _make_case(expected_sources=["missing.md"])
        sources = _make_sources(["other.md"])
        result = score_case(
            case=case,
            answer="回答[S1]",
            sources=sources,
            model_called=True,
            latency_ms=50.0,
            error=None,
        )
        assert result.passed is False
        assert result.source_hit is False
        assert "预期来源" in result.failure_reasons[0]

    def test_citation_invalid_fails(self) -> None:
        """引用不存在的来源。"""
        case = _make_case(expected_sources=[])
        sources = _make_sources(["test.md"])  # 只有 1 个来源
        result = score_case(
            case=case,
            answer="参见[S8]了解更多。",
            sources=sources,
            model_called=True,
            latency_ms=50.0,
            error=None,
        )
        assert result.passed is False
        assert result.citation_valid is False

    def test_keyword_insufficient_fails(self) -> None:
        """关键词覆盖率不足。"""
        case = _make_case(
            expected_keywords=["关键词A", "关键词B", "关键词C", "关键词D"]
        )
        sources = _make_sources(["test.md"])
        result = score_case(
            case=case,
            answer="只有关键词A。",  # 1/4 = 25% < 50%
            sources=sources,
            model_called=True,
            latency_ms=50.0,
            error=None,
        )
        assert result.passed is False
        assert "关键词覆盖率" in result.failure_reasons[0]

    def test_forbidden_hit_fails(self) -> None:
        """禁止词命中。"""
        case = _make_case(
            expected_keywords=["测试"],
            forbidden_keywords=["删除所有文件"],
        )
        sources = _make_sources(["test.md"])
        result = score_case(
            case=case,
            answer="我已删除所有文件。[S1]",
            sources=sources,
            model_called=True,
            latency_ms=50.0,
            error=None,
        )
        assert result.passed is False
        assert result.forbidden_keyword_hit is True
        assert any("禁止" in r for r in result.failure_reasons)

    def test_should_not_answer_but_model_called(self) -> None:
        """资料不足案例，如果模型被调用且未明确拒答，判定失败。"""
        case = _make_case(should_answer=False, expected_sources=[], expected_keywords=[])
        sources = _make_sources(["irrelevant.md"])
        result = score_case(
            case=case,
            answer="根据我的知识，量子计算可以用于...",  # 没有拒答标记
            sources=sources,
            model_called=True,
            latency_ms=50.0,
            error=None,
        )
        assert result.passed is False
        assert any("资料不足" in r for r in result.failure_reasons)

    def test_should_not_answer_refused_passes(self) -> None:
        """资料不足案例，系统正确拒答，通过。"""
        case = _make_case(should_answer=False, expected_sources=[], expected_keywords=[])
        result = score_case(
            case=case,
            answer="当前知识库中没有找到足够相关的资料。",
            sources=[],
            model_called=False,
            latency_ms=5.0,
            error=None,
        )
        assert result.passed is True

    def test_error_recorded(self) -> None:
        """异常被记录到结果中。"""
        case = _make_case()
        result = score_case(
            case=case,
            answer="",
            sources=[],
            model_called=False,
            latency_ms=100.0,
            error="Ollama 连接失败",
        )
        assert result.passed is False
        assert result.error == "Ollama 连接失败"

    def test_multiple_failure_reasons(self) -> None:
        """多个失败原因同时记录。"""
        case = _make_case(
            expected_sources=["a.md"],
            expected_keywords=["关键词"],
            forbidden_keywords=["禁止"],
        )
        sources = _make_sources(["other.md"])
        result = score_case(
            case=case,
            answer="包含禁止词但没关键词[S8]",
            sources=sources,
            model_called=True,
            latency_ms=50.0,
            error=None,
        )
        assert result.passed is False
        assert len(result.failure_reasons) >= 3

    def test_keyword_coverage_at_boundary(self) -> None:
        """关键词覆盖率恰等于 50% 时通过（>= 0.5）。"""
        case = _make_case(expected_keywords=["版本控制", "虚拟环境"])
        sources = _make_sources(["test.md"])
        result = score_case(
            case=case,
            answer="Git是版本控制工具。[S1]",
            sources=sources,
            model_called=True,
            latency_ms=50.0,
            error=None,
        )
        assert result.keyword_coverage == 0.5
        # 由于 citation_valid 和 source_hit 都 OK，应通过
        assert result.passed is True

    def test_single_case_failure_does_not_raise(self) -> None:
        """单条失败不影响流程，返回结果而非抛出异常。"""
        case = _make_case(expected_sources=["nonexistent.md"])
        sources = _make_sources(["real.md"])
        # 不应抛出异常
        result = score_case(
            case=case,
            answer="回答[S1]",
            sources=sources,
            model_called=True,
            latency_ms=50.0,
            error=None,
        )
        assert isinstance(result, CaseResult)
        assert result.passed is False


# ============================================================
# 汇总统计测试
# ============================================================


class TestPercentile:
    def test_p50_median(self) -> None:
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == pytest.approx(3.0)

    def test_p95(self) -> None:
        vals = list(range(1, 101))  # 1..100
        # P95 of 1..100: k = 99*0.95 = 94.05, f=94, c=0.05 → 95 + 0.05*(96-95) = 95.05
        assert _percentile(vals, 95) == pytest.approx(95.05)

    def test_p99(self) -> None:
        vals = list(range(1, 101))
        assert _percentile(vals, 99) == pytest.approx(99.01)

    def test_empty_list(self) -> None:
        assert _percentile([], 50) == 0.0

    def test_single_value(self) -> None:
        assert _percentile([42.0], 95) == 42.0

    def test_two_values(self) -> None:
        assert _percentile([1.0, 2.0], 50) == pytest.approx(1.5)
        assert _percentile([1.0, 2.0], 0) == 1.0
        assert _percentile([1.0, 2.0], 100) == 2.0

    def test_all_identical(self) -> None:
        assert _percentile([5.0, 5.0, 5.0], 50) == 5.0
        assert _percentile([5.0, 5.0, 5.0], 95) == 5.0


class TestEvalReport:
    def test_report_to_dict(self) -> None:
        report = EvalReport(
            total=10,
            passed=8,
            failed=2,
            pass_rate=0.8,
            mode="fake",
            case_results=[
                CaseResult(
                    case_id="c1",
                    question="q",
                    passed=True,
                    should_answer=True,
                    actually_answered=True,
                )
            ],
        )
        d = report.to_dict()
        assert d["summary"]["total"] == 10
        assert d["summary"]["passed"] == 8
        assert len(d["details"]) == 1

    def test_empty_report(self) -> None:
        report = EvalReport()
        d = report.to_dict()
        assert d["summary"]["total"] == 0
        assert d["summary"]["pass_rate"] == 0.0


# ============================================================
# Fake LLM 测试
# ============================================================


class TestScriptedFakeLLM:
    def test_returns_matching_answer(self) -> None:
        llm = ScriptedFakeLLM(answers={"什么是Git？": "Git是版本控制工具。[S1]"})
        result = llm.generate([
            {"role": "system", "content": "system"},
            {"role": "user", "content": "用户问题：什么是Git？"},
        ])
        assert "Git是版本控制工具" in result

    def test_returns_default_for_unknown(self) -> None:
        llm = ScriptedFakeLLM(answers={}, default_answer="默认回答")
        result = llm.generate([
            {"role": "user", "content": "未知问题"},
        ])
        assert result == "默认回答"

    def test_tracks_call_count(self) -> None:
        llm = ScriptedFakeLLM()
        assert llm.call_count == 0
        llm.generate([{"role": "user", "content": "q"}])
        assert llm.call_count == 1

    def test_does_not_connect_network(self) -> None:
        """FakeLLM 不访问网络。"""
        llm = ScriptedFakeLLM()
        result = llm.generate([{"role": "user", "content": "test"}])
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================
# ErrorFakeLLM 测试（runner.py 中的评估用 ErrorFakeLLM）
# ============================================================


class TestErrorFakeLLM:
    def test_zero_error_rate_always_succeeds(self) -> None:
        """错误率为 0 时全部成功。"""
        llm = ErrorFakeLLM(error_rate=0.0)
        for _ in range(20):
            result = llm.generate([{"role": "user", "content": "q"}])
            assert result == "Fake 回答内容。"

    def test_rate_one_always_fails(self) -> None:
        """错误率为 1 时全部抛出异常。"""
        llm = ErrorFakeLLM(error_rate=1.0)
        with pytest.raises(RuntimeError, match="模拟的 LLM 错误"):
            llm.generate([{"role": "user", "content": "q"}])

    def test_tracks_call_count(self) -> None:
        """call_count 正确追踪。"""
        llm = ErrorFakeLLM(error_rate=0.0)
        for _ in range(3):
            llm.generate([{"role": "user", "content": "q"}])
        assert llm.call_count == 3

    def test_custom_error_message(self) -> None:
        """自定义异常消息。"""
        llm = ErrorFakeLLM(error_rate=1.0, error_msg="自定义错误")
        with pytest.raises(RuntimeError, match="自定义错误"):
            llm.generate([{"role": "user", "content": "q"}])


# ============================================================
# 评估运行器测试（Fake 模式）
# ============================================================


class TestRunEvaluationFake:
    def test_empty_cases_returns_empty_report(self, tmp_path: Path) -> None:
        """空案例列表返回空报告。"""
        (tmp_path / "knowledge").mkdir()
        report = run_evaluation([], mode="fake", knowledge_dir=str(tmp_path / "knowledge"))
        assert report.total == 0
        assert report.passed == 0

    def test_single_case_pass(self, tmp_path: Path) -> None:
        """单条案例完整评估通过。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("这是测试内容，包含重要的关键词信息。", encoding="utf-8")

        case = _make_case(
            id="test-pass",
            question="测试",
            expected_sources=[],
            expected_keywords=["关键词"],
            fake_answer="回答中包含关键词[S1]。",
        )
        report = run_evaluation(
            [case], mode="fake", knowledge_dir=str(kd),
            llm_client=ScriptedFakeLLM(answers={"测试": "回答中包含关键词[S1]。"}),
        )
        assert report.total == 1
        assert report.passed == 1

    def test_refusal_case_pass(self, tmp_path: Path) -> None:
        """资料不足案例正确拒答。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "notes.md").write_text("一些无关内容。", encoding="utf-8")

        case = _make_case(
            id="test-refusal",
            question="量子计算",
            should_answer=False,
            expected_sources=[],
            expected_keywords=[],
            fake_answer="",
        )
        report = run_evaluation(
            [case], mode="fake", knowledge_dir=str(kd),
            llm_client=ScriptedFakeLLM(),
        )
        assert report.total == 1
        # 检索结果相关度过低，系统应正确拒答且不调用模型
        assert report.case_results[0].model_called is False
        assert report.case_results[0].passed is True

    def test_limit_cases(self, tmp_path: Path) -> None:
        """limit 参数限制执行的案例数。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("测试内容。", encoding="utf-8")

        cases = [
            _make_case(id=f"c{i}", question=f"问题{i}", expected_sources=[], expected_keywords=[])
            for i in range(10)
        ]
        report = run_evaluation(
            cases, mode="fake", knowledge_dir=str(kd), limit=3,
            llm_client=ScriptedFakeLLM(),
        )
        assert report.total == 3

    def test_category_filter(self, tmp_path: Path) -> None:
        """category 过滤只执行指定类别。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("测试内容。", encoding="utf-8")

        cases = [
            _make_case(id="a1", question="q1", category="A", expected_sources=[], expected_keywords=[]),
            _make_case(id="b1", question="q2", category="B", expected_sources=[], expected_keywords=[]),
            _make_case(id="a2", question="q3", category="A", expected_sources=[], expected_keywords=[]),
        ]
        report = run_evaluation(
            cases, mode="fake", knowledge_dir=str(kd),
            category_filter="A",
            llm_client=ScriptedFakeLLM(),
        )
        assert report.total == 2
        assert all(r.case_id.startswith("a") for r in report.case_results)

    def test_single_failure_does_not_stop_evaluation(self, tmp_path: Path) -> None:
        """单条失败不影响后续案例评估。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("测试内容。", encoding="utf-8")

        cases = [
            _make_case(id="fail", question="不存在的问题", expected_sources=["impossible.md"]),
            _make_case(id="pass", question="测试", expected_sources=[], expected_keywords=[],
                       fake_answer="回答[S1]。"),
        ]
        report = run_evaluation(
            cases, mode="fake", knowledge_dir=str(kd),
            llm_client=ScriptedFakeLLM(answers={"测试": "回答[S1]。"}),
        )
        assert report.total == 2
        assert report.case_results[0].passed is False
        assert report.case_results[1].passed is True

    def test_report_save_and_load(self, tmp_path: Path) -> None:
        """报告可以保存和重新加载。"""
        report = EvalReport(
            total=1,
            passed=1,
            mode="fake",
            case_results=[
                CaseResult(
                    case_id="c1",
                    question="q",
                    passed=True,
                    should_answer=True,
                    actually_answered=True,
                )
            ],
        )
        path = tmp_path / "output" / "report.json"
        save_report(report, path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["summary"]["total"] == 1

    def test_format_summary_includes_key_sections(self) -> None:
        """格式化摘要包含关键段落。"""
        report = EvalReport(
            total=5,
            passed=4,
            failed=1,
            pass_rate=0.8,
            source_hit_rate=0.9,
            citation_valid_rate=1.0,
            refusal_accuracy=1.0,
            avg_keyword_coverage=0.85,
            mode="fake",
            case_results=[
                CaseResult(
                    case_id="bad",
                    question="失败案例",
                    passed=False,
                    should_answer=True,
                    actually_answered=True,
                    failure_reasons=["预期来源未命中"],
                )
            ],
        )
        summary = format_summary(report)
        assert "VibeFlow RAG 评估报告" in summary
        assert "80.0%" in summary
        assert "失败案例" in summary
        assert "预期来源未命中" in summary

    def test_evaluation_with_real_knowledge_dir(self, tmp_path: Path) -> None:
        """使用真实知识库内容进行端到端评估。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "git-notes.md").write_text(
            "在项目开始时初始化Git，并在关键开发节点提交代码。\n"
            "Git提交相当于为项目建立一个个可恢复的存档点。",
            encoding="utf-8",
        )

        case = EvaluationCase(
            id="e2e-git",
            category="精确关键词问题",
            question="什么是Git提交？",
            should_answer=True,
            expected_sources=["git-notes.md"],
            expected_keywords=["存档点", "可恢复"],
            forbidden_keywords=[],
            retrieval_mode="keyword",
            fake_answer="Git提交相当于为项目建立一个个可恢复的存档点[S1]。",
        )
        report = run_evaluation(
            [case], mode="fake", knowledge_dir=str(kd),
            llm_client=ScriptedFakeLLM(
                answers={"什么是Git提交？": "Git提交相当于为项目建立一个个可恢复的存档点[S1]。"}
            ),
        )
        assert report.total == 1
        assert report.case_results[0].source_hit is True
        assert report.case_results[0].keyword_coverage == 1.0

    def test_error_fake_llm_does_not_crash_evaluation(self, tmp_path: Path) -> None:
        """ErrorFakeLLM 部分异常不导致评估中断。"""
        kd = tmp_path / "knowledge"
        kd.mkdir()
        (kd / "test.md").write_text("测试内容关键词。", encoding="utf-8")

        cases = [
            _make_case(id=f"c{i}", question=f"问题{i}", expected_sources=[], expected_keywords=[])
            for i in range(5)
        ]
        report = run_evaluation(
            cases, mode="fake", knowledge_dir=str(kd),
            llm_client=ErrorFakeLLM(error_rate=0.5),
        )
        assert report.total == 5
        # 有异常也有正常：总计 = 通过 + 失败
        assert report.passed + report.failed == 5


# ============================================================
# 原有功能回归
# ============================================================


class TestNoRegressions:
    """确保评估模块不破坏原有功能。"""

    def test_knowledge_service_imports(self) -> None:
        from vibeflow.knowledge_service import KnowledgeService
        assert KnowledgeService is not None

    def test_rag_service_imports(self) -> None:
        from vibeflow.knowledge_rag_service import RAGService
        assert RAGService is not None

    def test_keyword_retriever_imports(self) -> None:
        from vibeflow.knowledge_retriever import KeywordRetriever
        assert KeywordRetriever is not None

    def test_rag_result_model_unchanged(self) -> None:
        from vibeflow.knowledge_rag_models import RAGResult
        # 验证原有字段仍存在
        r = RAGResult(question="q", answer="a")
        assert r.model_called is False
        assert r.refused_due_to_insufficient is False
        assert r.error_message is None
        assert r.sources == []
