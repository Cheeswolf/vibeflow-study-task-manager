"""评估数据模型。

所有结构均为 dataclass，便于 JSON 序列化和测试构造。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvaluationCase:
    """单条评估案例。

    从 evaluation/rag_cases.json 加载，描述期望的评估标准。
    """

    id: str
    category: str
    question: str
    should_answer: bool
    expected_sources: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)
    retrieval_mode: str = "keyword"
    require_all_sources: bool = False
    description: str = ""
    fake_answer: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluationCase":
        return cls(
            id=d["id"],
            category=d["category"],
            question=d["question"],
            should_answer=d["should_answer"],
            expected_sources=d.get("expected_sources", []),
            expected_keywords=d.get("expected_keywords", []),
            forbidden_keywords=d.get("forbidden_keywords", []),
            retrieval_mode=d.get("retrieval_mode", "keyword"),
            require_all_sources=d.get("require_all_sources", False),
            description=d.get("description", ""),
            fake_answer=d.get("fake_answer", ""),
        )


@dataclass
class CaseResult:
    """单条案例的评估结果。"""

    case_id: str
    question: str
    passed: bool
    should_answer: bool
    actually_answered: bool
    expected_sources: list[str] = field(default_factory=list)
    actual_sources: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    source_hit: bool = False
    citation_valid: bool = True
    keyword_coverage: float = 0.0
    forbidden_keyword_hit: bool = False
    model_called: bool = False
    latency_ms: float = 0.0
    error: str | None = None
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """评估总报告。"""

    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    source_hit_rate: float = 0.0
    citation_valid_rate: float = 0.0
    refusal_accuracy: float = 0.0
    avg_keyword_coverage: float = 0.0
    model_error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mode: str = "fake"
    case_results: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """序列化为字典，便于 JSON 输出。"""
        return {
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": round(self.pass_rate, 4),
                "source_hit_rate": round(self.source_hit_rate, 4),
                "citation_valid_rate": round(self.citation_valid_rate, 4),
                "refusal_accuracy": round(self.refusal_accuracy, 4),
                "avg_keyword_coverage": round(self.avg_keyword_coverage, 4),
                "model_error_rate": round(self.model_error_rate, 4),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "p50_latency_ms": round(self.p50_latency_ms, 2),
                "p95_latency_ms": round(self.p95_latency_ms, 2),
                "mode": self.mode,
            },
            "details": [
                {
                    "case_id": r.case_id,
                    "question": r.question,
                    "passed": r.passed,
                    "should_answer": r.should_answer,
                    "actually_answered": r.actually_answered,
                    "source_hit": r.source_hit,
                    "citation_valid": r.citation_valid,
                    "keyword_coverage": round(r.keyword_coverage, 4),
                    "forbidden_keyword_hit": r.forbidden_keyword_hit,
                    "model_called": r.model_called,
                    "latency_ms": round(r.latency_ms, 2),
                    "error": r.error,
                    "failure_reasons": r.failure_reasons,
                }
                for r in self.case_results
            ],
        }
