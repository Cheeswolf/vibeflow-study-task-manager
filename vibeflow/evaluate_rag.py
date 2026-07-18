"""RAG 评估 CLI 入口。

用法：
    # Fake 模式（默认，不连接真实 Ollama）
    python -m vibeflow.evaluate_rag

    # Fake 模式 + 限制案例数
    python -m vibeflow.evaluate_rag --mode fake --limit 5

    # 按类别过滤
    python -m vibeflow.evaluate_rag --mode fake --category "资料不足问题"

    # 输出 JSON 报告
    python -m vibeflow.evaluate_rag --mode fake --output eval_results/result.json

    # 真实 Ollama 模式（需 Ollama 运行中）
    python -m vibeflow.evaluate_rag --mode ollama
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vibeflow.evaluation.runner import (
    format_summary,
    load_cases,
    run_evaluation,
    save_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VibeFlow RAG 评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m vibeflow.evaluate_rag                     # Fake 模式全部案例\n"
            "  python -m vibeflow.evaluate_rag --limit 5            # 只跑前 5 条\n"
            "  python -m vibeflow.evaluate_rag --category 资料不足问题 # 按类别过滤\n"
            "  python -m vibeflow.evaluate_rag --output result.json # 保存 JSON 报告\n"
            "  python -m vibeflow.evaluate_rag --mode ollama        # 真实 Ollama 模式"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["fake", "ollama"],
        default="fake",
        help="评估模式：fake（默认，不连接真实模型）/ ollama（需本地 Ollama 运行中）",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="评估案例文件路径（默认：evaluation/rag_cases.json）",
    )
    parser.add_argument(
        "--knowledge-dir",
        default="knowledge",
        help="知识库目录路径（默认：knowledge）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制执行的案例数量",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="仅执行指定类别的案例",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="将完整报告输出为 JSON 文件（如 eval_results/result.json）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="只输出摘要，不显示失败明细",
    )

    args = parser.parse_args()

    # 显示模式
    mode_label = (
        "Fake LLM 模式（不连接真实模型）"
        if args.mode == "fake"
        else "本地 Ollama 模式（需要 Ollama 服务运行中）"
    )
    print(f"评估模式：{mode_label}")
    print()

    # 加载案例
    cases_path = args.cases or str(
        Path(__file__).resolve().parent.parent / "evaluation" / "rag_cases.json"
    )

    try:
        cases = load_cases(cases_path)
    except FileNotFoundError as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

    if args.category:
        cases = [c for c in cases if c.category == args.category]
        if not cases:
            print(f"没有找到类别为 '{args.category}' 的案例", file=sys.stderr)
            sys.exit(1)

    if args.limit:
        cases = cases[: args.limit]

    print(f"加载 {len(cases)} 条评估案例")
    print()

    # 执行评估
    report = run_evaluation(
        cases,
        mode=args.mode,
        knowledge_dir=args.knowledge_dir,
    )

    # 输出摘要
    print(format_summary(report))

    # 保存 JSON（如需要）
    if args.output:
        save_report(report, args.output)
        print(f"完整报告已保存至：{args.output}")


if __name__ == "__main__":
    main()
