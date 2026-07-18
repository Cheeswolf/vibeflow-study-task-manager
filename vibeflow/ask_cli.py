from __future__ import annotations

import argparse
import os
import sys

from vibeflow.knowledge_llm_client import LLMClient, OllamaClient
from vibeflow.knowledge_rag_service import RAGService
from vibeflow.knowledge_service import KnowledgeService


def _check_ollama_available() -> None:
    """检查 Ollama 是否可用，不可用时给出清晰指引。"""
    try:
        import ollama  # noqa: F401
    except ImportError:
        print(
            "RAG 回答功能需要 ollama Python 库。\n"
            "请运行: pip install ollama",
            file=sys.stderr,
        )
        sys.exit(1)

    # 检查 Ollama 服务是否在运行
    try:
        import ollama

        ollama.list()
    except Exception:
        print(
            "无法连接到 Ollama 服务。\n"
            "\n"
            "请确认 Ollama 已安装并启动：\n"
            "  • 下载安装: https://ollama.com\n"
            "  • 启动服务: ollama serve\n"
            "  • 下载模型: ollama pull <模型名称>\n"
            "  • 查看模型: ollama list\n"
            "\n"
            "配置模型名称：\n"
            '  set VIBEFLOW_OLLAMA_MODEL=<模型名称>    (Windows CMD)\n'
            '  $env:VIBEFLOW_OLLAMA_MODEL="<模型名称>" (PowerShell)\n'
            '  export VIBEFLOW_OLLAMA_MODEL=<模型名称>  (Bash)',
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="VibeFlow 知识库问答（RAG）")
    parser.add_argument(
        "--mode",
        choices=["keyword", "vector", "hybrid"],
        default="hybrid",
        help="检索模式：keyword、vector、hybrid。默认：hybrid",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式：显示检索结果摘要（不显示完整系统提示词）",
    )
    args = parser.parse_args()

    _check_ollama_available()

    service = KnowledgeService("knowledge", mode=args.mode)
    llm_client: LLMClient = OllamaClient()
    rag = RAGService(service, llm_client)

    print(f"VibeFlow 知识库问答（模式：{args.mode}，模型：{llm_client.model_name}）")
    print("输入问题后按回车获取回答，输入 /q 退出\n")

    while True:
        try:
            question = input("请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            result = rag.ask(question)
            print(f"\n{result.answer}\n")
            continue
        if question in ("/q", "/quit", "/exit"):
            print("再见！")
            break

        result = rag.ask(question)

        if result.error_message:
            print(f"\n错误：{result.error_message}\n")
            continue

        print(f"\n回答：\n{result.answer}\n")

        if args.debug and result.sources:
            print(f"检索到 {len(result.sources)} 条参考来源：")
            for s in result.sources:
                print(
                    f"  {s.reference_label} {s.source_file} "
                    f"/ chunk-{s.chunk_index} "
                    f"(score: {s.score:.4f})"
                )
            print()

        if result.sources:
            print("引用来源：")
            for s in result.sources:
                print(f"  {s.reference_label} {s.source_file} / chunk-{s.chunk_index}")
            print()

        if result.refused_due_to_insufficient:
            print("（当前知识库资料不足以提供完整回答）\n")


if __name__ == "__main__":
    main()
