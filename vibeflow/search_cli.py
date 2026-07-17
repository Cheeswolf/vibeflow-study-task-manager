from __future__ import annotations

import argparse

from vibeflow.knowledge_service import KnowledgeService


def main() -> None:
    parser = argparse.ArgumentParser(description="VibeFlow 知识检索")
    parser.add_argument(
        "--mode",
        choices=["keyword", "vector", "hybrid"],
        default="keyword",
        help="检索模式：keyword（关键词）、vector（向量语义）、hybrid（混合）。默认：keyword",
    )
    args = parser.parse_args()

    service = KnowledgeService("knowledge", mode=args.mode)
    print(f"VibeFlow 知识检索（模式：{args.mode}，输入 /q 退出）\n")

    while True:
        try:
            query = input("搜索 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query in ("/q", "/quit", "/exit"):
            print("再见！")
            break

        results = service.search(query)

        if not results:
            print("未找到相关内容。\n")
            continue

        print(f"找到 {len(results)} 个结果：\n")

        for i, r in enumerate(results, 1):
            print(f"── 结果 {i} ──")
            print(f"来源：{r.source_file}")
            print(f"块号：{r.chunk_index}  相关度：{r.score:.4f}")
            print(r.content)
            print()


if __name__ == "__main__":
    main()
