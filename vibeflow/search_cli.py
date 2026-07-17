from __future__ import annotations

from vibeflow.knowledge_service import KnowledgeService


def main() -> None:
    service = KnowledgeService("knowledge")
    print("VibeFlow 知识检索（输入 /q 退出）\n")

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
