from __future__ import annotations

from pathlib import Path


class KnowledgeLoader:
    _SUPPORTED_SUFFIXES = {".md", ".txt"}

    def __init__(self, knowledge_dir: str | Path = "knowledge") -> None:
        self.knowledge_dir = Path(knowledge_dir)
        if not self.knowledge_dir.is_dir():
            raise FileNotFoundError(f"知识目录不存在：{self.knowledge_dir}")

    def load(self) -> list[dict[str, str]]:
        documents: list[dict[str, str]] = []

        for file_path in sorted(self.knowledge_dir.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self._SUPPORTED_SUFFIXES:
                continue

            content = file_path.read_text(encoding="utf-8")
            documents.append(
                {
                    "content": content,
                    "source_file": str(file_path),
                }
            )

        return documents
