from __future__ import annotations

import re

from vibeflow.knowledge_models import TextChunk

_MIN_CHUNK_SIZE = 20
_MAX_CHUNK_SIZE = 800
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？.!?\n])(?=\S)")


class TextChunker:
    def chunk(self, raw_text: str, source_file: str) -> list[TextChunk]:
        if not raw_text.strip():
            return []

        blocks = self._split_preserving_code(raw_text)
        chunks: list[str] = []

        for block in blocks:
            if self._is_code_block(block):
                chunks.append(block)
            else:
                paragraphs = block.split("\n\n")
                for paragraph in paragraphs:
                    cleaned = paragraph.strip()
                    if cleaned:
                        chunks.append(cleaned)

        chunks = self._merge_short(chunks)
        chunks = self._split_large(chunks)

        return [
            TextChunk(content=chunk, source_file=source_file, chunk_index=i)
            for i, chunk in enumerate(chunks)
        ]

    def _split_preserving_code(self, raw_text: str) -> list[str]:
        sections: list[str] = []
        in_code = False
        buffer = ""

        for line in raw_text.splitlines(keepends=True):
            if line.lstrip().startswith("```"):
                if in_code:
                    buffer += line
                    sections.append(buffer)
                    buffer = ""
                    in_code = False
                else:
                    if buffer.strip():
                        sections.append(buffer)
                    buffer = line
                    in_code = True
            else:
                buffer += line

        if buffer.strip():
            sections.append(buffer)

        return sections

    @staticmethod
    def _is_code_block(block: str) -> bool:
        stripped = block.lstrip()
        return stripped.startswith("```") and stripped.rstrip().endswith("```")

    @staticmethod
    def _merge_short(chunks: list[str]) -> list[str]:
        if len(chunks) <= 1:
            return chunks

        merged: list[str] = []
        buffer = ""

        for chunk in chunks:
            if buffer:
                buffer += "\n\n" + chunk
            else:
                buffer = chunk

            if len(buffer) >= _MIN_CHUNK_SIZE:
                merged.append(buffer)
                buffer = ""

        if buffer:
            if merged:
                merged[-1] += "\n\n" + buffer
            else:
                merged.append(buffer)

        return merged

    @staticmethod
    def _split_large(chunks: list[str]) -> list[str]:
        result: list[str] = []

        for chunk in chunks:
            if len(chunk) <= _MAX_CHUNK_SIZE:
                result.append(chunk)
                continue

            sub_chunks = _SENTENCE_BOUNDARY.split(chunk)
            current = ""

            for sub in sub_chunks:
                if len(current) + len(sub) > _MAX_CHUNK_SIZE and current:
                    result.append(current.strip())
                    current = sub
                else:
                    current += sub

            if current.strip():
                result.append(current.strip())

        return result
