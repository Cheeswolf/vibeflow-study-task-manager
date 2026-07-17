from __future__ import annotations

import math
import re

from vibeflow.knowledge_models import SearchResult, TextChunk

_CN_STOP_WORDS = set(
    "的 了 在 是 我 有 和 就 不 人 都 一 个 上 也 很 到 说 要 去 "
    "你 会 着 没有 看 好 自己 这 他 她 它 们 那 些 所 以 之 而 "
    "能 把 被 让 向 从 对 与 或 但 只 并 且 何 再 又 太 可 吗 "
    "呢 吧 啊 哦 嗯 哎 为 还 最 小 大 用 做 来 为 请 叫".split()
)

_WORD_PATTERN = re.compile(r"[a-zA-Z0-9]{2,}")


class KeywordRetriever:
    """关键词检索器。

    通过中文分词（1-gram + 2-gram）和英文单词匹配进行检索，
    使用加权关键词长度平方 / 对数长度归一化计算得分。
    """

    _DEFAULT_RESULT_LIMIT = 3

    def search(
        self,
        chunks: list[TextChunk],
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        limit = top_k if top_k is not None else self._DEFAULT_RESULT_LIMIT

        if not chunks:
            return []

        terms = self._tokenize(query)
        if not terms:
            return []

        scored: list[SearchResult] = []

        for chunk in chunks:
            score = self._score(chunk.content, terms)
            if score > 0:
                scored.append(
                    SearchResult(
                        content=chunk.content,
                        source_file=chunk.source_file,
                        chunk_index=chunk.chunk_index,
                        score=round(score, 4),
                    )
                )

        scored.sort(key=lambda r: (-r.score, r.chunk_index))
        return scored[:limit]

    def _tokenize(self, query: str) -> set[str]:
        query = query.strip()
        if not query:
            return set()

        terms: set[str] = set()
        chars = list(query)

        cn_chars = [c for c in chars if "一" <= c <= "鿿"]

        for i, c in enumerate(chars):
            if "一" <= c <= "鿿":
                if c not in _CN_STOP_WORDS:
                    terms.add(c)
                if i + 1 < len(chars) and "一" <= chars[i + 1] <= "鿿":
                    bigram = c + chars[i + 1]
                    terms.add(bigram)

        english_words = _WORD_PATTERN.findall(query.lower())
        terms.update(w for w in english_words if len(w) >= 2)

        if cn_chars and not any(
            "一" <= t[0] <= "鿿" for t in terms if len(t) == 2
        ):
            pass

        return terms

    def _score(self, chunk: str, terms: set[str]) -> float:
        chunk_lower = chunk.lower()

        numerator = 0
        for term in terms:
            if term in chunk_lower:
                numerator += len(term) ** 2

        if numerator == 0:
            return 0.0

        denominator = math.log(len(chunk) + math.e)
        return numerator / denominator


# 向后兼容别名
KnowledgeRetriever = KeywordRetriever
