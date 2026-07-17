from __future__ import annotations

from typing import Any

# 集中配置默认嵌入模型名称。支持中文和英文的多语言模型，本地运行，不依赖云端 API。
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    """文本嵌入模型封装。

    模型延迟加载：构造 Embedder 时不会下载或初始化模型，
    仅在首次调用 encode() 时加载。模型名称集中在模块常量中，
    不散落硬编码。
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or DEFAULT_MODEL_NAME
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载。用于测试验证延迟加载行为。"""
        return self._model is not None

    def encode(self, texts: list[str]) -> "np.ndarray":  # type: ignore[name-defined]  # noqa: F821
        """将文本列表编码为向量矩阵，每行是一个文本的向量。"""
        self._ensure_loaded()
        import numpy as np

        vectors = self._model.encode(texts, convert_to_numpy=True)
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        return vectors

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "向量检索需要 sentence-transformers 库。\n"
                "请运行: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(self._model_name)
