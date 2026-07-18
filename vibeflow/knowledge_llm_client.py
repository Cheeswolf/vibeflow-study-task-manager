from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """大模型调用抽象接口。

    定义为 generate(messages) -> str，所有大模型实现
    必须遵循此接口，便于测试替换和未来扩展。
    """

    @property
    def model_name(self) -> str:
        """模型名称。子类可覆盖，默认返回 'unknown'。"""
        return "unknown"

    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        """根据消息列表生成回答文本。"""
        ...


# --- 默认配置 ---

DEFAULT_OLLAMA_HOST = os.environ.get(
    "VIBEFLOW_OLLAMA_HOST", "http://localhost:11434"
)


def _resolve_model_name() -> str:
    """解析模型名称。

    优先级：
    1. 环境变量 VIBEFLOW_OLLAMA_MODEL
    2. 自动探测 Ollama 中已安装的模型（取第一个）
    3. 无法确定时报错
    """
    env_model = os.environ.get("VIBEFLOW_OLLAMA_MODEL", "").strip()
    if env_model:
        return env_model

    # 尝试自动探测
    try:
        import ollama

        resp = ollama.list()
        # ollama v0.6+ 返回 ListResponse 对象，模型列表在 .models 属性中
        if isinstance(resp, list):
            model_list = resp
        elif hasattr(resp, "models"):
            model_list = resp.models
        else:
            model_list = []

        if model_list:
            first = model_list[0]
            if isinstance(first, dict):
                return first.get("name", first.get("model", ""))
            elif hasattr(first, "model"):
                return first.model
            return str(first)
    except Exception:
        pass

    raise RuntimeError(
        "无法确定 Ollama 模型名称。请设置环境变量 VIBEFLOW_OLLAMA_MODEL，\n"
        "例如：\n"
        '  set VIBEFLOW_OLLAMA_MODEL=qwen3:latest       (Windows CMD)\n'
        '  $env:VIBEFLOW_OLLAMA_MODEL="qwen3:latest"    (PowerShell)\n'
        '  export VIBEFLOW_OLLAMA_MODEL=qwen3:latest    (Bash)\n'
        "\n"
        "查看已安装的模型：ollama list"
    )


class OllamaClient(LLMClient):
    """本地 Ollama 大模型客户端。

    使用 Ollama 官方 Python 库的 chat 接口。
    延迟初始化：构造时不会连接 Ollama，仅在首次 generate() 时连接。

    配置：
    - VIBEFLOW_OLLAMA_MODEL：模型名称（可选，未设置时自动探测）
    - VIBEFLOW_OLLAMA_HOST：Ollama 服务地址（默认 http://localhost:11434）
    """

    def __init__(
        self,
        model_name: str | None = None,
        host: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._host = host or DEFAULT_OLLAMA_HOST
        self._client: Any = None
        self._resolved_model: str | None = None

    @property
    def is_loaded(self) -> bool:
        """是否已初始化连接。用于测试验证延迟加载行为。"""
        return self._client is not None

    @property
    def model_name(self) -> str:
        """实际使用的模型名称。首次访问时解析。"""
        if self._resolved_model is None:
            self._resolved_model = self._model_name or _resolve_model_name()
        return self._resolved_model

    def generate(self, messages: list[dict[str, str]]) -> str:
        self._ensure_loaded()

        try:
            response = self._client.chat(
                model=self.model_name,
                messages=messages,
            )
        except Exception as e:
            msg = str(e)
            # 识别常见错误并给出清晰提示
            if "connection refused" in msg.lower() or "connect" in msg.lower():
                raise RuntimeError(
                    "无法连接到 Ollama 服务。\n"
                    "请确认 Ollama 已启动：\n"
                    "  • 运行 ollama serve 启动服务\n"
                    "  • 如使用非默认地址，请设置 VIBEFLOW_OLLAMA_HOST"
                ) from e
            if "not found" in msg.lower() or "model" in msg.lower():
                raise RuntimeError(
                    f"模型 '{self.model_name}' 不存在。\n"
                    "请检查：\n"
                    f"  • 运行 ollama list 查看已安装的模型\n"
                    f"  • 运行 ollama pull {self.model_name} 下载模型\n"
                    "  • 或通过 VIBEFLOW_OLLAMA_MODEL 指定其他模型"
                ) from e
            raise RuntimeError(f"Ollama 调用失败：{msg}") from e

        answer = response.get("message", {}).get("content", "")
        if not answer:
            # 模型返回了空内容
            return ""
        return answer

    def _ensure_loaded(self) -> None:
        """延迟初始化 Ollama 客户端。"""
        if self._client is not None:
            return
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "RAG 回答功能需要 ollama Python 库。\n"
                "请运行: pip install ollama"
            ) from e

        # 设置自定义 host（显式赋值以覆盖系统环境变量）
        if self._host != "http://localhost:11434":
            os.environ["OLLAMA_HOST"] = self._host

        self._client = ollama
