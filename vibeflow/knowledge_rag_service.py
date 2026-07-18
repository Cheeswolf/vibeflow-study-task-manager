from __future__ import annotations

from vibeflow.knowledge_context_builder import ContextBuilder
from vibeflow.knowledge_llm_client import LLMClient
from vibeflow.knowledge_prompt_builder import PromptBuilder
from vibeflow.knowledge_rag_models import RAGResult
from vibeflow.knowledge_service import KnowledgeService

# --- 相关性阈值 ---
#
# 不同检索模式的分数含义不同，因此需要独立阈值：
#
# keyword：分数 = Σ(len(term)²) / log(len(chunk) + e)
#   单个中文字匹配 ≈ 0.3–0.5，双字词匹配 ≈ 1.3–2.0。
#   阈值 1.0 要求至少一个有效双字词或两个以上单字匹配。
#   低于 1.0 的匹配通常是停用词残留或噪声。
#
# vector：余弦相似度，范围 [0, 1]（实际极少出现负值）。
#   0.3 是语义检索中"弱相关"的常用下界。
#   相同主题内容通常 ≥ 0.5，无关内容通常 < 0.2。
#
# hybrid：混合得分经 min-max 归一化到 [0, 1]。
#   因为归一化是相对的（依赖于当前批次的最值），
#   阈值设为保守的 0.15 —— 仅过滤掉所有文本块
#   在两种检索器中都接近零信号的情况。

_MIN_KEYWORD_SCORE = 1.0
_MIN_VECTOR_SCORE = 0.3
_MIN_HYBRID_SCORE = 0.15


class RAGService:
    """RAG 回答服务。

    编排完整链路：
    KnowledgeService（检索）→ ContextBuilder（上下文）
    → PromptBuilder（提示词）→ LLMClient（生成）→ RAGResult

    不直接：
    - 读取知识文件
    - 计算相似度
    - 拼接提示词
    - 依赖 Ollama 具体返回结构
    - 执行 Bash / 文件写入 / Git
    """

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        llm_client: LLMClient,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._knowledge = knowledge_service
        self._llm = llm_client
        self._context_builder = context_builder or ContextBuilder()
        self._prompt_builder = prompt_builder or PromptBuilder()

    def ask(self, question: str) -> RAGResult:
        """处理用户问题，返回结构化 RAG 结果。"""
        # 情况一：查询为空
        if not question or not question.strip():
            return RAGResult(
                question=question,
                answer="请输入您的问题。",
                retrieval_mode=self._knowledge.mode,
            )

        question = question.strip()
        mode = self._knowledge.mode

        # 检索（最多获取足够上下文用的结果）
        results = self._knowledge.search(
            question, top_k=self._context_builder.max_chunks
        )

        # 情况二：没有检索结果
        if not results:
            return RAGResult(
                question=question,
                answer="当前知识库中没有找到足够相关的资料。",
                retrieval_mode=mode,
                refused_due_to_insufficient=True,
            )

        # 情况三：检索结果相关性过低
        if not self._is_relevant(results, mode):
            return RAGResult(
                question=question,
                answer="当前知识库无法支持完整回答 — 检索到的资料相关度过低。",
                retrieval_mode=mode,
                refused_due_to_insufficient=True,
            )

        # 情况四：有相关资料，构建上下文并调用模型
        context_text, sources = self._context_builder.build(results)
        messages = self._prompt_builder.build_messages(question, context_text)

        try:
            answer = self._llm.generate(messages)
        except Exception as e:
            return RAGResult(
                question=question,
                answer="",
                sources=sources,
                retrieval_mode=mode,
                model_called=False,
                error_message=str(e),
            )

        return RAGResult(
            question=question,
            answer=answer,
            sources=sources,
            retrieval_mode=mode,
            model_called=True,
        )

    @staticmethod
    def _is_relevant(results, mode: str) -> bool:
        """判断检索结果是否达到最低相关性要求。"""
        if not results:
            return False

        if mode == "keyword":
            return any(r.score >= _MIN_KEYWORD_SCORE for r in results)
        elif mode == "vector":
            return any(r.score >= _MIN_VECTOR_SCORE for r in results)
        elif mode == "hybrid":
            return any(r.score >= _MIN_HYBRID_SCORE for r in results)
        else:
            # 未知模式，保守放行
            return True
