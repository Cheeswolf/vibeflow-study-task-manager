# VibeFlow 项目记忆

## 项目目标
开发一个本地桌面学习任务管理器，帮助用户记录课程任务、截止日期和完成状态。

## 当前版本
V0.1 MVP

## 技术栈
- Python 3.10+
- Tkinter / ttk
- SQLite
- pytest
- sentence-transformers（向量检索）
- ollama（本地大模型 RAG 回答）

## 架构约束
- UI 只负责展示和接收输入。
- 数据库操作集中在 `vibeflow/db.py`。
- 业务逻辑集中在 `vibeflow/services.py`。
- 不要在 UI 文件里直接拼接 SQL。
- 新功能必须尽量补充单元测试。
- 修改前先阅读相关文件，不要无关重构。
- 每次只完成一个明确任务。

## 知识检索架构
- `knowledge_loader.py` — 文档加载，扫描 knowledge/ 目录
- `knowledge_chunker.py` — 文本切分，不依赖具体 Retriever
- `knowledge_models.py` — 共享数据模型（TextChunk, SearchResult）
- `knowledge_embedder.py` — 嵌入模型封装，延迟加载，集中配置模型名
- `knowledge_retriever.py` — KeywordRetriever 关键词检索
- `knowledge_vector_retriever.py` — VectorRetriever 余弦相似度检索
- `knowledge_hybrid_retriever.py` — HybridRetriever 加权融合 + 去重
- `knowledge_service.py` — 组合 Loader → Chunker → Retriever 流水线
- `search_cli.py` — CLI 入口，只负责交互，业务委托给 Service

检索器统一接口：`search(chunks, query, top_k=None) → list[SearchResult]`

## RAG 回答架构
- `knowledge_rag_models.py` — 共享数据模型（RAGResult, SourceInfo）
- `knowledge_llm_client.py` — LLMClient 抽象接口 + OllamaClient 实现
- `knowledge_context_builder.py` — ContextBuilder 检索结果 → 带编号上下文
- `knowledge_prompt_builder.py` — PromptBuilder 系统提示词 + 用户消息
- `knowledge_rag_service.py` — RAGService 编排检索 → 上下文 → 生成全链路
- `ask_cli.py` — CLI 问答入口，只负责交互，业务委托给 RAGService

LLM 客户端统一接口：`generate(messages) -> str`

配置方式：
- 默认模型：环境变量 `VIBEFLOW_OLLAMA_MODEL`，未设置时自动探测
- 服务地址：环境变量 `VIBEFLOW_OLLAMA_HOST`，默认 `http://localhost:11434`
- 客户端延迟初始化，import 时不连接 Ollama

## 评估体系
- `evaluation/rag_cases.json` — 黄金评估数据集（18 条案例，8 个类别）
- `vibeflow/evaluation/models.py` — 数据模型（EvaluationCase, CaseResult, EvalReport）
- `vibeflow/evaluation/scorer.py` — 确定性评分规则（不依赖 LLM 评判）
- `vibeflow/evaluation/runner.py` — 评估运行器（支持 fake/ollama 模式）
- `vibeflow/evaluate_rag.py` — CLI 入口，只负责交互，业务委托给 runner

评估规则：
- 来源命中：expected_sources 至少一个出现在 actual_sources
- 引用有效：回答中所有 [SN] 标签对应实际来源列表的合法索引
- 关键词覆盖：expected_keywords 在回答中的简单字符串命中率（≥50% 通过）
- 禁止词：forbidden_keywords 命中则失败
- 拒答判断：should_answer=false 时验证系统明确拒答
- 所有规则基于可解释的字符串匹配，不调用 LLM 评判

自动化测试强制使用 FakeLLMClient，禁止在 pytest 中连接真实 Ollama。

## 并发压力测试
- `vibeflow/load_test.py` — 并发压力测试工具（基于 concurrent.futures）
- 支持：并发数、总请求数、超时、Fake/Ollama 模式、异常率配置
- 指标：成功率、吞吐量、min/avg/P50/P95/P99/max 响应时间、异常分布
- 每个请求创建独立 RAGService，避免状态串扰
- pytest 中禁止运行真实 Ollama 压力测试

## 禁止行为
- 不得使用云端 API
- 不得让自动化测试依赖真实 Ollama
- 评估输出目录 eval_results/ 和 load_results/ 不纳入版本控制

## MVP 功能
- 新建学习任务
- 展示任务列表
- 按状态筛选
- 标记完成/未完成
- 删除任务
- 展示任务统计

## 数据字段
- id
- title
- course
- due_date
- status
- created_at

## 验收标准
- 程序可通过 `python -m vibeflow.main` 启动。
- 数据重启后仍然存在。
- 核心服务测试可通过 `pytest`。
