# VibeFlow 学习任务管理器

一个用于练习 Vibe Coding 工程化开发流程的 Python 桌面项目。

## 项目结构

```
VibeFlow
├── 学习任务管理
│   ├── vibeflow/main.py              Tkinter 桌面 GUI
│   ├── vibeflow/models.py            数据模型（Task）
│   ├── vibeflow/db.py                SQLite 数据库操作
│   └── vibeflow/services.py          业务逻辑
│
├── 本地知识库
│   └── knowledge/                    Markdown / TXT 知识文档
│       ├── vibe-coding.md
│       ├── claude-code.md
│       └── git-notes.md
│
├── 文档加载与文本切分
│   ├── vibeflow/knowledge_loader.py   扫描 knowledge/ 目录
│   ├── vibeflow/knowledge_chunker.py  文本切分
│   └── vibeflow/knowledge_models.py   TextChunk, SearchResult
│
├── 关键词检索
│   └── vibeflow/knowledge_retriever.py  KeywordRetriever（中文分词 + 英文匹配）
│
├── 向量检索
│   ├── vibeflow/knowledge_embedder.py         嵌入模型封装（sentence-transformers）
│   └── vibeflow/knowledge_vector_retriever.py  VectorRetriever（余弦相似度）
│
├── 混合检索
│   └── vibeflow/knowledge_hybrid_retriever.py  HybridRetriever（加权融合 + 去重）
│
├── Ollama RAG 回答
│   ├── vibeflow/knowledge_llm_client.py         LLMClient 接口 + OllamaClient
│   ├── vibeflow/knowledge_context_builder.py    检索结果 → 带编号上下文
│   ├── vibeflow/knowledge_prompt_builder.py     系统提示词 + 用户消息
│   ├── vibeflow/knowledge_rag_service.py        RAGService 编排全链路
│   ├── vibeflow/knowledge_rag_models.py         RAGResult, SourceInfo
│   ├── vibeflow/knowledge_service.py            组合 Loader → Chunker → Retriever
│   └── vibeflow/ask_cli.py                      交互式问答 CLI
│
├── 引用来源
│   └── [S1] [S2] 标签 + 来源文件列表
│
├── RAG 评估集
│   ├── evaluation/rag_cases.json         18 条黄金案例（8 个类别）
│   ├── vibeflow/evaluation/models.py     EvaluationCase, CaseResult, EvalReport
│   ├── vibeflow/evaluation/scorer.py     确定性评分（不依赖 LLM 评判）
│   ├── vibeflow/evaluation/runner.py     评估运行器 + ScriptedFakeLLM
│   └── vibeflow/evaluate_rag.py          CLI 入口
│
├── 并发压力测试
│   ├── vibeflow/load_test.py             ThreadPoolExecutor 并发压测
│   └── 指标：成功率、吞吐量、P50/P95/P99、异常分布
│
├── 单元测试
│   ├── tests/test_services.py            任务管理测试
│   ├── tests/test_knowledge.py           知识检索测试
│   ├── tests/test_rag.py                 RAG 服务测试
│   ├── tests/test_vector_retriever.py    向量检索测试
│   ├── tests/test_evaluation.py          评估体系测试（68 条）
│   ├── tests/test_load_test.py           压力测试测试（34 条）
│   └── tests/test_safety_guard.py        Hook 安全测试
│
└── Claude Code 工程化体系
    ├── CLAUDE.md                    项目规范与架构约束（AI 自动遵循）
    ├── .claude/skills/unit-test/    单元测试 Skill（/unit-test）
    ├── .claude/agents/test-expert.md      测试专家 Subagent（@test-expert）
    ├── .claude/agents/quality-engineer.md 质量工程师 Subagent（@quality-engineer）
    └── .claude/hooks/safety_guard.py      PreToolUse 安全 Hook（拦截危险命令）
```

## 快速开始

```bash
# 启动桌面 GUI
python -m vibeflow.main

# 运行全部测试（286 条）
pytest
```

## 学习任务管理

桌面 GUI 应用，支持添加、筛选、完成、删除学习任务，SQLite 本地持久化。

| 功能 | 说明 |
|------|------|
| 新建任务 | 填写标题、课程、截止日期 |
| 任务列表 | 查看全部 / 待完成 / 已完成 |
| 状态切换 | 标记完成 / 未完成 |
| 删除任务 | 移除任务记录 |
| 任务统计 | 完成率概览 |

## 知识检索

```bash
# 关键词检索（默认）
python -m vibeflow.search_cli --mode keyword

# 向量语义检索（首次自动下载模型约 420 MB）
python -m vibeflow.search_cli --mode vector

# 混合检索（关键词 + 向量加权融合）
python -m vibeflow.search_cli --mode hybrid
```

三种检索器统一接口 `search(chunks, query, top_k) → list[SearchResult]`，通过 `KnowledgeService(mode=...)` 切换。

## RAG 知识库问答

```bash
# 安装依赖
pip install ollama sentence-transformers

# 启动问答（默认混合检索）
python -m vibeflow.ask_cli

# 指定检索模式
python -m vibeflow.ask_cli --mode keyword
python -m vibeflow.ask_cli --mode hybrid
```

### 处理流程

1. 接收用户问题 → 2. 检索相关文本块 → 3. 判断相关性 → 4. 构造带编号上下文 → 5. 生成受约束提示词 → 6. 调用本地 Ollama 生成回答 → 7. 返回回答 + `[SN]` 引用来源

### 资料不足时的行为

系统在以下情况不会调用模型，避免编造信息：
- 问题为空 → 提示输入问题
- 无匹配结果 → "当前知识库中没有找到足够相关的资料"
- 相关性过低 → "当前知识库无法支持完整回答"

### 配置模型

```bash
export VIBEFLOW_OLLAMA_MODEL=qwen3:latest   # 未设置时自动探测
```

## RAG 评估体系

18 条黄金案例，8 个类别，确定性评分（不依赖 LLM 评判）。

```bash
python -m vibeflow.evaluate_rag                     # Fake 模式（可重复）
python -m vibeflow.evaluate_rag --mode ollama        # 真实 Ollama
python -m vibeflow.evaluate_rag --limit 5            # 限制数量
python -m vibeflow.evaluate_rag --category "资料不足问题"  # 按类别
python -m vibeflow.evaluate_rag --output eval_results/result.json  # 保存报告
```

### 评分维度

| 维度 | 说明 |
|------|------|
| 来源命中 | expected_sources 是否出现在检索结果中 |
| 引用有效 | [SN] 标签是否对应真实来源 |
| 关键词覆盖 | expected_keywords 命中率 ≥ 50% |
| 禁止词 | forbidden_keywords 是否出现 |
| 拒答准确 | 资料不足时是否正确拒答 |

## 并发压力测试

基于 `concurrent.futures`，每个请求独立创建 RAGService，无状态串扰。

```bash
python -m vibeflow.load_test                               # 默认（Fake, 20 请求, 并发 2）
python -m vibeflow.load_test --concurrency 5 --requests 50  # 高并发
python -m vibeflow.load_test --mode ollama --requests 5 --concurrency 1  # Ollama
python -m vibeflow.load_test --error-rate 0.3               # 模拟 30% 异常
python -m vibeflow.load_test --output load_results/report.json  # 保存报告
```

### 指标

成功率、吞吐量、min/avg/P50/P95/P99/max 响应时间、异常分布。

## V1.0 Ollama 评估基线

| 指标 | 数值 |
|------|------|
| 模型 | qwen2.5:1.5b |
| 通过率 | 66.7%（12/18） |
| 来源命中率 | 100.0% |
| 引用有效率 | 100.0% |
| 拒答准确率 | 100.0% |
| 平均响应时间 | 1425.9 ms |

## 技术栈

Python 3.10+ · Tkinter · SQLite · pytest · sentence-transformers · Ollama

## 禁止行为

- 使用云端 API
- 自动化测试依赖真实 Ollama
- eval_results/ 和 load_results/ 纳入版本控制
