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
