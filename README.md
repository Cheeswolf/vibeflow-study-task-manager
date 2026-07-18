# VibeFlow 学习任务管理器

一个用于练习 Vibe Coding 工程化开发流程的 Python 桌面项目。

## 当前功能

- 添加学习任务
- 填写所属课程和截止日期
- 查看全部、待完成、已完成任务
- 切换完成状态
- 删除任务
- 查看任务统计
- SQLite 本地持久化
- 本地 Markdown / TXT 知识库检索
- 命令行交互式搜索
- 基础单元测试

## 运行环境

- Python 3.10 或更高版本
- Tkinter（通常随 Python 自带）
- pytest（仅测试需要）

## 启动项目

```bash
python -m vibeflow.main
```

## 运行测试

```bash
python -m pip install -r requirements-dev.txt
pytest
```

## 推荐 Git 流程

```bash
git init
git add .
git commit -m "feat: initialize VibeFlow MVP"
git branch -M main
```

开发新功能时：

```bash
git checkout -b feat/task-editing
```

功能完成并测试后：

```bash
git add .
git commit -m "feat: add task editing"
git checkout main
git merge feat/task-editing
```

## 知识检索

将 Markdown（`.md`）或纯文本（`.txt`）文件放入 `knowledge/` 目录，然后启动交互式检索：

```bash
# 关键词检索（默认，无需额外依赖）
python -m vibeflow.search_cli --mode keyword

# 向量语义检索（首次运行自动下载嵌入模型约 420 MB）
python -m vibeflow.search_cli --mode vector

# 混合检索（综合关键词 + 向量）
python -m vibeflow.search_cli --mode hybrid
```

- **keyword**：中文分词 + 英文单词匹配，按关键词匹配度评分
- **vector**：使用多语言嵌入模型进行语义相似度检索
- **hybrid**：加权融合关键词得分和向量相似度得分（默认 keyword=0.3, vector=0.7）

输入中文或英文关键词进行搜索，最多返回 3 条结果，按相关度排序。输入 `/q` 退出。

### 安装向量检索依赖

```bash
pip install sentence-transformers
```

## 知识库问答（RAG）

将 Markdown / TXT 文件放入 `knowledge/` 目录，启动交互式问答：

```bash
# 安装依赖
pip install ollama

# 确认 Ollama 正在运行
ollama list

# 启动问答 CLI（默认使用混合检索）
python -m vibeflow.ask_cli

# 指定检索模式
python -m vibeflow.ask_cli --mode keyword
python -m vibeflow.ask_cli --mode vector
python -m vibeflow.ask_cli --mode hybrid
```

输入问题后，系统自动检索知识库，调用本地 Ollama 生成基于检索资料、带引用来源的回答。

### RAG 回答流程

1. 接收用户问题
2. 使用检索器获取相关文本块
3. 判断检索结果是否达到最低相关性要求
4. 将文本块整理成带编号的上下文
5. 构造受约束的提示词（只能依据知识库资料回答）
6. 调用本地 Ollama 大模型生成回答
7. 返回回答和引用来源

### 配置模型

```bash
# 设置模型名称（Windows CMD）
set VIBEFLOW_OLLAMA_MODEL=qwen3:latest

# 或 PowerShell
$env:VIBEFLOW_OLLAMA_MODEL="qwen3:latest"

# 或 Bash
export VIBEFLOW_OLLAMA_MODEL=qwen3:latest
```

如未设置，系统会自动探测已安装的模型。

### 确认 Ollama 正在运行

```bash
# 检查 Ollama 服务
ollama list

# 如未运行，启动服务
ollama serve
```

### 资料不足时的行为

系统在以下情况不会调用模型：
- 问题为空 → 提示输入问题
- 知识库无匹配结果 → 「当前知识库中没有找到足够相关的资料」
- 检索结果相关性过低 → 「当前知识库无法支持完整回答」

这避免了模型在没有可靠资料时编造信息。

### 当前局限

- 不支持跨文档推理（每次只基于单个文本块判断相关性）
- 上下文窗口受限于文本块数量和字符数（最多 5 块，总计约 3000 字符）
- 不精确计算 token 数，以字符数近似控制
- 需要本地运行 Ollama，首次使用需下载模型
- 不支持多轮对话，每次提问独立处理

## RAG 评估体系

项目包含一套可重复运行的黄金评估集，用于系统性地验证 RAG 回答质量。

### 评估集

评估案例文件位于 `evaluation/rag_cases.json`，目前包含 **18 条案例**，覆盖 8 个类别：

| 类别 | 说明 |
|------|------|
| 精确关键词问题 | 知识库中存在精确匹配的关键词 |
| 同义表达问题 | 用不同表达方式描述同一概念 |
| 多文档综合问题 | 需要跨文档的信息 |
| 资料不足问题 | 知识库中没有相关内容，应拒答 |
| 引用正确性问题 | 验证 [SN] 标签对应真实来源 |
| 提示词注入问题 | 问题中嵌入恶意指令 |
| 空输入或无效输入 | 空字符串、纯空白 |
| 中英文混合问题 | 中英文混合查询 |

### 评估指标

每条案例评估以下维度：

- **来源命中**：预期来源文件是否出现在检索结果中
- **引用有效**：回答中的 `[SN]` 是否对应实际提供的来源
- **关键词覆盖**：预期关键词在回答中的命中率
- **禁止词检查**：回答是否出现不应有的内容
- **拒答准确**：资料不足时是否正确拒答

评估总报告包含：通过率、来源命中率、引用有效率、拒答准确率、平均关键词覆盖率、模型调用错误率、平均/P50/P95 响应时间。

### 运行评估

```bash
# Fake 模式（默认，不连接真实模型，结果可重复）
python -m vibeflow.evaluate_rag

# Fake 模式 + 限制案例数
python -m vibeflow.evaluate_rag --limit 5

# 按类别过滤
python -m vibeflow.evaluate_rag --category "资料不足问题"

# 保存 JSON 报告
python -m vibeflow.evaluate_rag --output eval_results/result.json

# 真实 Ollama 模式（需 Ollama 运行中）
python -m vibeflow.evaluate_rag --mode ollama
```

### 查看失败案例

终端输出会列出每条失败案例的 `case_id`、问题和具体失败原因。JSON 报告（`--output`）包含完整明细。

### 评估方法的局限

- Fake 模式下 LLM 返回预设回答，无法评估真实生成质量
- 关键词覆盖使用简单字符串匹配，不理解同义词
- 来源命中依赖文件名匹配，不检查内容语义是否正确
- 拒答判断依赖固定文案标记（「没有找到」「相关度过低」等）
- 仅在 keyword 检索模式下测试，vector/hybrid 需加载真实模型

## 并发压力测试

一个基于 Python 标准库 `concurrent.futures` 的压力测试工具，不依赖外部服务框架。

### 运行压力测试

```bash
# Fake 模式（默认，20 请求，并发 2）
python -m vibeflow.load_test

# 自定义并发参数
python -m vibeflow.load_test --concurrency 5 --requests 50

# 设置单请求超时
python -m vibeflow.load_test --timeout 10

# 本地 Ollama 小规模压力（需显式开启）
python -m vibeflow.load_test --mode ollama --requests 5 --concurrency 1

# 模拟 30% 模型异常率
python -m vibeflow.load_test --error-rate 0.3

# 保存 JSON 报告
python -m vibeflow.load_test --output load_results/report.json
```

### 压力测试指标

| 指标 | 说明 |
|------|------|
| 成功率 | 成功请求 / 总请求 |
| 吞吐量 | 每秒完成的请求数 |
| 最小/平均/最大 | 响应时间的最值 |
| P50 | 50% 请求的响应时间不超过此值 |
| P95 | 95% 请求的响应时间不超过此值 |
| P99 | 99% 请求的响应时间不超过此值 |
| 异常分布 | 各类型异常的出现次数 |

### 设计说明

- **每个请求创建独立的 RAGService**，避免状态串扰（来源编号、上下文、错误状态）
- Fake 模式下不访问网络，可在 pytest 中安全执行
- Ollama 模式仅限手动 CLI 运行，pytest 中禁止
- 单个请求异常不会导致整个压力测试中断

## 下一阶段建议

1. 增加任务编辑功能
2. 增加逾期任务提示
3. 增加按课程筛选
4. 增加数据导出
5. 建立测试 Skill
6. 建立代码审查 Subagent
7. 添加危险命令 Hook
