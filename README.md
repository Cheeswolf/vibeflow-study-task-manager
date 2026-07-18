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

## 下一阶段建议

1. 增加任务编辑功能
2. 增加逾期任务提示
3. 增加按课程筛选
4. 增加数据导出
5. 建立测试 Skill
6. 建立代码审查 Subagent
7. 添加危险命令 Hook
