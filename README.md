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
python -m vibeflow.search_cli
```

输入中文或英文关键词进行搜索，最多返回 3 条结果，按相关度排序。输入 `/q` 退出。

## 下一阶段建议

1. 增加任务编辑功能
2. 增加逾期任务提示
3. 增加按课程筛选
4. 增加数据导出
5. 建立测试 Skill
6. 建立代码审查 Subagent
7. 添加危险命令 Hook
