良好的开发习惯应该是什么样子的

核心就一条：**每个项目用自己独立的虚拟环境，互不污染**。

## 日常操作模式

```powershell
# 进入项目
cd xxx-project

# 激活这个项目的 venv（Windows PowerShell）
.venv\Scripts\activate

# 现在所有 python/pip/pytest 都自动走这个项目的环境
python -m vibeflow.main
pip install xxx
pytest
```

## 三个原则

|原则|做法|为什么|
|---|---|---|
|**环境隔离**|每个项目建 `.venv`，不往全局装包|项目 A 用 pytest 8、项目 B 用 pytest 9，互不打架|
|**依赖声明**|把依赖写进 `requirements.txt`|换了电脑或给别人，一条命令就能还原|
|**用前激活**|进项目就 `.venv\Scripts\activate`|避免装错环境、版本混乱|

## 不要做的事

- ❌ 到处 `pip install` 到全局 Python
- ❌ 复制整个 `.venv` 文件夹来"分享"环境
- ❌ 把 `.venv` 提交到 git