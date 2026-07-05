# experiment_tracker — 可复用实验追踪

基于 SwanLab 的实验追踪模块，一个 `ExperimentTracker` 搞定。
复用到 BLIP / LLaVA 等项目时，把整个 `experiment_tracker/` 目录复制过去即可。

## 快速开始

```python
from experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(
    project="CLIP",                     # 项目名
    config={"lr": 1e-3, "epochs": 5},   # 超参会记录到面板
)

for epoch in range(5):
    tracker.log({"train_loss": 0.5, "valid_loss": 0.4, "epoch": epoch})

tracker.finish()
```

## 配置 API Key（三选一）

**方式 1：secrets.py（推荐）**  
编辑 `experiment_tracker/secrets.py`，填入 key：
```python
SWANLAB_API_KEY = "your-swanlab-key"
```
该文件已 gitignore，不会泄露。

**方式 2：环境变量**
```bash
export SWANLAB_API_KEY="your-swanlab-key"
```

**方式 3：命令行登录**
```bash
swanlab login
```

## 训练完如何查看结果

1. 打开 [https://swanlab.cn](https://swanlab.cn) 并登录
2. 训练时 swanlab 会自动同步实验数据
3. 在项目面板中把实验设为"公开"，即可分享链接给学生

## 目录结构

```
experiment_tracker/
├── __init__.py    # 导出 ExperimentTracker
├── tracker.py     # 核心：init / log / finish
├── secrets.py     # API Key（gitignore，不会上传）
└── README.md      # 本文件
```
