# 第八节课：Agent 进阶 —— 经典范式构建与项目实战

## 课程概述

本节课基于 Datawhale [hello-agents](https://github.com/datawhalechina/hello-agents) 开源项目（55K+ stars），从智能体三大经典范式出发，涵盖框架设计、记忆检索、上下文工程、通信协议，再到三个完整项目实战。

**学习跨度**：hello-agents 第 4~15 章（跳过第 11 章 Agentic RL），前三天自习 → 第四天直播 → 后三天三个项目。

---

## 目录结构

```
第八节课/
├── 讲义.md                        ← 主讲义（Agent 概念 + Loop Engineering 等）
├── LangChain与LangGraph入门指南.md  ← 扩展阅读：openai 库 + LangChain + LangGraph
├── README.md                      ← 本文件
└── hello-agents/                  ← Datawhale 开源项目（已克隆）
    ├── docs/                      ← 完整教程文档
    │   ├── chapter4/              ← 经典范式构建（核心实战）
    │   ├── chapter5/              ← 低代码平台
    │   ├── chapter6/              ← 框架实践
    │   ├── chapter7/              ← 构筑Agent框架
    │   ├── chapter8/              ← 记忆与检索
    │   ├── chapter9/              ← 上下文工程
    │   ├── chapter10/             ← 通信协议
    │   ├── chapter12/             ← 能力评估
    │   ├── chapter13/             ← 项目一：智能旅行助手
    │   ├── chapter14/             ← 项目二：深度研究Agent
    │   └── chapter15/             ← 项目三：赛博小镇
    └── code/                      ← 所有可运行代码
        ├── chapter4/              ← 三种范式（ReAct/PlanSolve/Reflection）
        ├── chapter7/              ← 框架测试
        ├── chapter8/              ← Memory+RAG示例
        ├── chapter9/              ← 上下文工程示例
        ├── chapter10/             ← 通信协议示例
        ├── chapter12/             ← 评估示例
        ├── chapter13/             ← 旅行助手（Vue3+FastAPI）
        ├── chapter14/             ← 深度研究（SSE流式）
        └── chapter15/             ← 赛博小镇（Godot引擎）
```

---

## 学习节奏

| 时间 | 学习方式 | 内容 |
| --- | --- | --- |
| 第 1 天 | 自习 | 经典范式（ReAct/PlanSolve/Reflection）+ 源码注解 |
| 第 2 天 | 自习 | 低代码平台 + 框架实践 + 自研框架 |
| 第 3 天 | 自习 | 记忆检索 + 上下文工程 + 通信协议 + 能力评估 |
| **第 4 天** | **直播** | **串讲全部章节 + 源码精读 + 三个项目启动指导** |
| 第 5 天 | 项目 | 项目一：智能旅行助手（多Agent协作+地图可视化） |
| 第 6 天 | 项目 | 项目二：深度研究Agent（TODO驱动三阶段+SSE） |
| 第 7 天 | 项目 | 项目三：赛博小镇（Godot+记忆系统+好感度） |

---

## 环境配置

### 通用依赖

```bash
pip install openai python-dotenv google-search-results
```

### .env 配置（基础部分）

```bash
LLM_MODEL_ID=Qwen/Qwen2.5-32B-Instruct
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.siliconflow.cn/v1/
SERPAPI_API_KEY=your_serpapi_key
```

### 运行经典范式（第 4 章）

```bash
cd hello-agents/code/chapter4
python llm_client.py        # 测试 LLM
python tools.py             # 测试工具
python ReAct.py             # ReAct
python Plan_and_solve.py    # Plan-and-Solve
python Reflection.py        # Reflection
```

### 运行项目

```bash
# 旅行助手
cd hello-agents/code/chapter13/helloagents-trip-planner/backend
pip install -r requirements.txt && python run.py
cd ../frontend && npm install && npm run dev

# 深度研究
cd hello-agents/code/chapter14/helloagents-deepresearch/backend
pip install -e . && python src/main.py
cd ../frontend && npm install && npm run dev

# 赛博小镇
cd hello-agents/code/chapter15/Helloagents-AI-Town/backend
pip install -r requirements.txt && python main.py
# Godot 4.2+ 导入 helloagents-ai-town/ → F5 运行
```

---

## 扩展阅读

- **[LangChain与LangGraph入门指南.md](./LangChain与LangGraph入门指南.md)**：通俗讲解 `openai` 库、LangChain / LangGraph 基础组件与调用方式，以及如何用 AI 辅助开发（推荐实习前或第八周自习阅读）

---

## 备注

- 本课程基于 Datawhale hello-agents（CC BY-NC-SA 4.0）项目
- 跳过原第 11 章（Agentic RL / 强化学习），后续课程单独补充
- 讲义从 hello-agents 第 4 章开始，前 3 章为基础内容已在前四节课覆盖
