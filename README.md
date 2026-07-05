# 零基础深度学习直通大模型

面向考研、保研学生的系统课程，从零基础到能读懂大模型相关代码与论文。

## 目录结构

```
├── 第零节课/
│   └── 讲义.md              # 课前说明 + 环境准备
├── 第一节课/
│   ├── 讲义.md              # 完整讲义（含图片，相对路径 images/）
│   ├── images/              # 讲义配图（29 张）
│   └── 项目/
│       ├── graduate_admission.csv      # 固定数据集（800 条，正样本约 35%）
│       ├── generate_dataset_csv.py     # 重新生成 CSV（一般无需运行）
│       ├── outputs/                    # 运行后生成的训练曲线与评估图
│       ├── graduate_admission_mlp.py   # PyTorch 综合小项目
│       ├── requirements.txt
│       └── README.md
├── 第三节课/
│   ├── 讲义.md              # CLIP 论文精读讲义
│   ├── CLIP_paper.pdf       # CLIP 论文原文（Radford et al., 2021）
│   └── 项目/
│       ├── clip_retrieval.py          # CLIP 零样本图文检索项目
│       ├── requirements.txt
│       ├── outputs/                    # 检索结果拼接图
│       ├── data/                       # Flickr8k 数据集（首次运行下载）
│       ├── cache/                      # 图像嵌入缓存（首次编码生成）
│       └── README.md
├── 第四节课/
│   ├── 讲义.md              # RAG & Agent 双项目讲义（逐行代码精读）
│   ├── README.md            # 双项目说明
│   ├── images/              # 讲义配图
│   ├── happy-llm/           # Datawhale happy-llm 完整仓库（git clone）
│   │   └── docs/chapter7/
│   │       ├── RAG/         # 原始 Tiny-RAG
│   │       └── Agent/       # 原始 Tiny-Agent
│   ├── 项目一-TinyRAG扩展/   # 扩展版 RAG（语义分块+BM25+融合检索）
│   └── 项目二-TinyAgent扩展/ # 扩展版 Agent（15个工具）
├── 第八节课/
│   ├── 讲义.md              # Agent进阶讲义（hello-agents第4~15章全文 ~22000行 + 源码注解）
│   ├── README.md            # 第八节课说明
│   └── hello-agents/        # Datawhale hello-agents 完整仓库（git clone）
│       ├── docs/
│       │   ├── chapter4/    # 经典范式构建（源码精读）
│       │   ├── chapter5/    # 低代码平台
│       │   ├── chapter6/    # 框架实践
│       │   ├── chapter7/    # 自研Agent框架
│       │   ├── chapter8/    # 记忆与检索
│       │   ├── chapter9/    # 上下文工程
│       │   ├── chapter10/   # 通信协议
│       │   ├── chapter12/   # 能力评估
│       │   ├── chapter13/   # 项目一：智能旅行助手
│       │   ├── chapter14/   # 项目二：深度研究Agent
│       │   └── chapter15/   # 项目三：赛博小镇
│       └── code/            # 所有可运行代码
│           ├── chapter4/    # 三种范式（ReAct/PlanSolve/Reflection）
│           ├── chapter7/    # 框架测试
│           ├── chapter8/    # Memory+RAG示例
│           ├── chapter9/    # 上下文工程示例
│           ├── chapter10/   # 通信协议示例
│           ├── chapter12/   # 评估示例
│           ├── chapter13/   # 旅行助手（Vue3+FastAPI）
│           ├── chapter14/   # 深度研究（SSE流式）
│           └── chapter15/   # 赛博小镇（Godot引擎）
└── README.md                # 本文件
```

## 课程节奏

每一周包含 **1 天直播上课 + 6 天自习任务**。详细计划表见各周讲义中的「学习计划表」。

| 周次 | 直播主题 | 自习重点 |
| --- | --- | --- |
| **第 0 周** | **课前准备** | 安装 VSCode + Typora + Anaconda3、解决梯子/网络问题、注册硅基流动 + AutoDL、安装 Cursor / Claude Code |
| 第 1 周 | 深度学习基础 → MLP | 环境配置、Cursor、刘二大人 PyTorch 前 4 集、小项目、自学 CNN |
| 第 2 周 | Transformer | 另有安排 |
| 第 3 周 | **CLIP 论文精读** | 论文原文、对比学习双塔架构、Flickr8k 零样本图文检索项目 |
| 第 4 周 | **RAG & Agent & happy-llm** | Tiny-RAG 知识库问答 + Tiny-Agent 工具调用双项目实战 |
| 第 8 周 | **Agent 进阶：全链路实战** | 经典范式 + 自研框架 + 记忆/上下文/协议 + 三个完整项目（旅行/研究/游戏） |

## 第零节课内容概览

> 课前准备，不涉及代码——纯环境搭建和平台注册。

1. **课程说明**：为什么现在开始、后续还有同学可能加入
2. **VSCode + Python 环境**：代码编辑器与插件安装
3. **Typora**：Markdown 阅读器（看讲义必备）
4. **Anaconda3**：Python 虚拟环境管理
5. **网络环境**：科学上网 / 镜像源配置（解决 GitHub/HuggingFace 访问问题）
6. **硅基流动（SiliconFlow）**：大模型 API 平台注册
7. **AutoDL**：GPU 云服务器注册
8. **Cursor / Claude Code**：AI 编程助手安装
8. **心态准备**：怎么学、怎么用 AI、八周后能做到什么

## 第一节课内容概览

1. **机器学习与深度学习基础概念**
2. **数据集划分与评估指标**（Accuracy、F1、MSE、AUC 等）
3. **线性回归**（解析解 + MSE 损失）
4. **梯度下降法**（方向导数、学习率、SGD/Mini-batch）
5. **反向传播**（计算图、链式法则、手推示例）
6. **激活函数**（Sigmoid、Tanh、ReLU、Leaky ReLU、GELU、Softmax）
7. **线性分类与逻辑回归**
8. **交叉熵损失函数**（从熵到 Softmax 梯度）
9. **多层感知机 MLP**（完整手算前向/反向传播例题）
10. **PyTorch 实战项目**（保研资格预测）

## 第三节课内容概览

> 论文原文见 `第三节课/CLIP_paper.pdf`

1. **论文背景与动机**（传统 CV 局限 vs 语言监督）
2. **对比式预训练**（双塔架构、对称 InfoNCE 损失、CLIP 伪代码）
3. **模型结构**（ViT/ResNet 图像编码器 + Transformer 文本编码器）
4. **零样本迁移机制**（分类→token 匹配、Prompt 工程）
5. **实验设计与分析**（Zero-Shot、Linear Probe、Few-Shot）
6. **论文局限与影响**（OOD、Prompt 敏感、数据偏见）
7. **PyTorch 实战项目**（Flickr8k + CLIP 零样本图文检索）

## 第四节课内容概览

> 核心项目：[happy-llm](https://github.com/datawhalechina/happy-llm)（Datawhale，31K+ stars）

1. **共用基础**：Embedding 技术演进 + Prompt 工程（System/User Prompt、CoT、Few-shot）
2. **项目一 Tiny-RAG**：RAG 五步流程 → 源码逐文件精读（utils/Embeddings/VectorBase/LLM）→ 进阶优化（语义分块、重排序、融合检索）
3. **项目二 Tiny-Agent**：Agent ReAct 循环 → Function Calling → 源码逐文件精读（utils/tools/core）→ 自定义工具实验

## 第八节课内容概览

> 核心项目：[hello-agents](https://github.com/datawhalechina/hello-agents)（Datawhale，55K+ stars）  
> 覆盖章节：第 4~15 章（跳过原第 11 章 Agentic RL）  
> 学习节奏：前三天自习（4~12章）→ 第四天直播串讲 → 后三天三个项目（13~15章）

1. **经典范式三剑客**（第 4 章，源码精读）：ReAct / Plan-and-Solve / Reflection
2. **低代码平台速览**（第 5 章）：Coze / Dify / FastGPT / n8n 定位与对比
3. **框架开发实践**（第 6 章）：AutoGen / AgentScope / CAMEL / LangGraph
4. **自研 Agent 框架**（第 7 章）：HelloAgents 设计哲学——"一切皆为 Tool"
5. **记忆与检索**（第 8 章）：四层 Memory 架构 + RAG Tool
6. **上下文工程**（第 9 章）：从 Prompt Engineering 到 Context Engineering 的范式升级
7. **通信协议**（第 10 章）：MCP（Agent↔工具）、A2A（Agent↔Agent）、ANP（Agent 网络）
8. **能力评估**（第 12 章）：BFCL / GAIA 等主流 Benchmark
9. **项目一：智能旅行助手**（第 13 章）：多 Agent 协作 + 高德地图 + Vue3 前端
10. **项目二：深度研究 Agent**（第 14 章）：TODO 驱动三阶段 + SSE 流式传输
11. **项目三：赛博小镇**（第 15 章）：Godot 游戏引擎 + NPC 记忆 + 好感度系统

## 快速开始

### 第一节课

1. 用 VS Code / Typora / Cursor 打开 `第一节课/讲义.md`
2. 进入 `第一节课/项目/` 运行：

```bash
pip install -r requirements.txt
python graduate_admission_mlp.py
```

### 第三节课

1. 阅读 `第三节课/讲义.md` 或直接打开 `第三节课/CLIP_paper.pdf` 对照
2. 进入 `第三节课/项目/` 运行：

```bash
pip install -r requirements.txt
python clip_retrieval.py "a boy in a red shirt"
```

### 第四节课

1. 阅读 `第四节课/讲义.md` 了解 Embedding→Agent→RAG 全链路 + 双项目源码解读
2. 项目一（Tiny-RAG）进入 `第四节课/happy-llm/docs/chapter7/RAG/` 运行：
```bash
# 1. 复制 .env 并填入 API Key
cp .env_example .env
# 2. 安装依赖
pip install -r requirements.txt
# 3. 放入文档到 data/ 并运行
python demo.py
```
3. 项目二（Tiny-Agent）进入 `第四节课/happy-llm/docs/chapter7/Agent/` 运行：
```bash
# 1. 修改 demo.py 中的 api_key
# 2. 安装依赖
pip install -r requirements.txt
# 3. 运行
python demo.py
# 或网页版：streamlit run web_demo.py
```

### 第八节课

1. 阅读 `第八节课/讲义.md` 了解 Agent 全链路知识（第 2~15 章）
2. **经典范式**（第 4 章）进入 `hello-agents/code/chapter4/`：
```bash
# 1. 创建 .env 并填入 API Key（LLM + SerpApi）
# 2. 安装依赖
pip install openai python-dotenv google-search-results

# 3. 运行三种范式
python llm_client.py        # 测试 LLM
python ReAct.py             # ReAct
python Plan_and_solve.py    # Plan-and-Solve
python Reflection.py        # Reflection
```

3. **项目一：旅行助手**（第 13 章）：
```bash
cd hello-agents/code/chapter13/helloagents-trip-planner/backend
pip install -r requirements.txt
# 配置 .env（LLM + 高德地图Key + Unsplash Key）
python run.py
cd ../frontend && npm install && npm run dev
```

4. **项目二：深度研究**（第 14 章）：
```bash
cd hello-agents/code/chapter14/helloagents-deepresearch/backend
pip install -e .
# 配置 .env（LLM + Search API）
python src/main.py
cd ../frontend && npm install && npm run dev
```

5. **项目三：赛博小镇**（第 15 章）：
```bash
cd hello-agents/code/chapter15/Helloagents-AI-Town/backend
pip install -r requirements.txt
# 配置 .env（LLM API Key）
python main.py
# Godot 4.2+ 导入 helloagents-ai-town/ → F5 运行
```

## 后续课程（待更新）

- 第二节课：Transformer（另有安排）
- 第五节课：大模型微调
- 第六节课：VLA 技术
- 第七节课：流匹配技术

---

> 讲义基于 CLIP 原论文（Radford et al., PMLR 2021）编写。
