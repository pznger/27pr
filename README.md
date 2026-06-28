# 零基础深度学习直通大模型

> 中科大学长 + 浙大学长联合主讲 | 面向 2028 届考研/保研学生
>
> 从四非本科逆袭华五的实战经验，手把手带你从零到能写进简历的硬核项目。

---

## 关于我们

这门课由 **中国科学技术大学学长** 与 **浙江大学学长** 联合设计和主讲。

我们都是从普通本科（四非）一路杀到华五的人。不是站在岸上教你怎么游泳的教练，而是真的从水里爬上来、知道哪一步会呛水的那个人。

### 学长的背景

**中科大学长** — 中国科学技术大学计算机硕士在读
- 四非考研上岸中科大，总分 390+
- 研究方向：多模态大模型、知识图谱、推荐系统
- 擅长：LLM 应用开发、知识增强系统、智能 Agent 项目设计

**浙大学长** — 浙江大学人工智能方向硕士在读
- 四非保研上岸浙大 Top 实验室 & Top 导师，本科期间发表 SCI 二区论文
- 研究方向：具身智能（VLN、VLA）、多模态大模型（GRPO、Agentic RL）、视觉智能
- 擅长：机器人基础模型、大模型强化学习、科研项目打磨

### 过往带教成绩

我们的早鸟科研项目班已经帮助多名 **2027 届保研学生**取得实打实的成果：

- 辅导的多名 27 保研学生，**均成功套磁到华五 / 计算机强 985 高校**（浙大软件学院、复旦工研院、电子科技大学等）的强组科研实习
- 多位导师已口头承诺留名额
- 与上岸同学保持长期联系，持续指导科研方向，**优秀同学直接内推组里科研实习**

---

## 这门课和我们的核心项目班的关系

我们的核心早鸟班在做的是工业级科研项目：

| 项目方向 | 核心技术 | 最终产出 |
| --- | --- | --- |
| **GraphRAG 408 智能答疑 Agent** | Neo4j 知识图谱、混合检索（BM25+HNSW）、LangChain ReAct、自修复流水线 | 可写入简历的工业级 Agent + 面经 |
| **具身智能 VLA 模型训练** | VLM/VLA、Flow Matching、LoRA 微调、仿真环境 | 可展示的机器人操作 Demo |
| **RL 驱动的 PPT 生成 Agent** | ViT、Function Calling、强化学习（GSPO/GRPO/PPO） | 进阶科研项目 |

但做这些项目的前提是——**你得先会写 Python、懂深度学习、能看懂大模型代码**。

这门课就是所有高阶项目的 **「必经之路」**。八周基础打牢之后，核心项目班随时欢迎你。

---

## 课程定位

- **面向谁**：目标 2028 年考研/保研的计算机相关专业本科生
- **前置要求**：真正零基础。会开机、会打字就能开始
- **课程周期**：8 周（第 0 周课前准备 + 第 1~4 周正课 + 后续待排）
- **上课形式**：每周 1 天直播 + 6 天自习任务
- **最终目标**：能独立阅读深度学习代码、理解大模型原理、动手搭建 RAG 和 Agent

---

## 目录结构

```
├── 第零节课/
│   └── 讲义.md              # 课前说明 + 环境准备
├── 第一节课/
│   ├── 讲义.md              # 完整讲义（含图片，相对路径 images/）
│   ├── images/              # 讲义配图（29 张）
│   └── 项目/
│       ├── graduate_admission.csv      # 固定数据集（800 条，正样本约 35%）
│       ├── graduate_admission_mlp.py   # PyTorch 综合小项目
│       ├── requirements.txt
│       └── README.md
├── 第三节课/
│   ├── 讲义.md              # CLIP 论文精读讲义
│   ├── CLIP_paper.pdf       # CLIP 论文原文（Radford et al., 2021）
│   └── 项目/
│       ├── clip_retrieval.py     # CLIP 零样本图文检索
│       ├── clip_train.py         # 对比学习训练模块（InfoNCE + 投影头）
│       ├── web_demo.py           # Streamlit Web 可视化界面
│       ├── requirements.txt
│       └── README.md
├── 第四节课/
│   ├── 讲义.md              # RAG & Agent 双项目讲义（逐行代码精读）
│   ├── README.md            # 双项目说明
│   ├── images/              # 讲义配图
│   ├── happy-llm/           # Datawhale happy-llm 完整仓库
│   ├── 项目一-TinyRAG扩展/   # 扩展版 RAG（语义分块+BM25+融合检索）
│   └── 项目二-TinyAgent扩展/ # 扩展版 Agent（15个工具）
└── README.md                # 本文件
```

## 课程节奏

| 周次 | 直播主题 | 自习重点 |
| --- | --- | --- |
| **第 0 周** | **课前准备** | 安装 VSCode + Typora + Anaconda3、解决梯子/网络问题、注册硅基流动 + AutoDL + DeepSeek、安装 Cursor（建议闲鱼买 Pro 账号）、闲鱼买谷歌账号 |
| 第 1 周 | 深度学习基础 → MLP | 环境配置、刘二大人 PyTorch 前 4 集、小项目（保研资格预测）、自学 CNN |
| 第 2 周 | Transformer | 另有安排 |
| 第 3 周 | **CLIP 论文精读** | 论文原文、对比学习双塔架构、InfoNCE 训练演示、Flickr8k 零样本图文检索 |
| 第 4 周 | **RAG & Agent & happy-llm** | Tiny-RAG 知识库问答 + Tiny-Agent 工具调用双项目实战 |

*第 5~8 周课程内容待更新。*

---

## 第零节课内容概览

> 课前准备篇，不涉及代码——纯环境搭建和平台注册。

1. **课程说明**：为什么现在开始、导师团队与过往成绩、与核心项目班的关系
2. **VSCode + Python 环境**：代码编辑器与插件安装
3. **Typora**：Markdown 阅读器（看讲义必备）
4. **Anaconda3**：Python 虚拟环境管理
5. **网络环境**：科学上网 / 镜像源配置（解决 GitHub/HuggingFace 访问问题）
6. **硅基流动（SiliconFlow）**：大模型 API 平台注册
7. **AutoDL**：GPU 云服务器注册
8. **Cursor / Claude Code**：AI 编程助手安装
9. **账号省钱攻略**：闲鱼买谷歌账号 + Cursor Pro + DeepSeek 充值
10. **心态准备**：怎么学、怎么用 AI、八周后能做到什么

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
7. **PyTorch 实战项目**（Flickr8k + CLIP 图文检索 + 对比学习训练 + Web 界面）

## 第四节课内容概览

> 核心项目：[happy-llm](https://github.com/datawhalechina/happy-llm)（Datawhale，31K+ stars）

1. **共用基础**：Embedding 技术演进 + Prompt 工程（System/User Prompt、CoT、Few-shot）
2. **项目一 Tiny-RAG**：RAG 五步流程 → 源码逐文件精读（utils/Embeddings/VectorBase/LLM）→ 进阶优化（语义分块、重排序、融合检索）
3. **项目二 Tiny-Agent**：Agent ReAct 循环 → Function Calling → 源码逐文件精读（utils/tools/core）→ 自定义工具实验

---

## 快速开始

### 第零节课（先做这个！）

用 Typora 打开 `第零节课/讲义.md`，按清单逐项完成环境准备。全部搞定后再开始第一节课。

### 第一节课

1. 用 Typora 打开 `第一节课/讲义.md`
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

# 训练模块
python clip_train.py

# Web 界面
streamlit run web_demo.py
```

### 第四节课

1. 阅读 `第四节课/讲义.md` 了解 Embedding→Agent→RAG 全链路 + 双项目源码解读
2. 项目一（Tiny-RAG）进入 `第四节课/happy-llm/docs/chapter7/RAG/` 运行：

```bash
cp .env_example .env      # 填入硅基流动 API Key
pip install -r requirements.txt
python demo.py
```

3. 项目二（Tiny-Agent）进入 `第四节课/happy-llm/docs/chapter7/Agent/` 运行：

```bash
pip install -r requirements.txt
python demo.py
# 或网页版：streamlit run web_demo.py
```

4. 完成基础实验后，尝试扩展项目：
   - `第四节课/项目一-TinyRAG扩展/` — 语义分块 + BM25 + RRF 融合检索
   - `第四节课/项目二-TinyAgent扩展/` — 新增计算器、文件读写等 8 个工具

---

## 我们的教学风格

不是「丢一堆论文和代码让学生自己摸索」。课程提供：

- ✅ 系统化的课程讲义和学习计划表——每周清楚知道该做什么
- ✅ 逐行注释的项目代码——能跑、能改、能讲清楚每行在做什么
- ✅ 1V1 答疑——卡住了随时问，不让你一个人对着报错发呆
- ✅ 简历表达指导——做完项目怎么写到简历上、面试怎么讲

最终目标不是让你「听过一个概念」，而是让你真正拥有一个**能跑、能改、能讲、能写进简历**的项目经历。

---

## 八周后你能做什么

- 独立阅读并运行深度学习项目代码
- 理解大模型（LLM）的基本原理和应用方式
- 动手搭建自己的 RAG 知识库和 AI Agent
- 看懂 CLIP、Transformer 等经典论文的核心思路
- **具备进入核心项目班的基础能力**——上手 GraphRAG Agent、具身 VLA 等工业级项目

---

## 后续课程（待更新）

- 第二节课：Transformer（另有安排）
- 第 5~8 周：待定

---

> 中科大学长 × 浙大学长 | 从双非到华五 | 2028 早鸟科研
