# CLIP 零样本图文检索 — 第三节课 PyTorch 小项目

> **建议完成时间**：第三周第 6 天（直播后）  
> **前置条件**：已完成前两节课 PyTorch 基础，已阅读 CLIP 讲义一～四节

本项目包含 **零样本检索** 和 **对比学习训练** 两大模块：

- **`clip_retrieval.py`**：推理/检索（用预训练 CLIP 查图，无需训练）
- **`clip_train.py`**：训练演示（冻结 CLIP 编码器，训练轻量投影头，演示 InfoNCE 对比学习）
- **`web_demo.py`**：Web 可视化界面（Streamlit 交互式检索）

## 什么是图文检索？

```
┌─────────────┐        CLIP 编码       ┌──────────────┐
│  输入文本    │  ──────────────────→  │  文本向量 [512]│
│ "a dog..."  │                        │  cosine sim   │
└─────────────┘                        │      ⇅        │
                                       │  图像向量 [512]│
┌─────────────┐        CLIP 编码       │               │
│  图片库      │  ──────────────────→  └──────────────┘
│  8091 张     │
└─────────────┘

文本向量与 8091 个图像向量逐一算 cosine similarity，
排序后取 Top-K 即为检索结果。全程无需微调 CLIP。
```

## 对应讲义章节

| 代码模块 | 讲义内容 |
| --- | --- |
| `_ensure_dataset()` | 对比学习需要成对图文数据 |
| `CLIPRetriever` / `encode_images` / `encode_texts` | 三、图像编码器 / 文本编码器 |
| `retrieve()` | 二、核心架构（cosine similarity）|
| `make_image_grid()` | 五、实验可视化 |
| `main()` 零样本检索 | 四、零样本迁移机制 |
| 双向检索验证 | 二、对称 InfoNCE 损失含义 |
| `clip_train.py` 全流程 | 二、对比学习（InfoNCE 详细推导 + 训练演示）|

## 环境准备

```bash
pip install -r requirements.txt
```

## 运行

```bash
# 零样本检索（命令行版）
python clip_retrieval.py

# 自定义 query
python clip_retrieval.py "a cat sleeping on a sofa"

# 对比学习训练（命令行版）
python clip_train.py
python clip_train.py --epochs 10 --max-samples 2000

# Web 可视化界面（推荐）
streamlit run web_demo.py
```

**首次运行会**：
1. 自动下载 Flickr8k 数据集（约 1.1 GB，`data/` 目录）
2. 自动下载 CLIP ViT-B/32 模型（约 600 MB，HuggingFace 缓存）
3. 编码全量图片（CPU 约 3-5 分钟）→ 缓存到 `cache/`
4. 第二次运行直接用缓存，秒级出结果

**硬件需求**：4 GB+ 内存，CPU 可运行（无 GPU 要求）。

## 输出文件

| 文件 | 说明 |
| --- | --- |
| `outputs/retrieval_results_*.png` | 每次检索的 Top-5 图片拼接（含相似度分数） |
| `cache/flickr8k_img_emb_vitb32.npy` | 全量图片 CLIP 嵌入缓存，避免重复编码 |

## 建议学习方式（对应第三周第 6 天任务）

1. 先通读 `../讲义.md`「一～四」节（CLIP 动机与方法）
2. 运行项目，观察默认 query 的 Top-5 检索图
3. 修改 `QUERY_TEXT` 为 3-5 种不同描述，对比结果差异：
   - 具体描述 vs 抽象描述（如 "a person running" vs "jogger in a park during sunset"）
   - 中英文对比（CLIP 弱于英文，但可尝试）
4. 观察双向检索结果：Top-1 图片反问文本库，是否匹配原描述
5. 右键打开 `outputs/retrieval_results_*.png` 查看检索效果

## Query 实验建议

| 难度 | 示例 query | 预期 |
| --- | --- | --- |
| 简单 | "a dog" | Top-5 应全部是狗 |
| 中等 | "a child riding a bicycle" | 应出现小孩+自行车的图 |
| 较难 | "two people hugging on a beach" | 需同时满足多条件 |
| 中文 | "一匹马在草地上奔跑" | CLIP 弱于英文，可能偏差 |

## 项目文件结构

```
项目/
├── clip_retrieval.py     # 主程序：零样本检索（逐段【讲义·X】注释）
├── clip_train.py         # 训练模块：对比学习投影头 + InfoNCE loss
├── web_demo.py           # Web 可视化界面（Streamlit）
├── requirements.txt
├── data/                  # 下载的 Flickr8k（首次运行生成）
├── cache/                 # 缓存：图像嵌入 + 训练好的投影头权重
├── outputs/               # 检索结果图 + 训练曲线（每次运行生成）
└── README.md              # 本文件
```

## 关键代码速读

### 编码图片（correspond to 讲义·三）

```python
img_emb = model.get_image_features(**inputs)       # ViT 输出 [B, 512]
img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)  # L2 归一化 → cosine 空间
```

### 检索（correspond to 讲义·二）

```python
sim = query_emb @ gallery_emb.T   # 矩阵乘法，一次完成全部相似度计算
top_idx = sim.argsort()[::-1][:k]  # 降序取 Top-K
```

### 零样本（correspond to 讲义·四）

整个流程**没有任何训练循环**——CLIP 预训练时为对比学习，推理时直接用冻结参数做前向编码。这就是「零样本迁移」在检索场景的体现。

---

> 论文原文见 `../CLIP_paper.pdf`

---

## Web 可视化界面 (`web_demo.py`)

除了命令行检索，项目还提供了一个基于 **Streamlit** 的交互式 Web 界面：

```bash
streamlit run web_demo.py
```

### 功能一览

| 功能 | 说明 |
| --- | --- |
| **以文搜图** | 输入中/英文自然语言描述 → 实时返回 Top-K 张匹配图片 |
| **以图搜文** | 上传任意图片 → 从 Flickr8k 中搜索最相似的图片，展示其原始描述 |
| **预设示例** | 一键点击预设 query，快速体验检索效果 |
| **相似度可视化** | 自动绘制 Top-K 结果的相似度柱状图 |
| **参数调节** | 侧边栏可自由调节 Top-K 返回数量（1~20） |

### 界面示意

```
┌──────────────────────────────────────────────────┐
│  🔍 CLIP 零样本图文检索                           │
│  基于 OpenAI CLIP (ViT-B/32) ...                  │
├─────────────────────┬────────────────────────────┤
│ ⚙️ 参数设置           │  🎨 以文搜图               │
│                      │                            │
│ ○ 以文搜图            │  [输入搜索文字...]  [🔍搜索] │
│ ○ 以图搜文            │                            │
│ Top-K: [====5====]   │  ┌─────┐ ┌─────┐ ┌─────┐  │
│                      │  │ 图1  │ │ 图2  │ │ 图3  │  │
│                      │  │0.32  │ │0.31  │ │0.29  │  │
│ ──────────────────── │  └─────┘ └─────┘ └─────┘  │
│ 模型: ViT-B/32        │                            │
│ 数据: 8091 张         │  📊 相似度柱状图             │
└─────────────────────┴────────────────────────────┘
```

---

## 对比学习训练模块 (`clip_train.py`)

除了用预训练 CLIP 做零样本检索，本模块让你**亲手跑一次对比学习的训练过程**：

```bash
python clip_train.py                     # 默认：500 对，5 epoch
python clip_train.py --epochs 10 --max-samples 2000 --lr 1e-4
```

### 设计思路

完整训练 CLIP 需要 4 亿图文对 + 数百 GPU 天，不现实。本模块采用 **冻结编码器 + 可训练投影头** 的轻量方案：

```
Frozen CLIP Image Encoder ──→ img_emb [512] ──→ ImageProjectionHead ──→ img_proj [512]  ┐
                                                                                          ├── cosine sim [B×B] ──→ InfoNCE Loss
Frozen CLIP Text Encoder  ──→ txt_emb [512] ──→ TextProjectionHead  ──→ txt_proj [512]  ┘
```

- **冻结**：CLIP 编码器约 150M 参数不参与训练
- **可训练**：两个投影 MLP 各约 264K 参数（总共约 0.5M / 0.35%）
- **CPU 可运行**：默认 500 个训练对 + batch_size=16，约 30-60 分钟

### 训练流程

| 步骤 | 说明 |
| --- | --- |
| 1. 数据准备 | 从 Flickr8k 训练集取图文对（每张图 5 条描述），支持随机采样 |
| 2. 加载冻结 CLIP | 加载 ViT-B/32，设置 `requires_grad=False` |
| 3. 投影头前向 | 512 → 256 → 512 MLP，将冻结编码映射到对比学习空间 |
| 4. 对称 InfoNCE | 同时计算 image→text 和 text→image 两个方向的交叉熵 |
| 5. 反向传播 | 只更新投影头参数，编码器不动 |
| 6. 验证 | 每个 epoch 后在验证集上计算 Recall@K |

### 输出文件一览

| 文件 | 说明 |
| --- | --- |
| `outputs/train_loss_curve.png` | 训练损失 / 准确率 / 验证召回率曲线 |
| `outputs/sim_matrix_epoch*.png` | 每个 epoch 的相似度矩阵（对角线应逐渐变亮） |
| `cache/clip_projection_head.pt` | 训练好的投影头权重 |

### 可调参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--epochs` | 5 | 训练轮数 |
| `--batch-size` | 16 | 批次大小（CPU 建议 ≤8） |
| `--lr` | 3e-4 | 学习率 |
| `--temperature` | 0.07 | InfoNCE 温度（同 CLIP 论文默认值） |
| `--max-samples` | 500 | 最大训练样本数（设 0 使用全部 ~30000 对） |
| `--top-k` | 5 | 验证时 Recall@K 的 K |

### 观察重点

1. **Loss 下降**：InfoNCE 损失应逐渐降低（通常从 ~5.5 降至 ~4.0）
2. **Acc 上升**：对角线匹配准确率应逐渐升高
3. **相似度矩阵**：训练前对角线不突出 → 训练后对角线显著亮于周围区域
4. **召回升高**：验证集 Recall@5 应逐步提升
5. **温度效应**：尝试 `--temperature 0.02` vs `0.2`，观察损失变化

### 关键代码解读

#### InfoNCE 损失（对应讲义·二公式）

```python
def contrastive_loss(img_emb, txt_emb, temperature=0.07):
    img_emb = F.normalize(img_emb, dim=-1)     # L2 归一化
    txt_emb = F.normalize(txt_emb, dim=-1)

    logits = img_emb @ txt_emb.T / temperature  # [B, B] 相似度矩阵 / τ
    labels = torch.arange(B)                     # 对角线 = 正样本

    loss_i2t = F.cross_entropy(logits, labels)      # 图像→文本
    loss_t2i = F.cross_entropy(logits.T, labels)    # 文本→图像
    return (loss_i2t + loss_t2i) / 2.0              # 对称损失
```

**直观理解**：将 batch 内 B 张图的相似度矩阵看作 B 分类问题——第 i 张图应与第 i 条文本匹配（正样本），与其余 (B-1) 条不匹配（负样本）。

#### 投影头（对应讲义·二 CLIP 架构中的线性投影）

```python
class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),  # 降维
            nn.ReLU(),
            nn.Linear(256, 512),  # 升回
        )
    def forward(self, x):
        return self.net(x)
```

> 讲义参考：见 `../讲义.md`「九、PyTorch 实战项目 → 训练模块详解」
