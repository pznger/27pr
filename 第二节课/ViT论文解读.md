# ViT 论文精读：An Image is Worth 16x16 Words

> **论文**：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale  
> **作者**：Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby（Google Research, Brain Team）  
> **发表**：ICLR 2021  
> **论文文件**：见本目录下 `ViT_paper.pdf`
>
> 以下按论文章节顺序，逐段给出**原文**、**中文翻译**、**大白话解读**。标记 🎯 表示高频考点。  
> ViT 是 Transformer 从 NLP 跨越到计算机视觉的标志性工作——**证明了纯 Transformer 架构在图像任务上也能打败 CNN**。

---

## 零、为什么 ViT 如此重要？

在 ViT（2020 年）之前，计算机视觉领域是 CNN（卷积神经网络）的天下——从 2012 年的 AlexNet 到 2015 年的 ResNet，再到 2019 年的 EfficientNet，所有 SOTA 模型都是 CNN 或其变体。虽然有人在 CNN 里加入过注意力机制，但从来没有人敢**把 CNN 整个扔掉，只用 Transformer**。

ViT 的标题 "An Image is Worth 16x16 Words" 是一个巧妙的双关—参考了英语谚语"A picture is worth a thousand words"（一图胜千言）。ViT 的做法是：**把一张图切成 16×16 的小块（patches），每个 patch 当成一个"单词"，然后直接喂给标准 Transformer**。

结果出人意料：**只要数据够多，纯 Transformer 在图像分类上可以打败最先进的 CNN，而且训练成本更低**。

> 🎯 ViT 与《Transformer》论文的关系：ViT 几乎原封不动地使用了 Transformer 的 **Encoder** 部分（就是 BERT 用的那个），只是把输入从"词向量序列"改成了"图像块向量序列"。论文中多次提到"follow the original Transformer as closely as possible"。

---

## 一、摘要（Abstract）

**原文：**

> While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.

**翻译：**

> 虽然 Transformer 架构已成为自然语言处理任务的事实标准，但其在计算机视觉中的应用仍然有限。在视觉领域，注意力要么与卷积网络一起使用，要么用来替换卷积网络的某些组件，同时保持其整体结构不变。我们证明，这种对 CNN 的依赖并非必要——直接应用于图像块序列的纯 Transformer 可以在图像分类任务上表现得非常好。

**大白话：**

当时做 CV 的人说："Transformer 是给文本用的，图像还得靠 CNN。" ViT 的回答是："不需要 CNN，给我把图切成小块，每块当做一个词，剩下的交给标准 Transformer 就行。" 这是从 **CNN + Attention（混搭）→ Pure Transformer（纯用）** 的跨越。

---

**原文：**

> When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

**翻译：**

> 当在大规模数据上预训练并迁移到多个中型或小型图像识别基准（ImageNet、CIFAR-100、VTAB 等）时，ViT 取得了优异的成绩，与最先进的卷积网络相比，所需训练计算资源大幅减少。

**大白话：**

ViT 的杀手锏：**同样精度下，训练计算量只有 CNN 的 1/2 到 1/4**。在 AI 界，这意味着你可以用更少的 GPU 小时获得同样好（甚至更好）的模型。

---

## 二、引言（1. Introduction）

**原文：**

> Self-attention-based architectures, in particular Transformers, have become the model of choice in natural language processing (NLP). The dominant approach is to pre-train on a large text corpus and then fine-tune on a smaller task-specific dataset. Thanks to Transformers' computational efficiency and scalability, it has become possible to train models of unprecedented size, with over 100B parameters.

**翻译：**

> 基于自注意力的架构，特别是 Transformer，已成为自然语言处理的首选模型。主导方法是在大型文本语料库上预训练，然后在较小的任务特定数据集上微调。由于 Transformer 的计算效率和可扩展性，训练超过 1000 亿参数的模型已成为可能。

**大白话：**

这句话描述了 NLP 领域的"预训练+微调"范式（BERT、GPT 都这样）。ViT 的野心是把这套在 NLP 已被验证的流程搬进 CV。

---

**原文：**

> In computer vision, however, convolutional architectures remain dominant. Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention, some replacing the convolutions entirely. The latter models, while theoretically efficient, have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns.

**翻译：**

> 然而在计算机视觉中，卷积架构仍然占主导地位。受 NLP 成功的启发，许多工作尝试将类似 CNN 的架构与自注意力结合，有些完全替换了卷积。后一种模型虽然在理论上高效，但由于使用了专门的注意力模式，尚未在现代硬件加速器上有效扩展。

**大白话：**

CV 领域曾有人尝试只用注意力不用卷积，但都失败了。为什么？因为按像素做自注意力计算量是 O(n²)，224×224 的图像有 50,176 个像素——对每个像素做全局注意力的计算量无法承受。所以之前的工作只能在"局部区域"做注意力（类似 CNN 的局部性），导致实现复杂且不好扩展。

ViT 的巧妙解法：**不做像素级的注意力，做"区块级"（patch）的注意力**。16×16 的 patch 让序列长度从 50,176 暴降到 196——Transformer 完全可以搞定。

---

**原文（ViT 的核心思路）：**

> We split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application. We train the model on image classification in supervised fashion.

**翻译：**

> 我们将图像分割成块（patches），并将这些块的线性嵌入序列作为 Transformer 的输入。图像块的处理方式与 NLP 应用中的 token（单词）完全相同。我们以监督方式训练模型进行图像分类。

**大白话：**

🎯 ViT 的核心理念——**把图当作句子来读**：

```
NLP 的 Transformer：
  句子 → [词向量₁, 词向量₂, 词向量₃, ...] → Transformer → 输出

ViT：
  图像 → [块向量₁, 块向量₂, 块向量₃, ...] → Transformer → 输出
```

唯一的不同是：NLP 用词嵌入表（lookup table）把词变成向量；ViT 用**线性投影**（一个全连接层）把每个 16×16×3=768 像素的 RGB 块变成向量。

---

**原文（小数据集 vs 大数据集——论文最重要的洞察）：**

> When trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNets of comparable size. This seemingly discouraging outcome may be expected: Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality. However, the picture changes if the models are trained on larger datasets (14M-300M images). We find that large scale training trumps inductive bias.

**翻译：**

> 在中等规模数据集（如 ImageNet）上训练且没有强正则化时，这些模型的准确率比同等规模的 ResNet 低几个百分点。这个看似令人沮丧的结果可能是预料之中的：Transformer 缺少 CNN 固有的一些归纳偏置，如平移等变性和局部性。然而，如果在更大的数据集（1400 万-3 亿张图像）上训练，情况就会改变。我们发现大规模训练胜过归纳偏置。

**大白话：**

🎯 论文最核心的结论就在这一句：

- **数据少时**：CNN > ViT（因为 CNN 内置了"图像先验知识"，比如一个 3×3 的卷积核天然假设相邻像素关系密切）
- **数据多时**：ViT > CNN（Transformer 从海量数据中自己学会了这些规律，而且因为不受 CNN 归纳偏置的限制，最终学到的表示更灵活、更强大）

类比：CNN 像是给了你一个"参考答案模板"（归纳偏置）——数据少时有用，但可能限制你的发挥。ViT 是从零开始自学——数据少时容易"学歪"，但数据多时能学到比模板更好的东西。

---

## 三、方法（3. Method）

### 3.1 ViT 架构

**原文：**

> The standard Transformer receives as input a 1D sequence of token embeddings. To handle 2D images, we reshape the image x ∈ R^(H×W×C) into a sequence of flattened 2D patches x_p ∈ R^(N×(P²·C)), where (H,W) is the resolution, C is the number of channels, (P,P) is the resolution of each image patch, and N = HW/P² is the resulting number of patches.

**翻译：**

> 标准 Transformer 接收一维 token 嵌入序列作为输入。为处理二维图像，我们将图像 x ∈ R^(H×W×C) 重塑为扁平二维块的序列 x_p ∈ R^(N×(P²·C))，其中 (H,W) 是分辨率，C 是通道数，(P,P) 是每个图像块的分辨率，N = HW/P² 是生成的块数。

**大白话：**

🎯 ViT 的输入流程——"切块→展平→投影"三步走：

```
原始图像：224×224×3

切块（P=16）：分成 (224/16)×(224/16) = 14×14 = 196 个块
  每个块：16×16×3 = 768 个像素值

展平：196 个块 → 196 个 768 维向量

投影（线性层：768 → D，如 768 → 768 或 768→1024）：
  196 个 D 维向量 → 输入 Transformer
```

> 注意：ViT 只用了 Transformer 的 **Encoder**（没有 Decoder，因为图像分类不需要生成序列）。这就是为什么 ViT 论文说"follow BERT"——BERT 也只用了 Encoder。

---

**原文（补充 [class] token）：**

> Similar to BERT's [class] token, we prepend a learnable embedding to the sequence of embedded patches, whose state at the output of the Transformer encoder serves as the image representation y. Both during pre-training and fine-tuning, a classification head is attached to z⁰_L.

**翻译：**

> 与 BERT 的 [class] token 类似，我们在嵌入块序列前添加一个可学习的嵌入，其在 Transformer 编码器输出端的状态作为图像表示 y。在预训练和微调期间，一个分类头被附加到 z⁰_L。

**大白话：**

BERT 的做法是：在输入句子前面加一个特殊的 `[CLS]` token，经过 Transformer 处理后，`[CLS]` 的输出向量就代表"整个句子的语义"。ViT 完全照搬这个设计：

```
输入：[class_token, patch₁, patch₂, ..., patch₁₉₆]  （共 197 个 token）
输出：[class_vector, patch₁_vec, ..., patch₁₉₆_vec]  （197 个输出向量）
       ↑
       取这个 → 送入分类头（MLP）→ 输出分类结果
```

为什么需要这个额外的 [class] token？理论上可以对 196 个 patch 输出做全局平均池化——实际上 ViT 也试过，效果差不多。用 [class] token 纯粹是为了和 BERT 保持一致。

---

**原文（位置编码的选择）：**

> Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings.

**翻译：**

> 位置嵌入被添加到块嵌入中以保留位置信息。我们使用标准的可学习一维位置嵌入，因为我们没有观察到使用更高级的二维感知位置嵌入带来的显著性能提升。

**大白话：**

ViT 加位置编码的方式比原版 Transformer 更直接——直接用可学习的向量（每个位置一个独立可训练的向量），而不是 sin/cos 函数。

更重要的是：有人尝试过"2D 位置编码"（给每个块一个 (x,y) 坐标编码），结果和简单的 1D 位置编码**几乎一样**。这说明 Transformer 自己就能从 1D 位置编码中学到块之间的 2D 空间关系——根本不需要人工注入图像先验。

---

### ViT 的公式总结

**论文公式 (1)-(4) 一图打尽 ViT 的计算流程：**

```
z₀ = [x_class; x¹E; x²E; ...; x^N E] + E_pos     (1) 输入：拼接 + 位置编码
z'ℓ = MSA(LN(z_{ℓ-1})) + z_{ℓ-1}                   (2) 多头自注意力 + 残差
z_ℓ = MLP(LN(z'_ℓ)) + z'_ℓ                          (3) 前馈网络 + 残差
y   = LN(z⁰_L)                                      (4) 最终分类向量
```

和原版 Transformer 相比，核心结构**一模一样**。唯一区别：
- 没有 Decoder（因为图像分类不是序列生成任务）
- LayerNorm 放在子层前面（Pre-LN）而不是后面（Post-LN）——这是后来的实践经验，训练更稳定

---

### ViT 变体

| 模型 | 层数 L | 隐藏维度 D | MLP 大小 | 头数 | 参数量 |
|------|:---:|:---:|:---:|:---:|:---:|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

命名规则：`ViT-L/16` = Large 模型 + 16×16 patch 大小。Patch 越小 → 序列越长 → 计算量越大但效果越好。

---

## 四、归纳偏置：ViT vs CNN

**原文：**

> We note that Vision Transformer has much less image-specific inductive bias than CNNs. In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer. In ViT, only MLP layers are local and translationally equivariant, while the self-attention layers are global. The two-dimensional neighborhood structure is used very sparingly: in the beginning of the model by cutting the image into patches and at fine-tuning time for adjusting the position embeddings for images of different resolution.

**翻译：**

> 我们注意到，ViT 比 CNN 具有更少的图像特定归纳偏置。在 CNN 中，局部性、二维邻域结构和平移等变性被内置到每一层中。在 ViT 中，只有 MLP 层是局部且平移等变的，而自注意力层是全局的。二维邻域结构的使用非常少：在模型开始时将图像切割成块，以及在微调时针对不同分辨率调整位置嵌入。

**大白话：**

🎯 CNN 和 ViT 的核心区别——"先验知识" vs "从头学习"：

| | CNN | ViT |
|------|------|------|
| **局部性** | ✅ 每个 3×3 卷积核只看邻近像素 | ❌ 自注意力全局看所有块 |
| **平移等变性** | ✅ 猫在左边还是右边，CNN 天然知道只是平移了 | ❌ 需要从数据中学习 |
| **2D 邻域结构** | ✅ 每层都保持 2D 结构 | ❌ 只在切块时用了一次 2D 信息 |
| **数据需求** | 中等 | 大量（需要数据弥补归纳偏置的缺失） |

CNN 就像一个"戴了眼镜"的人——眼镜（卷积核）预设了"相邻像素有联系"的看世界方式。ViT 是一个"不戴眼镜"的人——一开始什么都看不清，但给他看足够多的图像后，他学到的看世界方式可能比戴眼镜更准确。

---

## 五、实验（4. Experiments）

### 5.1 核心结果对比

**原文（Table 2——SOTA 对比表）：**

> The smaller ViT-L/16 model pre-trained on JFT-300M outperforms BiT-L on all tasks, while requiring substantially less computational resources to train. The larger model, ViT-H/14, further improves the performance, especially on the more challenging datasets.

**翻译：**

> 在 JFT-300M 上预训练的较小 ViT-L/16 模型在所有任务上均优于 BiT-L，同时训练所需的计算资源大幅减少。更大的 ViT-H/14 模型进一步提高了性能，尤其是在更具挑战性的数据集上。

**大白话：**

🎯 ViT 最重要的结果表：

| 模型 | ImageNet | CIFAR-100 | VTAB(19任务) | 训练计算量 |
|------|:---:|:---:|:---:|:---:|
| BiT-L (ResNet152x4) | 87.54 | 93.51 | 76.29 | 9.9k TPU天数 |
| Noisy Student (EfficientNet-L2) | 88.4 | - | - | 12.3k |
| ViT-L/16 (JFT) | 87.76 | 93.90 | 76.28 | **0.68k** |
| **ViT-H/14 (JFT)** | **88.55** | **94.55** | **77.63** | **2.5k** |

关键洞察：ViT-H/14 在所有数据集上全面超越 CNN，且训练计算量只有排名第二的 Noisy Student 的 **1/5**。ViT-L/16 在多数任务上追平或超越 BiT-L，但训练计算量只有它的 **1/14**。

---

### 5.2 数据规模的影响——论文最著名的图（Figure 3）

**原文：**

> When pre-trained on the smallest dataset, ImageNet, ViT-Large models underperform compared to ViT-Base models. With ImageNet-21k pre-training, their performances are similar. Only with JFT-300M, do we see the full benefit of larger models.

**翻译：**

> 在最小数据集 ImageNet 上预训练时，ViT-Large 模型的表现不如 ViT-Base。在 ImageNet-21k 上预训练时，它们的表现相似。只有在 JFT-300M 上，我们才看到更大模型的全部优势。

**大白话：**

🎯 ViT 论文最重要的发现——**数据量决定了模型的上限**：

| 预训练数据 | ViT-Base | ViT-Large | ViT-Huge | BiT (ResNet) |
|------|:---:|:---:|:---:|:---:|
| ImageNet (1.3M) | 一般 | **差**（大模型过拟合） | - | **好**（CNN 胜） |
| ImageNet-21k (14M) | 好 | 差不多 | - | 好 |
| JFT-300M (303M) | 好 | **很好** | **最佳** | 好（已饱和） |

**数据少时 CNN 赢，数据多时 ViT 赢**。Figure 3 清晰展示了这个交叉——它是 ViT 论文被引用最多的图。

为什么？因为 CNN 的归纳偏置在数据少时是"拐杖"——帮你不摔倒；数据多时是"束缚"——让你跑不到最快。ViT 不需要拐杖，但需要足够多的数据来学会行走。

---

### 5.3 性能 vs 计算量（Figure 5——Scaling Study）

**原文：**

> Vision Transformers dominate ResNets on the performance/compute trade-off. ViT uses approximately 2−4× less compute to attain the same performance (average over 5 datasets). Hybrids slightly outperform ViT at small computational budgets, but the difference vanishes for larger models.

**翻译：**

> ViT 在性能/计算量权衡上全面优于 ResNet。ViT 使用约 2-4 倍少的计算量就能达到相同的性能（5 个数据集的平均）。混合模型在小计算量预算下略优于 ViT，但差距在较大模型时消失。

**大白话：**

🎯 同计算量下 ViT > CNN；同精度下 ViT 所需的计算量 = CNN 的 1/2 ~ 1/4。而且 ViT 在实验范围内没有出现性能饱和——继续增大模型和数据，性能还能涨。

混合模型（ResNet 提取特征 + ViT 处理）在小模型时略有优势，但模型大了之后纯 ViT 追平——说明**非卷积的特征提取是可以完全学会的**。

---

### 5.4 模型内部的可解释性分析（Figure 7）

**原文：**

> The model learns to encode distance within the image in the similarity of position embeddings, i.e. closer patches tend to have more similar position embeddings. Self-attention allows ViT to integrate information across the entire image even in the lowest layers. Some heads attend to most of the image already in the lowest layers, showing that the ability to integrate information globally is indeed used by the model.

**翻译：**

> 模型学会了在位置嵌入的相似性中编码图像内的距离，即更近的块倾向于有更相似的位置嵌入。自注意力使得 ViT 即使在最低层也能整合整个图像的信息。一些注意力头在最低层就已经关注了图像的大部分区域，表明模型确实使用了全局信息整合的能力。

**大白话：**

🎯 ViT 内部表示的可视化揭示了三件事：

1. **位置嵌入学会了 2D 空间结构**：即使只给了 1D 位置编码，训练后相近的块自动有了相似的位置嵌入——模型自己"悟出"了图像的 2D 拓扑结构！

2. **低层就有全局视野**：CNN 的低层只能看到局部（3×3 感受野），要堆很多层才能看到全图。ViT 从第一层开始，每个块就能看到所有其他块——这是自注意力天生的能力

3. **不同注意力头分工明确**：有的头专注"看全图"，有的头专注"看局部邻居"——ViT 自动学会了类似 CNN 不同层的不同功能，但不是通过人工设计，而是从数据中学到的

---

### 5.5 自监督预训练探索

**原文：**

> We also perform a preliminary exploration on masked patch prediction for self-supervision, mimicking the masked language modeling task used in BERT. With self-supervised pre-training, our smaller ViT-B/16 model achieves 79.9% accuracy on ImageNet, a significant improvement of 2% to training from scratch, but still 4% behind supervised pre-training.

**翻译：**

> 我们还对用于自监督的掩码块预测进行了初步探索，模仿 BERT 中使用的掩码语言建模任务。通过自监督预训练，我们较小的 ViT-B/16 模型在 ImageNet 上达到了 79.9% 的准确率——比从头训练显著提高了 2%，但仍比监督预训练低 4%。

**大白话：**

ViT 论文也试了"图像版的 BERT"——随机遮住一些图像块，让模型预测被遮住的是什么。效果还行但不如监督学习。不过论文说"这是一个有前途的方向，留给未来工作"——而后来出现的工作（MAE、BEiT等）正是沿着这条路走，取得了巨大成功！

---

## 六、结论（5. Conclusion）

**原文：**

> We have explored the direct application of Transformers to image recognition. Unlike prior works using self-attention in computer vision, we do not introduce image-specific inductive biases into the architecture apart from the initial patch extraction step. This simple, yet scalable, strategy works surprisingly well when coupled with pre-training on large datasets.

**翻译：**

> 我们探索了 Transformer 在图像识别中的直接应用。与之前在计算机视觉中使用自注意力的工作不同，除了初始的块提取步骤外，我们没有在架构中引入图像特定的归纳偏置。这种简单但可扩展的策略在大规模数据集预训练时表现出奇的好。

**大白话：**

论文最后点出了 ViT 的哲学：**少即是多**。不需要为图像设计特化结构，只要数据够多，标准 Transformer 就能自己学会图像的所有规律。

---

**原文（未来方向——部分已实现）：**

> While these initial results are encouraging, many challenges remain. One is to apply ViT to other computer vision tasks, such as detection and segmentation. Another challenge is to continue exploring self-supervised pre-training methods.

**翻译：**

> 虽然这些初步结果令人鼓舞，但仍有许多挑战。一是将 ViT 应用于其他计算机视觉任务，如检测和分割。另一个挑战是继续探索自监督预训练方法。

**大白话：**

论文提到的两个方向：
- **检测和分割**：现在已经有 DETR（2020）、MaskFormer（2021）等基于 Transformer 的检测/分割模型
- **自监督预训练**：MAE（何恺明 2022）沿着 ViT 的路，用掩码自编码器在 ImageNet 上达到 87.8% Top-1（不依赖 JFT-300M）

ViT 论文本身只是图像分类，但它的思想已经扩散到 CV 的几乎所有子领域。

---

## 七、ViT 核心贡献总结

| 贡献 | 详情 |
|------|------|
| **纯 Transformer 用于 CV** | 第一个完全不用 CNN 且能在 ImageNet 上达到 SOTA 的纯注意力模型 |
| **"图像 = 词序列"的范式** | 把图像切成 patch 当作 token——简单但极其有效 |
| **大规模预训练 > 归纳偏置** | 数据够多时，模型自己学到的规律比人工设计的先验更好 |
| **性能/计算量比 CNN 更优** | 同样精度下训练计算量只需 CNN 的 1/2 ~ 1/4 |
| **可解释性分析** | 位置嵌入学会 2D 结构，注意力头自动分工 |
| **开启了 Transformer CV 时代** | 后续的 DETR、MAE、Swin Transformer、DALL·E 都建立在 ViT 的基础上 |

> 🎯 ViT 与 Transformer 论文的关系一句话：ViT 几乎原封不动用了 Transformer 的 Encoder（就是 BERT 那套），唯一改动是把输入从词向量变成图像块的线性投影向量。**证明了 Transformer 是跨模态的通用架构**。

> 如果想进一步了解，可以看 ViT 论文的 **Figure 1**（架构全景图）和 **Figure 3**（数据规模效应——这是整篇论文最重要的实验结论）。

---

> **论文原文**见本目录下 `ViT_paper.pdf`。建议对照 Figure 1（模型结构图）和 Figure 3（数据规模对性能的影响）来理解。
