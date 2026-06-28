# 保研资格预测 — 第一节课 PyTorch 小项目

> **建议完成时间**：第一周第 6 天（直播后）  
> **前置条件**：已完成 Anaconda 环境配置、Cursor 安装，并观看刘二大人课程前 4 集

本项目将第一节课全部核心知识点串成一个可运行的完整 pipeline。

**代码阅读方式**：打开 `graduate_admission_mlp.py`，按文件内标记分段学习：

| 搜索标记 | 对应讲义 |
| --- | --- |
| `【讲义·二】` | 数据集划分、评估指标 |
| `【讲义·三】` | 线性回归 + MSE |
| `【讲义·四】【讲义·五】` | 梯度下降、反向传播 |
| `【讲义·六】` | 激活函数 |
| `【讲义·七】` | 逻辑回归 |
| `【讲义·八】` | 交叉熵 |
| `【讲义·九】` | MLP |
| `【Part 1/2/3】` | 三段主线训练流程 |

每个函数在 `graduate_admission_mlp.py` 内均有行内注释；**下方「代码逐段对照解读」** 提供更系统的讲义对照说明，建议两处结合阅读。

| 代码模块 | 讲义内容 |
| --- | --- |
| `load_dataset_from_csv` / `prepare_data` | 数据集划分、归一化、监督学习 |
| `LinearRegressionModel` | 线性回归 + MSE |
| `LogisticRegressionModel` | 逻辑回归 + Sigmoid |
| `MLPClassifier` | MLP + ReLU 激活函数 |
| `loss.backward()` | 反向传播（自动求导） |
| `optimizer.step()` | 梯度下降 / 小批量更新 |
| `classification_metrics` | 准确率、精确率、召回率、F1 |
| `demo_cross_entropy_gradient` | Softmax + 交叉熵梯度推导验证 |

## 数据文件

项目使用固定 CSV 数据集 `graduate_admission.csv`（800 条样本）。**同一份数据**上反复调参，指标变化才具有可比性。

| 列名 | 说明 |
| --- | --- |
| 绩点GPA ~ 综合排名分位 | 特征，均为 0~1 归一化值 |
| 综合评分 | 回归标签，0~100 |
| 保研资格 | 分类标签，0/1（约 **35%** 为正样本，模拟名额有限） |

### 造数规则（合成演示数据）

| 标签 | 规则 |
| --- | --- |
| 综合评分 | 五维特征线性加权 + 噪声，裁剪到 0~100 |
| 保研资格 | latent 分数含非线性项「高绩点 × 低科研」惩罚，按 **Top 35%** 切分为 1，其余为 0 |

因此逻辑回归与 MLP 的 F1 会有可见差异，调 `hidden` / `lr` 时指标变化更有意义。可用 Excel 或 Cursor 直接打开 CSV 查看。

若文件丢失或需重新生成：

```bash
python generate_dataset_csv.py
```

## 环境准备

```bash
pip install -r requirements.txt
```

## 运行

```bash
python graduate_admission_mlp.py
```

运行结束后在 `outputs/` 目录查看训练曲线与评估图：

| 文件 | 内容 |
| --- | --- |
| `01_activation_functions.png` | Sigmoid / ReLU / Tanh 曲线 |
| `02_regression_loss_curve.png` | 线性回归 train/val MSE 随 epoch |
| `03_regression_scatter.png` | 测试集真实 vs 预测散点 |
| `04_feature_scatter.png` | 绩点 vs 科研（按保研标签着色） |
| `05_logistic_training.png` | 逻辑回归 loss + 验证 Acc/F1 |
| `05_logistic_confusion_matrix.png` | 逻辑回归混淆矩阵 |
| `06_MLP_training.png` | MLP loss + 验证 Acc/F1 |
| `06_MLP_confusion_matrix.png` | MLP 混淆矩阵 |
| `07_model_comparison.png` | 两模型测试集指标柱状对比 |

## 建议学习方式（对应第一周第 6 天任务）

1. 先通读 `../讲义.md` 对应章节（直播日内容）
2. **对照阅读**：代码内 `【讲义·X】` 注释 + 本 README「代码逐段对照解读」
3. 运行脚本，打开 `outputs/` 对照训练曲线理解收敛过程
4. 尝试修改 `hidden=16` 为 `4` 或 `64`，观察 MLP 过拟合/欠拟合（**对比测试集 F1，数据不变**）
5. 尝试把学习率 `lr` 调大 10 倍，观察 `02/05/06` 号图中 loss 是否震荡
6. 对照 `demo_cross_entropy_gradient()` 与 README 第五部分，验证讲义第八节梯度公式

## 第一周自习进度对照

| 天数 | 与本项目的关系 |
| --- | --- |
| 第 1～2 天 | 配置 `dl-course` 虚拟环境，在 Cursor 中打开本项目 |
| 第 3～4 天 | 视频中的线性回归、梯度下降、反向传播为本项目理论基础 |
| 第 5 天 | 直播讲解本项目结构与运行方式 |
| 第 6 天 | **运行并完成本项目**（本 README 下方命令） |
| 第 7 天 | 复盘项目代码，预习 Transformer |

## 场景说明

特征均为 0~1 归一化值：

- **绩点 GPA**
- **科研经历**
- **竞赛获奖**
- **英语成绩**
- **综合排名分位**（越小越好）

标签：

- 回归：综合评分（0~100）
- 分类：是否获得保研资格（0/1）

数据保存在 `graduate_admission.csv`：800 条固定合成样本，分类正样本约 35%，仅用于教学。

---

## 代码逐段对照解读

> 阅读方式：**左栏打开本 README，右栏打开 `graduate_admission_mlp.py`**，按六个部分顺序跳转。  
> 讲义路径：`../讲义.md`

### 文件总览

```
graduate_admission_mlp.py
├── 第一部分  数据加载与预处理     【讲义·二】
├── 第二部分  模型定义             【讲义·三 / 七 / 九】
├── 第三部分  评估指标             【讲义·二】
├── 3.5 部分  可视化               【讲义·二 / 三 / 四 / 六】
├── 第四部分  训练循环             【讲义·四 / 五 / 八】
├── 第五部分  小实验               【讲义·六 / 八】
└── 第六部分  main() 主流程        【讲义·十一】
```

运行顺序（`main()`）：`prepare_data` → 画图 → 两个小实验 → Part1 回归 → Part2 逻辑回归 → Part3 MLP → 模型对比。

---

### 第一部分：数据加载与预处理【讲义·二】

#### `load_dataset_from_csv()` — 读 CSV

| 代码 | 含义 |
| --- | --- |
| `path.open(encoding="utf-8-sig")` | 读取带中文表头的 CSV（Excel 兼容编码） |
| `header != expected` | 校验 7 列顺序，防止特征与标签列错位 |
| `data[:, :NUM_FEATURES]` | 前 5 列 → 特征矩阵 **x**，shape `(800, 5)` |
| `data[:, NUM_FEATURES]` | 第 6 列 → 回归标签 **y_reg**（综合评分） |
| `data[:, NUM_FEATURES + 1]` | 第 7 列 → 分类标签 **y_cls**（0/1） |

**讲义对应**：监督学习样本集 $\{(x_i, y_i)\}$；每个样本有输入和标签。

#### `train_test_split_stratified()` — 分层划分

```python
idx0 = np.where(y_cls == 0)[0]   # 负类下标
idx1 = np.where(y_cls == 1)[0]   # 正类下标
# 两类各自取 test_size 比例，再合并 → 保证 train/test 正负比例接近
```

**原理**：若随机划分，小数据集 + 类别不平衡时，测试集可能几乎没有正样本，F1 波动很大。分层划分按类内比例切分，更稳定。

**讲义对应**：训练集 / 验证集 / 测试集划分；验证集用于训练过程中看指标，测试集**只用一次**做最终评估。

#### `standard_scale()` — 标准化

```python
mean = x_train.mean(axis=0)      # 仅用训练集统计量
x_train_s = (x_train - mean) / std
others_s = [(arr - mean) / std for arr in x_other]  # val/test 用同一组 mean/std
```

**原理**：$x' = (x - \mu) / \sigma$，使各特征尺度一致，梯度下降更新更均衡。

**易错点**：若用「全量 800 条」算 mean/std，测试集信息会泄露到预处理，指标虚高。

#### `prepare_data()` — 数据流水线

| 步骤 | 代码 | 结果 |
| --- | --- | --- |
| 1 | `load_dataset_from_csv()` | 800 条原始数据 |
| 2 | `test_size=0.3` 第一次划分 | train **560** + temp 240 |
| 3 | temp 再 `test_size=0.5` | val **120** + test **120** |
| 4 | `standard_scale(...)` | 标准化 |
| 5 | `torch.from_numpy(...)` | 转为 PyTorch 张量，供 GPU/自动求导使用 |

三个模型共用同一份 `DataBundle`，保证对比公平。

---

### 第二部分：模型定义【讲义·三 / 七 / 九 / 六】

PyTorch 约定：继承 `nn.Module` → `__init__` 定义层 → `forward` 写前向公式 → `loss.backward()` 自动求梯度。

#### `LinearRegressionModel`【讲义·三】

```python
self.linear = nn.Linear(in_dim, 1)   # ŷ = W·x + b，W 形状 (1,5)，b 形状 (1,)
return self.linear(x).squeeze(-1)    # 输出 (N,) 与 y_reg 对齐
```

- **任务**：预测连续值「综合评分」
- **损失**：MSE $= \frac{1}{n}\sum(\hat y - y)^2$
- **参数**：5 个权重 + 1 个偏置 = 6 个可学习参数

#### `LogisticRegressionModel`【讲义·七 + 六】

```python
return torch.sigmoid(self.linear(x).squeeze(-1))  # p ∈ (0,1)
```

- **公式**：$p = \sigma(w^T x + b)$，$\sigma$ 为 Sigmoid
- **推断**：$p \ge 0.5$ → 预测保研（1）
- **局限**：决策边界是线性的，对「高绩点 × 低科研」这类**非线性**规律拟合能力有限

#### `MLPClassifier`【讲义·九 + 六】

```
Input(5) → Linear(5→16) → ReLU → Linear(16→16) → ReLU → Linear(16→2) → logits
```

| 层 | 作用 |
| --- | --- |
| `Linear` | 矩阵乘法 + 偏置，可学习 |
| `ReLU` | $\max(0,x)$，负值截断，引入**非线性** |
| 输出 2 维 logits | 未归一化分数；`CrossEntropyLoss` 内部做 Softmax |

**调参入口**：`MLPClassifier(..., hidden=16)` — 改 `hidden` 观察 `06_MLP_training.png` 中 val F1 变化。

---

### 第三部分：评估指标【讲义·二】

#### 混淆矩阵与四指标

```
              预测 0    预测 1
真实 0         TN        FP
真实 1         FN        TP
```

| 指标 | 公式 | 代码 |
| --- | --- | --- |
| 准确率 | $(TP+TN)/总数$ | `accuracy` |
| 精确率 | $TP/(TP+FP)$ | `precision` — 预测为正中多少真 |
| 召回率 | $TP/(TP+FN)$ | `recall` — 真实正中被找出多少 |
| F1 | $2PR/(P+R)$ | `f1` — 不平衡数据比 Accuracy 更可靠 |

`classification_metrics()` 用手算实现上述公式，对应讲义第二节，便于理解 `05/06_*_confusion_matrix.png`。

#### `regression_mse()`

```python
torch.mean((y_true - y_pred) ** 2)   # 均方误差，越小越好
```

对应 `03_regression_scatter.png`：点越贴近对角线 $y=x$，MSE 越小。

---

### 3.5 部分：可视化

训练时每个 epoch 把 loss / val 指标写入 `TrainHistory`，训练结束后调用绘图函数保存到 `outputs/`。

| 函数 | 何时调用 | 观察什么 |
| --- | --- | --- |
| `plot_feature_scatter` | main 开头 | 数据在绩点-科研平面的分布 |
| `plot_activation_curves` | 小实验后 | Sigmoid/ReLU/Tanh 形状 |
| `plot_regression_loss` | Part1 结束 | MSE 是否收敛；train vs val 是否分叉（过拟合） |
| `plot_regression_scatter` | Part1 结束 | 回归拟合质量 |
| `plot_classifier_training` | Part2/3 结束 | loss 下降 + val F1 上升 |
| `plot_confusion_matrix` | Part2/3 结束 | FP/FN 错在哪 |
| `plot_model_comparison` | 全部训练完 | 逻辑回归 vs MLP 柱状对比 |

---

### 第四部分：训练循环【讲义·四 / 五 / 八】— 核心四步

**每个 epoch 都重复以下四步**（讲义训练循环图）：

```python
pred = model(x_train)        # ① 前向传播：算预测 ŷ
loss = criterion(pred, y)    # ② 算损失：衡量误差

optimizer.zero_grad()        # ③a 清空旧梯度（必须！否则梯度累加）
loss.backward()              # ③b 反向传播：链式法则求 ∂L/∂θ
optimizer.step()             # ④ 梯度下降：θ ← θ - lr·∂L/∂θ
```

| 概念 | 讲义 | 代码 |
| --- | --- | --- |
| 梯度下降 | 第四节 | `optimizer.step()`，沿负梯度更新参数 |
| 反向传播 | 第五节 | `loss.backward()`，PyTorch 自动求导 |
| 学习率 lr | 第四节 | `SGD(..., lr=0.1)`，过大→震荡，过小→收敛慢 |
| `torch.no_grad()` | — | 验证/测试时不建计算图，省内存 |

#### `train_regression()`【Part 1 / 讲义·三】

- `criterion = nn.MSELoss()`
- 全量 560 条一次更新（批梯度下降）
- 默认 `epochs=200`, `lr=0.05`

#### `train_binary_classifier()`【Part 2 & 3】

| 模型 | 前向输出 | 损失函数 | 标签类型 |
| --- | --- | --- | --- |
| 逻辑回归 | Sigmoid 概率 | `nn.BCELoss()` | `float` |
| MLP | logits（未 Softmax） | `nn.CrossEntropyLoss()` | `long` (0/1) |

**为何 MLP 不用 BCE？** `CrossEntropyLoss(logits, target)` 内部合并 Softmax + 负对数，数值更稳定（讲义第八节）。

#### 推断函数

```python
predict_logistic: (model(x) >= 0.5).long()   # 概率阈值 0.5
predict_mlp:      model(x).argmax(dim=1)     # 取概率最大类（等价 Softmax+argmax）
```

---

### 第五部分：小实验【讲义·六 / 八】

#### `demo_activation_functions()`

对 $x=0.5, -0.5$ 打印 Sigmoid / ReLU / Tanh 数值，并生成 `01_activation_functions.png`。

#### `demo_cross_entropy_gradient()`

验证讲义公式：**Softmax + 交叉熵的梯度 = 预测概率 q − one-hot 标签 p**

```python
loss = CrossEntropyLoss(reduction="sum")(logits, target)
loss.backward()
grad_manual = softmax(logits) - one_hot(target)   # 手算
# 与 logits.grad 对比 → 两者一致
```

---

### 第六部分：`main()` 主流程【讲义·十一】

```python
torch.manual_seed(42)          # 固定权重初始化，结果可复现
data = prepare_data()        # 加载 + 划分 + 标准化

plot_feature_scatter(data)   # 先看数据长什么样
demo_activation_functions()  # 激活函数
demo_cross_entropy_gradient()# 交叉熵梯度验证

# Part 1：线性回归 → 综合评分
train_regression(LinearRegressionModel(5), data)

# Part 2：逻辑回归 → 是否保研（线性边界）
train_binary_classifier(..., BCELoss, predict_logistic)

# Part 3：MLP → 是否保研（非线性边界）
train_binary_classifier(..., CrossEntropyLoss, predict_mlp, hidden=16)

plot_model_comparison(...)   # 柱状图对比 F1
```

**预期现象**（默认超参）：MLP 测试集 F1 通常略高于逻辑回归，因为数据含非线性决策规律。

---

### 调参对照表

| 改什么 | 改哪里 | 观察什么 |
| --- | --- | --- |
| 隐藏层宽度 | `MLPClassifier(..., hidden=4/64)` | `06_MLP_training.png` 的 val F1；`07_model_comparison.png` |
| 学习率 | `train_regression(..., lr=0.5)` 或 `train_binary_classifier(..., lr=1.0)` | loss 曲线是否震荡、不收敛 |
| 训练轮数 | `epochs=50` | loss 是否还未降到位就停止 |

**注意**：始终在**同一份** `graduate_admission.csv` 上对比，否则指标不可比。

---

### 函数速查表

| 函数 / 类 | 行号区间（约） | 讲义 | 一句话 |
| --- | --- | --- | --- |
| `load_dataset_from_csv` | 66–96 | 二 | 读 CSV，拆出 x / y_reg / y_cls |
| `prepare_data` | 179–215 | 二 | 划分 train/val/test + 标准化 |
| `LinearRegressionModel` | 230–245 | 三 | $\hat y = w^Tx+b$ |
| `LogisticRegressionModel` | 248–265 | 七 | $\sigma(w^Tx+b)$ |
| `MLPClassifier` | 268–292 | 九 | 两层 ReLU + 线性输出 |
| `classification_metrics` | 321–341 | 二 | Acc / P / R / F1 |
| `train_regression` | 555–605 | 三·四·五 | MSE + 四步训练循环 |
| `train_binary_classifier` | 608–692 | 七·八·九 | BCE 或 CE + 四步训练循环 |
| `demo_cross_entropy_gradient` | 734–761 | 八 | 验证 $\partial L/\partial z = q-p$ |
| `main` | 770–841 | 十一 | 串联全流程 |
