"""
第一节课综合实战：研究生保研资格预测
====================================

【阅读顺序】建议边看 ../讲义.md 边读本文件，搜索以下标记分段学习：

  【讲义·二】  数据集划分、评估指标
  【讲义·三】  线性回归 + MSE
  【讲义·四】  梯度下降、反向传播
  【讲义·六】  激活函数
  【讲义·七】  逻辑回归
  【讲义·八】  交叉熵
  【讲义·九】  多层感知机 MLP
  【讲义·十一】PyTorch 实战（本文件）

【运行】python graduate_admission_mlp.py

【调参实验】修改 MLPClassifier(hidden=...) 或 train_* 中的 lr，
           在固定 CSV 上对比测试集 F1 / MSE。

【可视化】运行后在 项目/outputs/ 目录生成 PNG 图，便于观察训练过程与效果。

【README 解读】项目/README.md 末尾「代码逐段对照解读」与本文件注释互补，含函数速查与调参表。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # 无 GUI 环境也能保存图片
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# 中文字体（Windows 常见）；若缺字会自动回退，不影响保存
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# 第一部分：数据加载与预处理
# 【讲义·二】数据集与评估指标 —— 训练集 / 验证集 / 测试集划分
# =============================================================================
#
# 原理：机器学习 = 从样本 {(x, y)} 中学习映射 f，使 f(x) ≈ y。
#       必须留出「从未参与训练」的数据来评估泛化能力，否则无法发现过拟合。
#
# 本项目的 CSV 列：
#   特征 x：绩点GPA, 科研经历, 竞赛获奖, 英语成绩, 综合排名分位（均已 0~1 归一化）
#   回归标签：综合评分（0~100）
#   分类标签：保研资格（0/1，约 35% 为正样本）
# =============================================================================

FEATURE_NAMES = ["绩点GPA", "科研经历", "竞赛获奖", "英语成绩", "综合排名分位"]
LABEL_REG = "综合评分"       # 回归任务标签
LABEL_CLS = "保研资格"       # 分类任务标签（0=未获得，1=获得）
NUM_FEATURES = len(FEATURE_NAMES)
DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "graduate_admission.csv"  # 固定 CSV，调参时数据不变
OUTPUT_DIR = DATA_DIR / "outputs"               # 训练曲线等图片保存目录


def load_dataset_from_csv(path: Path = CSV_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 CSV 读取全部样本。

    返回：
        x     — shape (N, 5)，特征矩阵
        y_reg — shape (N,)，回归标签（综合评分）
        y_cls — shape (N,)，分类标签（0/1）

    【讲义·二】监督学习：每个样本都有输入 x 和标签 y，模型从 (x,y) 对学习规律。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"找不到数据文件 {path.name}，请先运行: python generate_dataset_csv.py"
        )

    # 先读表头做校验，防止 CSV 列顺序错乱导致标签对不上特征
    with path.open(encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")
    expected = FEATURE_NAMES + [LABEL_REG, LABEL_CLS]
    if header != expected:
        raise ValueError(f"CSV 表头应为 {expected}，当前为 {header}")

    # skiprows=1 跳过表头；encoding 保证中文列名在 Windows 下可读
    data = np.loadtxt(path, delimiter=",", skiprows=1, encoding="utf-8-sig")

    # 按列切片：前 5 列是特征，第 6 列回归标签，第 7 列分类标签
    x = data[:, :NUM_FEATURES].astype(np.float32)
    y_reg = data[:, NUM_FEATURES].astype(np.float32)
    y_cls = data[:, NUM_FEATURES + 1].astype(np.int64)
    return x, y_reg, y_cls


@dataclass
class DataBundle:
    """
    封装划分 + 归一化后的三组数据，供三个模型共用。

    为什么三个模型共用同一份划分？
      → 只有数据划分一致，逻辑回归 vs MLP 的 F1 对比才有意义。
    """
    x_train: torch.Tensor
    x_val: torch.Tensor
    x_test: torch.Tensor
    y_reg_train: torch.Tensor
    y_reg_val: torch.Tensor
    y_reg_test: torch.Tensor
    y_cls_train: torch.Tensor
    y_cls_val: torch.Tensor
    y_cls_test: torch.Tensor


def train_test_split_stratified(
    x: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    分层划分：正负样本各自按比例切分，再合并。

    【讲义·二】
      - 普通随机划分在类别不平衡时，某折可能几乎没有正样本，指标不稳定。
      - 分层（stratified）保证 train / test 中正样本比例接近总体。

    参数 test_size=0.3 表示 30% 进入 test，70% 进入 train。
    """
    rng = np.random.default_rng(seed)  # 固定 seed → 每次运行划分结果相同

    # 分别取出正类(1)和负类(0)的样本下标
    idx0 = np.where(y_cls == 0)[0]
    idx1 = np.where(y_cls == 1)[0]
    rng.shuffle(idx0)  # 打乱顺序，避免原始 CSV 行序带来偏差
    rng.shuffle(idx1)

    # 两类各取 test_size 比例进测试集，至少保留 1 条（防止极端小样本集为空）
    n_test0 = max(1, int(len(idx0) * test_size))
    n_test1 = max(1, int(len(idx1) * test_size))
    test_idx = np.concatenate([idx0[:n_test0], idx1[:n_test1]])
    train_idx = np.concatenate([idx0[n_test0:], idx1[n_test1:]])

    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return (
        x[train_idx], x[test_idx],
        y_reg[train_idx], y_reg[test_idx],
        y_cls[train_idx], y_cls[test_idx],
    )


def standard_scale(
    x_train: np.ndarray, x_other: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    标准化（Z-score）：x' = (x - mean) / std

    【讲义·三 / 优化技巧】
      - 各特征量纲不同（如 GPA vs 排名）时，梯度下降对不同参数的更新幅度差异大。
      - 标准化后各维特征均值≈0、方差≈1，优化更稳定。
      - **关键**：mean/std 只从训练集计算，再应用到 val/test。
        若用全量数据算均值，等于把测试集信息「泄露」进预处理，指标会虚高。
    """
    mean = x_train.mean(axis=0)   # 每列特征的均值，shape (5,)
    std = x_train.std(axis=0)     # 每列特征的标准差
    std[std == 0] = 1.0           # 常数列除以 0 会出错，置 1 表示不做缩放

    x_train_s = (x_train - mean) / std
    # 验证集、测试集用**训练集**的 mean/std，不能各自重新计算
    others_s = [(arr - mean) / std for arr in x_other]
    return x_train_s, others_s


def prepare_data(seed: int = 42) -> DataBundle:
    """
    完整数据流水线：加载 CSV → 划分 → 标准化 → 转为 PyTorch Tensor。

    划分比例：800 条 → 70% train(560) → 剩余 240 再对半 → val(120) + test(120)
    """
    x, y_reg, y_cls = load_dataset_from_csv()

    # 第一次划分：70% train，30% temp
    x_train, x_temp, y_reg_train, y_reg_temp, y_cls_train, y_cls_temp = train_test_split_stratified(
        x, y_reg, y_cls, test_size=0.3, seed=seed
    )
    # 第二次划分：temp 对半 → val 和 test（各 15% 总量）
    x_val, x_test, y_reg_val, y_reg_test, y_cls_val, y_cls_test = train_test_split_stratified(
        x_temp, y_reg_temp, y_cls_temp, test_size=0.5, seed=seed + 1
    )

    x_train, [x_val, x_test] = standard_scale(x_train, [x_val, x_test])

    # PyTorch 默认用 float32 做神经网络计算；分类标签用 int64（CrossEntropyLoss 要求）
    def to_f32(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float32))

    def to_i64(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.int64))

    return DataBundle(
        x_train=to_f32(x_train),
        x_val=to_f32(x_val),
        x_test=to_f32(x_test),
        y_reg_train=to_f32(y_reg_train),
        y_reg_val=to_f32(y_reg_val),
        y_reg_test=to_f32(y_reg_test),
        y_cls_train=to_i64(y_cls_train),
        y_cls_val=to_i64(y_cls_val),
        y_cls_test=to_i64(y_cls_test),
    )


# =============================================================================
# 第二部分：模型定义
# 【讲义·三】线性回归  【讲义·七】逻辑回归  【讲义·九】MLP
# =============================================================================
#
# PyTorch 约定：
#   - 继承 nn.Module，在 __init__ 里定义可学习参数（层）
#   - 在 forward 里写前向传播公式
#   - 训练时调用 loss.backward() 会自动对 forward 中涉及的参数求梯度（【讲义·四·五】）
# =============================================================================


class LinearRegressionModel(nn.Module):
    """
    【讲义·三】线性回归：ŷ = w^T x + b

    原理：
      - 用一条「直线」（高维即超平面）拟合连续目标，如综合评分。
      - nn.Linear(in_dim, 1) 内部维护权重 W(1×5) 和偏置 b(1)，共 6 个可学习参数。
      - squeeze(-1) 把输出从 (N,1) 变成 (N,)，与标签 y_reg 形状对齐。
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)  # 一个全连接层：5 维输入 → 1 维输出

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)   # 前向：矩阵乘法 + 加偏置


class LogisticRegressionModel(nn.Module):
    """
    【讲义·七】逻辑回归：p = σ(w^T x + b)，σ 为 Sigmoid

    原理：
      - 输出 ∈ (0,1)，可解释为「获得保研资格的概率」。
      - Sigmoid 把线性输出 (-∞,+∞) 压到 (0,1)。
      - 决策：p ≥ 0.5 → 预测为 1，否则为 0。
      - 只能学**线性**决策边界，对「高绩点×低科研」这类非线性规律拟合有限。
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先线性变换，再过 Sigmoid 激活（【讲义·六】Sigmoid 激活函数）
        return torch.sigmoid(self.linear(x).squeeze(-1))


class MLPClassifier(nn.Module):
    """
    【讲义·九】多层感知机（MLP）

    结构：Input(5) → Linear→ReLU → Linear→ReLU → Linear(2) → logits

    原理：
      - 每一层 Linear 做线性变换，ReLU 引入非线性（【讲义·六】）。
      - 堆叠多层后可拟合复杂决策边界，适合本数据集中「绩点×科研」的交互效应。
      - 最后一层输出 2 个 logits（未归一化分数），交给 CrossEntropyLoss 内部做 Softmax。
      - hidden=16：隐藏层宽度；改小(4)易欠拟合，改大(64)可能过拟合，可对比测试 F1。
    """

    def __init__(self, in_dim: int, hidden: int = 16, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),   # 第 1 层：5 → hidden
            nn.ReLU(),                   # ReLU(x)=max(0,x)，负值截断为 0，引入非线性
            nn.Linear(hidden, hidden),   # 第 2 层：hidden → hidden
            nn.ReLU(),
            nn.Linear(hidden, num_classes),  # 输出层：hidden → 2（两类 logits）
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # Sequential 按顺序执行各层


# =============================================================================
# 第三部分：评估指标
# 【讲义·二】分类任务指标 —— Accuracy / Precision / Recall / F1
# =============================================================================
#
# 混淆矩阵四格：
#              预测 0    预测 1
#   真实 0      TN        FP
#   真实 1      FN        TP
#
# Accuracy  = (TP+TN) / 总数        — 整体正确率
# Precision = TP / (TP+FP)          — 预测为正的里有多少真阳性（查准率）
# Recall    = TP / (TP+FN)          — 真实阳性里有多少被找出（查全率）
# F1        = 2·P·R / (P+R)         — P 和 R 的调和平均，类别不平衡时比 Accuracy 更有参考价值
# =============================================================================


@dataclass
class ClsMetrics:
    """封装四个分类指标，便于打印和对比。"""
    accuracy: float
    precision: float
    recall: float
    f1: float


def classification_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> ClsMetrics:
    """
    手算分类指标（对应讲义第二节公式），加深对 TP/FP/TN/FN 的理解。

    注意：本数据集正样本约 35%，Accuracy 可能看起来不错但 F1 更能反映模型质量。
    """
    y_true = y_true.cpu().numpy()  # 转到 CPU 方便用 numpy 布尔索引
    y_pred = y_pred.cpu().numpy()

    # 逐格统计混淆矩阵（讲义第二节符号）
    tp = int(((y_pred == 1) & (y_true == 1)).sum())  # True Positive：预测 1 且真实 1
    tn = int(((y_pred == 0) & (y_true == 0)).sum())  # True Negative
    fp = int(((y_pred == 1) & (y_true == 0)).sum())  # False Positive：误报
    fn = int(((y_pred == 0) & (y_true == 1)).sum())  # False Negative：漏报

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0   # 分母为 0 时置 0，避免除零
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ClsMetrics(accuracy, precision, recall, f1)


def regression_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    【讲义·二 / 三】均方误差 MSE = mean((ŷ - y)²)

    回归任务主指标：越小表示预测越接近真实综合评分。
    """
    return float(torch.mean((y_true - y_pred) ** 2))


# =============================================================================
# 3.5 可视化（观察训练效果）
# 【讲义·二/三/四/六】损失下降、拟合效果、激活函数形状、模型对比
# =============================================================================


@dataclass
class TrainHistory:
    """记录每个 epoch 的指标，用于绘制训练曲线。"""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)   # 回归：val MSE；分类：val loss
    val_f1: list[float] = field(default_factory=list)   # 仅分类任务使用
    val_acc: list[float] = field(default_factory=list)    # 仅分类任务使用


def _save_fig(fig: plt.Figure, filename: str) -> Path:
    """保存图片到 outputs/ 并关闭 figure，避免内存泄漏。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] 已保存 → {path.relative_to(DATA_DIR)}")
    return path


def plot_activation_curves() -> None:
    """
    【讲义·六】绘制 Sigmoid / ReLU / Tanh 曲线。

    直观理解：ReLU 在负半轴为 0（引入非线性），Sigmoid 把输出压到 (0,1)。
    """
    x = np.linspace(-3, 3, 300)
    xt = torch.tensor(x, dtype=torch.float32)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, torch.sigmoid(xt).numpy(), label="Sigmoid", linewidth=2)
    ax.plot(x, torch.relu(xt).numpy(), label="ReLU", linewidth=2)
    ax.plot(x, torch.tanh(xt).numpy(), label="Tanh", linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("激活函数对比（讲义第六节）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, "01_activation_functions.png")


def plot_regression_loss(history: TrainHistory) -> None:
    """【讲义·三/四】线性回归：训练 MSE vs 验证 MSE 随 epoch 变化。"""
    epochs = range(1, len(history.train_loss) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history.train_loss, label="训练集 MSE", linewidth=1.5)
    ax.plot(epochs, history.val_loss, label="验证集 MSE", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("线性回归 — 损失曲线（Part 1）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, "02_regression_loss_curve.png")


def plot_regression_scatter(
    y_true: torch.Tensor, y_pred: torch.Tensor, test_mse: float
) -> None:
    """
    【讲义·三】真实综合评分 vs 预测值散点图。

    点越贴近对角线 y=x，拟合越好；系统性偏离说明模型有偏差。
    """
    yt = y_true.cpu().numpy()
    yp = y_pred.cpu().numpy()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(yt, yp, alpha=0.6, edgecolors="k", linewidths=0.3, s=40)
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="理想预测 y=x")
    ax.set_xlabel("真实综合评分")
    ax.set_ylabel("预测综合评分")
    ax.set_title(f"线性回归 — 测试集拟合（MSE={test_mse:.2f}）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    _save_fig(fig, "03_regression_scatter.png")


def plot_classifier_training(
    history: TrainHistory, title: str, filename: str
) -> None:
    """
    【讲义·四/七/九】分类模型：左图画 loss 下降，右图画验证集 Acc / F1 上升。

    若 train loss 持续降而 val F1 停滞或下降 → 可能过拟合（讲义·三 欠拟合/过拟合）。
    """
    epochs = range(1, len(history.train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history.train_loss, color="#1f77b4", linewidth=1.5)
    ax1.plot(epochs, history.val_loss, color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("损失曲线")
    ax1.legend(["训练 loss", "验证 loss"])
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history.val_acc, label="验证 Acc", linewidth=1.5)
    ax2.plot(epochs, history.val_f1, label="验证 F1", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("指标")
    ax2.set_title("验证集分类指标")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    _save_fig(fig, filename)


def plot_confusion_matrix(
    y_true: torch.Tensor, y_pred: torch.Tensor, title: str, filename: str
) -> None:
    """【讲义·二】混淆矩阵热力图：TN/FP/FN/TP 四格一目了然。"""
    yt = y_true.cpu().numpy()
    yp = y_pred.cpu().numpy()
    tp = int(((yp == 1) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["预测 0", "预测 1"])
    ax.set_yticklabels(["真实 0", "真实 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save_fig(fig, filename)


def plot_feature_scatter(data: DataBundle) -> None:
    """
    用前两个特征（绩点 GPA vs 科研经历）展示数据分布。

    颜色 = 是否保研；帮助学生理解分类任务在做什么。
    """
    # 取测试集原始特征的前两维（已标准化，但相对位置仍保留聚类结构）
    x = data.x_test[:, :2].cpu().numpy()
    y = data.y_cls_test.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, name, color in [(0, "未保研 (0)", "#1f77b4"), (1, "保研 (1)", "#d62728")]:
        mask = y == label
        ax.scatter(x[mask, 0], x[mask, 1], alpha=0.7, label=name, c=color, edgecolors="k", s=45)
    ax.set_xlabel(FEATURE_NAMES[0])
    ax.set_ylabel(FEATURE_NAMES[1])
    ax.set_title("测试集样本分布（二维投影）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, "04_feature_scatter.png")


def plot_model_comparison(log_m: ClsMetrics, mlp_m: ClsMetrics) -> None:
    """【讲义·二】逻辑回归 vs MLP 测试集指标柱状对比。"""
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["准确率", "精确率", "召回率", "F1"]
    log_vals = [log_m.accuracy, log_m.precision, log_m.recall, log_m.f1]
    mlp_vals = [mlp_m.accuracy, mlp_m.precision, mlp_m.recall, mlp_m.f1]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, log_vals, width, label="逻辑回归", color="#1f77b4")
    ax.bar(x + width / 2, mlp_vals, width, label="MLP", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("分数")
    ax.set_title("测试集模型对比（Part 2 vs Part 3）")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, "07_model_comparison.png")


# =============================================================================
# 第四部分：训练循环
# 【讲义·四】梯度下降  【讲义·五】反向传播
# =============================================================================
#
# 每一轮 epoch 的四步（讲义训练循环图）：
#
#   1. 前向传播 forward  → 得到预测 ŷ
#   2. 计算损失 loss     → 衡量 ŷ 与 y 的差距
#   3. 反向传播 backward → 链式法则求各参数梯度 ∂L/∂w
#   4. 参数更新 step     → w ← w - lr · ∂L/∂w  （梯度下降）
#
# optimizer.zero_grad() 必须在 backward 前调用，否则梯度会累加上一轮的值。
# =============================================================================


def train_regression(model: nn.Module, data: DataBundle, epochs: int = 200, lr: float = 0.05) -> float:
    """
    【Part 1 / 讲义·三】训练线性回归，预测综合评分。

    损失函数：MSE（均方误差）
    优化器：SGD（随机梯度下降），本例用全量 train 批（批大小 = 560）
    lr：学习率，控制每步更新幅度；过大则 loss 震荡，过小则收敛慢。
    """
    criterion = nn.MSELoss()                          # L = mean((ŷ - y)²)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # 要更新的就是 model 里所有 W, b
    history = TrainHistory()

    print("\n" + "=" * 60)
    print("【Part 1】线性回归 —— 预测综合评分（MSE 损失）")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()  # 训练模式（本模型无 Dropout/BN，主要是语义标记）

        # --- 步骤 1：前向传播 ---
        pred = model(data.x_train)                   # ŷ，shape (560,)
        # --- 步骤 2：计算损失 ---
        loss = criterion(pred, data.y_reg_train)     # 标量，可对其求导

        # --- 步骤 3：反向传播（PyTorch 自动求导，对应讲义链式法则）---
        optimizer.zero_grad()   # 清空上一步累积的梯度
        loss.backward()         # 从 loss 往回传，计算每个参数的 ∂L/∂θ
        # --- 步骤 4：梯度下降更新参数 ---
        optimizer.step()        # θ ← θ - lr · ∂L/∂θ

        # 每个 epoch 记录指标，用于绘制 loss 曲线
        model.eval()
        with torch.no_grad():
            val_pred = model(data.x_val)
            val_mse = regression_mse(data.y_reg_val, val_pred)
        history.train_loss.append(loss.item())
        history.val_loss.append(val_mse)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train MSE = {loss.item():.4f} | val MSE = {val_mse:.4f}")

    # 最终在测试集上评估（测试集全程未参与训练和调参）
    model.eval()
    with torch.no_grad():
        test_pred = model(data.x_test)
        test_mse = regression_mse(data.y_reg_test, test_pred)
    print(f"  测试集 MSE = {test_mse:.4f}")

    plot_regression_loss(history)
    plot_regression_scatter(data.y_reg_test, test_pred, test_mse)
    return test_mse


def train_binary_classifier(
    name: str,
    model: nn.Module,
    data: DataBundle,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    predict_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    epochs: int = 300,
    lr: float = 0.1,
    plot_prefix: str = "05",
) -> ClsMetrics:
    """
    【Part 2 / Part 3】训练二分类模型（逻辑回归 或 MLP）。

    逻辑回归用 BCELoss（输入概率 + float 标签）；
    MLP 用 CrossEntropyLoss（输入 logits + long 标签，内部含 Softmax）。

    【讲义·八】交叉熵衡量预测分布与真实分布的差距；梯度形式简洁：∂L/∂z = q - p。
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = TrainHistory()

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()

        if isinstance(model, LogisticRegressionModel):
            # 逻辑回归：forward 已输出 Sigmoid 概率 ∈ (0,1)
            prob = model(data.x_train)
            # BCELoss 要求标签为 float，与概率同类型
            loss = loss_fn(prob, data.y_cls_train.float())
        else:
            # MLP：forward 输出 logits（未 Softmax 的原始分数）
            logits = model(data.x_train)
            # CrossEntropyLoss 内部会做 Softmax + 负对数，数值更稳定
            loss = loss_fn(logits, data.y_cls_train)

        optimizer.zero_grad()
        loss.backward()   # 反向传播：从 loss 一路求导到每层 W, b
        optimizer.step()

        # 每 epoch 记录在验证集上的 loss / Acc / F1
        model.eval()
        with torch.no_grad():
            if isinstance(model, LogisticRegressionModel):
                val_prob = model(data.x_val)
                val_loss = float(loss_fn(val_prob, data.y_cls_val.float()))
            else:
                val_logits = model(data.x_val)
                val_loss = float(loss_fn(val_logits, data.y_cls_val))
            val_pred = predict_fn(model, data.x_val)
            val_m = classification_metrics(data.y_cls_val, val_pred)

        history.train_loss.append(loss.item())
        history.val_loss.append(val_loss)
        history.val_f1.append(val_m.f1)
        history.val_acc.append(val_m.accuracy)

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | loss = {loss.item():.4f} | "
                f"val acc = {val_m.accuracy:.3f} | val F1 = {val_m.f1:.3f}"
            )

    model.eval()
    with torch.no_grad():
        test_pred = predict_fn(model, data.x_test)
        test_m = classification_metrics(data.y_cls_test, test_pred)

    print(
        f"  测试集 | Acc={test_m.accuracy:.3f}  "
        f"Prec={test_m.precision:.3f}  Rec={test_m.recall:.3f}  F1={test_m.f1:.3f}"
    )

    short = "logistic" if isinstance(model, LogisticRegressionModel) else "MLP"
    title_cn = "逻辑回归" if short == "logistic" else "MLP"
    plot_classifier_training(history, f"{title_cn} — 训练过程", f"{plot_prefix}_{short}_training.png")
    plot_confusion_matrix(
        data.y_cls_test, test_pred,
        f"{title_cn} — 测试集混淆矩阵",
        f"{plot_prefix}_{short}_confusion_matrix.png",
    )
    return test_m


def predict_logistic(model: LogisticRegressionModel, x: torch.Tensor) -> torch.Tensor:
    """逻辑回归推断：概率 ≥ 0.5 判为正类（【讲义·七】）。"""
    return (model(x) >= 0.5).long()


def predict_mlp(model: MLPClassifier, x: torch.Tensor) -> torch.Tensor:
    """MLP 推断：取 logits 最大值的类别下标（等价于 Softmax 后概率最大的类）。"""
    return model(x).argmax(dim=1)


# =============================================================================
# 第五部分：讲义配套小实验（激活函数 & 交叉熵梯度）
# =============================================================================


def demo_activation_functions() -> None:
    """
    【讲义·六】激活函数数值验证

    对 x = 0.5 和 x = -0.5 分别计算 Sigmoid / ReLU / Tanh：
      - Sigmoid：平滑映射到 (0,1)，x=0 时为 0.5
      - ReLU：负值变 0，正值不变 → MLP 中引入「分段线性」非线性
      - Tanh：零中心，输出 (-1,1)
    """
    print("\n" + "=" * 60)
    print("【补充】激活函数数值例子（讲义第六节练习题）")
    print("=" * 60)
    x = torch.tensor([0.5, -0.5])
    sigmoid = torch.sigmoid(x)
    relu = torch.relu(x)
    tanh = torch.tanh(x)
    for i, xi in enumerate(x.tolist()):
        print(
            f"  x={xi:+.1f}  |  Sigmoid={sigmoid[i]:.4f}  "
            f"ReLU={relu[i]:.4f}  Tanh={tanh[i]:.4f}"
        )
    plot_activation_curves()  # 保存激活函数曲线图到 outputs/


def demo_cross_entropy_gradient() -> None:
    """
    【讲义·八】验证 Softmax + 交叉熵的梯度：∂L/∂z = q - p

    q = Softmax(z) 为预测概率，p 为 one-hot 真实标签。
    手动算 (q - p) 与 autograd 结果对比，印证讲义公式。
    """
    print("\n" + "=" * 60)
    print("【补充】Softmax + 交叉熵梯度验证（讲义第八节）")
    print("=" * 60)

    # requires_grad=True：告诉 PyTorch 要追踪此张量的梯度
    logits = torch.tensor([[0.1, 0.2, 0.9], [1.1, 0.1, 0.2]], requires_grad=True)
    target = torch.tensor([2, 0])  # 样本 0 真实类别 2，样本 1 真实类别 0

    # reduction='sum'：loss 为各样本 CE 之和，便于与手推公式逐项对照
    loss = nn.CrossEntropyLoss(reduction="sum")(logits, target)
    loss.backward()  # 自动求 ∂loss/∂logits

    prob = torch.softmax(logits, dim=1)       # q：预测概率
    one_hot = torch.zeros_like(prob)
    one_hot.scatter_(1, target.unsqueeze(1), 1.0)  # p：one-hot 真实标签
    grad_manual = prob - one_hot              # 讲义结论：梯度 = q - p

    print(f"  loss = {loss.item():.4f}")
    print(f"  手动梯度 (q - p):\n{grad_manual.detach().numpy()}")
    print(f"  autograd 梯度:\n{logits.grad.numpy()}")
    print("  => 两者一致，验证了 dL/dz = q - p")


# =============================================================================
# 第六部分：主流程 —— 串联三个模型并对比
# 【讲义·十一】PyTorch 实战项目
# =============================================================================


def main() -> None:
    # 固定随机种子：PyTorch 初始化权重可复现（数据划分在 prepare_data 里也已固定 seed）
    torch.manual_seed(42)
    data = prepare_data()

    print("=" * 60)
    print("第一节课 PyTorch 综合项目：研究生保研资格预测")
    print("=" * 60)
    print(f"数据文件: {CSV_PATH.name}（固定数据集，调参结果可对比）")
    print(f"特征: {FEATURE_NAMES}")
    print(
        f"样本量: train={len(data.x_train)}, val={len(data.x_val)}, test={len(data.x_test)}"
    )
    print(f"保研正样本比例: {data.y_cls_train.float().mean():.2%} (训练集)")

    plot_feature_scatter(data)  # 数据分布图

    # 先做两个小实验，再进入三个 Part 的主线训练
    demo_activation_functions()
    demo_cross_entropy_gradient()

    # ------------------------------------------------------------------
    # Part 1：线性回归 → 预测「综合评分」（连续值，MSE 损失）
    # 【讲义·三】
    # ------------------------------------------------------------------
    reg_model = LinearRegressionModel(NUM_FEATURES)
    train_regression(reg_model, data)

    # ------------------------------------------------------------------
    # Part 2：逻辑回归 → 预测「是否保研」（0/1，Sigmoid + BCE）
    # 【讲义·七 / 八】BCE 即二分类交叉熵
    # ------------------------------------------------------------------
    bce = nn.BCELoss()
    log_model = LogisticRegressionModel(NUM_FEATURES)
    log_metrics = train_binary_classifier(
        "【Part 2】逻辑回归 —— Sigmoid + 二分类交叉熵",
        log_model,
        data,
        loss_fn=bce,
        predict_fn=predict_logistic,
        plot_prefix="05",
    )

    # ------------------------------------------------------------------
    # Part 3：MLP → 同样预测「是否保研」，但用 ReLU 非线性 + CrossEntropyLoss
    # 【讲义·九 / 六 / 八】
    # 数据含「高绩点×低科研」非线性规律，MLP 通常 F1 高于逻辑回归
    # 实验：改 hidden=4 或 64，或 lr=1.0，对比下方测试集 F1
    # ------------------------------------------------------------------
    ce = nn.CrossEntropyLoss()
    mlp_model = MLPClassifier(NUM_FEATURES, hidden=16, num_classes=2)
    mlp_metrics = train_binary_classifier(
        "【Part 3】多层感知机 MLP —— ReLU + Softmax/交叉熵",
        mlp_model,
        data,
        loss_fn=ce,
        predict_fn=predict_mlp,
        plot_prefix="06",
    )

    plot_model_comparison(log_metrics, mlp_metrics)

    # 同一测试集上对比两个分类模型（【讲义·二】F1 为主要对比指标）
    print("\n" + "=" * 60)
    print("模型对比（测试集 F1）")
    print("=" * 60)
    print(f"  逻辑回归 F1 = {log_metrics.f1:.3f}")
    print(f"  MLP       F1 = {mlp_metrics.f1:.3f}")
    if mlp_metrics.f1 >= log_metrics.f1:
        print("  => MLP 通过 ReLU 非线性激活，拟合更复杂的决策边界。")
    print(f"\n全部图表已保存至: {OUTPUT_DIR.relative_to(DATA_DIR)}/")
    print("完成！请对照 ../讲义.md 各【讲义·X】章节理解每一行代码的含义。")


if __name__ == "__main__":
    # 仅当直接运行本文件时执行 main；被其他模块 import 时不自动训练
    main()
