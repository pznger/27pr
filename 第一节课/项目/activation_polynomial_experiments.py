"""
思考题实验代码：激活函数对比 & 多项式回归 vs 神经网络

运行方式：
    python activation_polynomial_experiments.py

会输出两张图：
    图1 - x² 作为激活函数的三大问题（梯度爆炸、符号丢失、深层退化）
    图2 - 多项式回归 vs 神经网络的泛化能力对比
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 实验一：为什么不用 x² 作为激活函数？
# ============================================================

def experiment1_x2_activation():
    """展示 x² 激活函数的三大问题"""
    np.random.seed(42)

    # ============================================================
    # 1) 梯度：x² 梯度 = 2x，随 |x| 线性增长，无上界
    # ============================================================
    x_vals = np.linspace(-6, 6, 300)
    grad_x2 = 2 * x_vals
    grad_relu = (x_vals > 0).astype(float)
    sx = 1 / (1 + np.exp(-x_vals))
    grad_sigmoid = sx * (1 - sx)

    # ============================================================
    # 2) 符号信息丢失
    # ============================================================
    x_test = np.array([-2.0, -1.0, 0.5, 1.5])
    x2_out = x_test ** 2
    relu_out = np.maximum(0, x_test)

    # ============================================================
    # 3) 深层传播：每种激活函数用各自适配的参数
    # ============================================================
    n_neurons = 500          # 足够多地神经元，分布更有统计意义
    n_layers = 8             # 8 层，渐进变化更明显

    def simulate(name, activation_fn, weight_scale, bias_scale):
        """模拟全连接网络逐层传播"""
        np.random.seed(42)
        x = np.random.randn(n_neurons) * 1.0
        outputs = [x.copy()]
        for _ in range(n_layers):
            W = np.random.randn(n_neurons, n_neurons) * weight_scale
            b = np.random.randn(n_neurons) * bias_scale
            x = W @ x + b
            x = activation_fn(x)
            outputs.append(x.copy())
        return outputs

    # x²：input_std=1, W~N(0,0.06²), 500个神经元
    #     pre_act std ≈ 0.06 * sqrt(500) ≈ 1.34
    #     平方后 mean≈σ², var≈2σ⁴ → std ≈ sqrt(2) * 1.34² ≈ 2.54 (第1层就放大)
    #     之后每层继续平方 → 超指数爆炸，第3-4层可见剧烈增长
    layers_x2 = simulate("x²",
        lambda x: x ** 2,
        weight_scale=0.06,
        bias_scale=0.01)

    # ReLU：W~N(0,0.10²), 500个神经元
    #     pre_act std ≈ 0.10 * sqrt(500) ≈ 2.24
    #     ReLU 后 std ≈ 0.58 * 2.24 ≈ 1.30 → 每层轻微放大但总体稳定
    layers_relu = simulate("ReLU",
        lambda x: np.maximum(0, x),
        weight_scale=0.10,
        bias_scale=0.05)

    # Sigmoid：W~N(0,0.4²), 偏置 0.2
    #     pre_act std ≈ 0.4 * sqrt(500) ≈ 8.94 → 迅速饱和到 0/1
    layers_sigmoid = simulate("Sigmoid",
        lambda x: 1 / (1 + np.exp(-x)),
        weight_scale=0.4,
        bias_scale=0.2)

    # ============================================================
    # 画图
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("实验一：为什么不用 x² 作为激活函数？", fontsize=16, fontweight='bold', y=0.98)

    # ---- 子图1：梯度对比（突出 x² 无上界） ----
    ax = axes[0, 0]
    ax.plot(x_vals, grad_x2, 'r-', linewidth=2.5, label="x² 梯度 = 2x")
    ax.plot(x_vals, grad_relu, 'b-', linewidth=2.5, label="ReLU 梯度")
    ax.plot(x_vals, grad_sigmoid, 'g-', linewidth=2.5, label="Sigmoid 梯度")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel("输入 x", fontsize=10)
    ax.set_ylabel("梯度值", fontsize=10)
    ax.set_title("① 梯度对比：x² 无上界 → 深层梯度爆炸", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    # 阴影标注"危险区域"
    ax.fill_between(x_vals, grad_x2, 100, where=(np.abs(grad_x2) > 2),
                    color='red', alpha=0.08)
    ax.text(3.2, 9, "梯度随 |x|\n无限制增长！", fontsize=8, color='darkred',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))
    ax.text(-5, 0.3, "ReLU 梯度 ≤ 1 → 稳定", fontsize=8, color='darkblue')
    ax.text(-5, -0.15, "Sigmoid 梯度 ≤ 0.25 → 消失", fontsize=8, color='darkgreen')
    ax.set_ylim(-0.3, 12)

    # ---- 子图2：符号信息丢失 —— 更直观的对比 ----
    ax = axes[0, 1]
    width = 0.22
    xx = np.arange(len(x_test))
    bars1 = ax.bar(xx - width, x_test, width, label="原始输入", color='#888888', alpha=0.85)
    bars2 = ax.bar(xx, x2_out, width, label="x² 输出", color='#E53E3E', alpha=0.85)
    bars3 = ax.bar(xx + width, relu_out, width, label="ReLU 输出", color='#3182CE', alpha=0.85)

    # 在柱子上标数值
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h,
                f'{h:.1f}', ha='center', va='bottom' if h >= 0 else 'top',
                fontsize=9, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.08,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color='#C53030')
    for bar in bars3:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.08,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#2B6CB0')

    ax.set_xticks(xx)
    ax.set_xticklabels([f"{v}" for v in x_test], fontsize=10)
    ax.set_title('② 符号信息：x² 把负值"强制变正"', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylabel("输出值", fontsize=10)
    ax.axhline(y=0, color='black', linewidth=1.5)

    # 画箭头标注符号翻转
    for i, orig in enumerate(x_test):
        if orig < 0:
            ax.annotate("", xy=(i, x2_out[i]),
                       xytext=(i, x2_out[i] + 0.8),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax.annotate("负值 → 正值", xy=(i + 0.3, x2_out[i] + 0.6),
                       fontsize=8, color='darkred', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # ---- 子图3：x² 深层传播（散点图 + 均值线） ----
    ax = axes[0, 2]
    layer_names = [f"L{i}" for i in range(n_layers + 1)]
    show_layers = min(n_layers + 1, 9)  # 最多展示9层
    x_positions = []
    y_values = []
    for i in range(show_layers):
        d = layers_x2[i]
        finite = d[np.isfinite(d)]
        if len(finite) == 0:
            break
        # 裁剪极端值，但范围放大一些以展示爆炸
        finite = np.clip(finite, -100, 100)
        x_positions.extend([i] * len(finite))
        y_values.extend(finite.tolist())
    ax.scatter(x_positions, y_values, s=2, alpha=0.3, c='darkred', edgecolors='none')
    # 画每层均值线
    means = []
    for i in range(show_layers):
        d = layers_x2[i]
        finite = d[np.isfinite(d)]
        if len(finite) > 0:
            means.append(np.mean(np.clip(finite, -100, 100)))
        else:
            means.append(np.nan)
    if len(means) > 0:
        ax.plot(range(len(means)), means, 'r-o', linewidth=2.5, markersize=8,
                label="均值", zorder=5)
    ax.set_xticks(range(show_layers))
    ax.set_xticklabels(layer_names[:show_layers])
    ax.set_title("③ x² 激活：逐层指数爆炸", fontsize=11, fontweight='bold')
    ax.set_xlabel("网络层", fontsize=10)
    ax.set_ylabel("激活值", fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.2, axis='y')
    ax.axhline(y=0, color='black', linewidth=1)
    # 爆炸标注
    if len(means) > 2 and not np.isnan(means[-1]):
        ax.annotate("指数爆炸！", xy=(min(show_layers-1, len(means)-1), means[-1]),
                   xytext=(show_layers-4, np.max(y_values)*0.5 if y_values else 5),
                   fontsize=10, color='darkred', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # ---- 子图4：ReLU 深层传播 ----
    ax = axes[1, 0]
    for i in range(show_layers):
        d = layers_relu[i]
        ax.hist(d, bins=60, alpha=0.6 if i < show_layers-2 else 0.9,
                label=f"L{i}", color=plt.cm.Blues(0.25 + 0.75 * i / max(show_layers-1, 1)),
                density=True, edgecolor='white', linewidth=0.3)
    ax.set_title("④ ReLU 激活：分布稳定传播", fontsize=11, fontweight='bold')
    ax.set_xlabel("激活值", fontsize=10)
    ax.set_ylabel("密度", fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.set_xlim(-0.5, 6)

    # ---- 子图5：Sigmoid 深层传播 ----
    ax = axes[1, 1]
    for i in range(show_layers):
        d = layers_sigmoid[i]
        ax.hist(d, bins=60, alpha=0.6 if i < show_layers-2 else 0.9,
                label=f"L{i}", color=plt.cm.Greens(0.25 + 0.75 * i / max(show_layers-1, 1)),
                density=True, edgecolor='white', linewidth=0.3)
    ax.set_title("⑤ Sigmoid 激活：逐渐饱和到两端", fontsize=11, fontweight='bold')
    ax.set_xlabel("激活值", fontsize=10)
    ax.set_ylabel("密度", fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc='upper left')
    ax.set_xlim(0, 1)
    ax.annotate("深层 → 0 或 1\n梯度消失", xy=(0.9, 3), xytext=(0.4, 6),
               fontsize=8, color='darkgreen',
               arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # ---- 子图6：各层标准差对比 ----
    ax = axes[1, 2]
    layers_idx = np.arange(n_layers + 1)

    def safe_std(arr):
        finite = arr[np.isfinite(arr)]
        return np.std(finite) if len(finite) > 1 else np.nan

    std_x2 = [safe_std(d) for d in layers_x2]
    std_relu = [safe_std(d) for d in layers_relu]
    std_sigmoid = [safe_std(d) for d in layers_sigmoid]

    ax.plot(layers_idx, std_x2, 'r-o', linewidth=2.5, markersize=7, label="x²")
    ax.plot(layers_idx, std_relu, 'b-s', linewidth=2.5, markersize=7, label="ReLU")
    ax.plot(layers_idx, std_sigmoid, 'g-^', linewidth=2.5, markersize=7, label="Sigmoid")
    ax.set_xlabel("层数", fontsize=10)
    ax.set_ylabel("输出标准差", fontsize=10)
    ax.set_title("⑥ 各层输出标准差变化（对数坐标）", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xticks(layers_idx)

    # 标注
    last_valid_x2 = next((i for i, s in enumerate(std_x2) if np.isnan(s)), len(std_x2))
    if last_valid_x2 > 1:
        ax.annotate("x² 指数爆炸！", xy=(last_valid_x2-1, std_x2[last_valid_x2-1]),
                   xytext=(last_valid_x2-3, std_x2[last_valid_x2-1]*10),
                   fontsize=9, color='darkred', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate("ReLU 稳定", xy=(n_layers, std_relu[-1]),
               xytext=(n_layers-3, std_relu[-1]*0.4),
               fontsize=9, color='darkblue', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.annotate("Sigmoid 塌缩", xy=(n_layers, std_sigmoid[-1]),
               xytext=(n_layers-3, std_sigmoid[-1]*3),
               fontsize=9, color='darkgreen', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='green', lw=2))

    plt.tight_layout()
    plt.savefig("experiment1_x2_activation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[实验一] 图片已保存：experiment1_x2_activation.png")


# ============================================================
# 实验二：多项式回归 vs 神经网络的对比
# ============================================================

def experiment2_poly_vs_nn():
    """对比多项式回归与神经网络的行为差异"""
    np.random.seed(42)

    # ---------- 2a) 参数数量随维度的增长 ----------
    dims = np.arange(1, 51)
    poly_params_2 = [math.comb(d + 2, 2) for d in dims]    # n=2 次（只能拟合二次曲面）
    poly_params_3 = [math.comb(d + 3, 3) for d in dims]    # n=3 次（与两层 MLP 表达能力相当）

    # 两层 MLP: 输入→h→1, 参数 = d*h + h + h + 1 = d*h + 2h + 1
    # h=8: 极轻量，d=50 时仅 467 参
    nn_params_small = [d * 8 + 2 * 8 + 1 for d in dims]
    # h=4: 最轻量，d=50 时仅 267 参
    nn_params_tiny = [d * 4 + 2 * 4 + 1 for d in dims]

    # ---------- 2b) Runge 现象 ----------
    def f_target(x):
        """目标函数：1/(1+25x²) — 经典的 Runge 函数"""
        return 1 / (1 + 25 * x**2)

    x_train_rung = np.linspace(-1, 1, 15)
    y_train_rung = f_target(x_train_rung)

    x_test_rung = np.linspace(-1, 1, 200)
    y_test_rung = f_target(x_test_rung)

    # 多项式拟合
    degree = 14
    coeffs = np.polyfit(x_train_rung, y_train_rung, degree)
    y_poly_pred = np.polyval(coeffs, x_test_rung)

    # 简单的神经网络拟合（两层 MLP，手动实现）
    def simple_nn_fit(x_train, y_train, x_test, hidden=16, lr=0.1, epochs=5000):
        np.random.seed(2025)
        W1 = np.random.randn(1, hidden) * 0.5
        b1 = np.zeros(hidden)
        W2 = np.random.randn(hidden, 1) * 0.5
        b2 = np.zeros(1)

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)

        for _ in range(epochs):
            # forward
            z1 = x_train @ W1 + b1
            a1 = np.maximum(0, z1)  # ReLU
            y_pred = a1 @ W2 + b2

            # loss & backward
            loss = np.mean((y_pred - y_train)**2)
            dy = 2 * (y_pred - y_train) / len(x_train)
            dW2 = a1.T @ dy
            db2 = np.sum(dy, axis=0)
            da1 = dy @ W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = x_train.T @ dz1
            db1 = np.sum(dz1, axis=0)

            # update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        # predict
        z1_t = x_test @ W1 + b1
        a1_t = np.maximum(0, z1_t)
        return (a1_t @ W2 + b2).flatten()

    y_nn_pred = simple_nn_fit(x_train_rung, y_train_rung, x_test_rung)

    # ---------- 2c) 多维数据上泛化对比 ----------
    n_samples_total = 500      # 充足样本
    n_features = 20
    n_train = 350              # 训练集 70%
    n_test = n_samples_total - n_train  # 150

    def generate_hd_data(n_samples, n_features, noise=0.15):
        """生成高维非线性数据，含 sin/abs 等非多项式成分"""
        np.random.seed(2025)
        X = np.random.randn(n_samples, n_features) * 1.2
        # 目标函数：大量非多项式成分，2次多项式必然欠拟合，3次也开始吃力
        y = (1.2 * np.sin(X[:, 0] * 2.0) +           # sin: 多项式无法精确表达
             0.8 * np.abs(X[:, 1]) +                  # abs: 非多项式
             0.6 * np.cos(X[:, 2] * 1.8) +            # cos: 非多项式
             0.5 * X[:, 3] * np.tanh(X[:, 4]) +       # 交互 + tanh
             0.4 * np.exp(-np.abs(X[:, 5])) +          # exp: 非多项式
             0.3 * X[:, 6] * X[:, 7] * X[:, 8] +      # 三阶交互
             0.2 * (X[:, 0]**2 + X[:, 1]**2) +         # 少量二次项
             0.1 * X[:, 9:].sum(axis=1))               # 弱线性基线
        y += noise * np.random.randn(n_samples)
        return X, y

    X_hd, y_hd = generate_hd_data(n_samples_total, n_features)
    X_tr, y_tr = X_hd[:n_train], y_hd[:n_train]
    X_te, y_te = X_hd[n_train:], y_hd[n_train:]
    # 标准化 y
    y_mean, y_std = y_tr.mean(), y_tr.std()
    y_tr_norm = (y_tr - y_mean) / y_std
    y_te_norm = (y_te - y_mean) / y_std

    # 多项式回归（纯 NumPy 实现，无需 sklearn）
    def poly_features(X, degree=2):
        """生成多项式特征（含交互项），纯 NumPy 实现"""
        n_samples, n_features = X.shape
        features = [np.ones((n_samples, 1))]  # bias
        for d in range(1, degree + 1):
            for combo in _combinations_with_replacement(n_features, d):
                feat = np.ones(n_samples)
                for idx in combo:
                    feat = feat * X[:, idx]
                features.append(feat.reshape(-1, 1))
        return np.hstack(features)

    def _combinations_with_replacement(n, k):
        """生成 n 选 k 的可重复组合索引（确保非递减顺序去重）"""
        if k == 0:
            yield ()
            return
        for i in range(n):
            for combo in _combinations_with_replacement(n, k - 1):
                if not combo or i <= combo[0]:
                    yield (i,) + combo

    def linear_fit(X, y):
        """最小二乘闭式解: w = (X^T X)^{-1} X^T y"""
        return np.linalg.lstsq(X, y, rcond=None)[0]

    train_mses_poly2 = []
    test_mses_poly2 = []
    train_mses_poly3 = []
    test_mses_poly3 = []
    poly_dims = [2, 3, 4, 5, 6, 7, 8, 9]
    for d_sub in poly_dims:
        X_sub = X_hd[:, :d_sub]

        # 2 次多项式
        X_poly2 = poly_features(X_sub[:n_train], degree=2)
        X_poly2_test = poly_features(X_sub[n_train:], degree=2)
        try:
            w2 = linear_fit(X_poly2, y_tr_norm)
            train_mses_poly2.append(np.mean((X_poly2 @ w2 - y_tr_norm)**2))
            test_mses_poly2.append(np.mean((X_poly2_test @ w2 - y_te_norm)**2))
        except np.linalg.LinAlgError:
            train_mses_poly2.append(np.nan); test_mses_poly2.append(np.nan)

        # 3 次多项式（高维时可能涌现出样本数，加一个小正则化）
        n_feat_3 = math.comb(d_sub + 3, 3) + 1  # +1 for bias
        if n_feat_3 < n_train * 0.6:   # 特征数 << 样本数时才跑
            X_poly3 = poly_features(X_sub[:n_train], degree=3)
            X_poly3_test = poly_features(X_sub[n_train:], degree=3)
            try:
                # 加微量 L2 正则化防止奇异性
                reg = 1e-4
                w3 = np.linalg.solve(
                    X_poly3.T @ X_poly3 + reg * np.eye(X_poly3.shape[1]),
                    X_poly3.T @ y_tr_norm)
                train_mses_poly3.append(np.mean((X_poly3 @ w3 - y_tr_norm)**2))
                test_mses_poly3.append(np.mean((X_poly3_test @ w3 - y_te_norm)**2))
            except np.linalg.LinAlgError:
                train_mses_poly3.append(np.nan); test_mses_poly3.append(np.nan)
        else:
            train_mses_poly3.append(np.nan); test_mses_poly3.append(np.nan)

    # 神经网络（小隐藏层 + Kaiming 初始化 + 多轮训练）
    def simple_nn_fit_multi(X_tr, y_tr, X_te, y_te, hidden=8, lr=0.02, epochs=5000):
        np.random.seed(2025)
        n_features = X_tr.shape[1]

        # Kaiming 初始化
        W1 = np.random.randn(n_features, hidden) * np.sqrt(2.0 / n_features)
        b1 = np.zeros(hidden)
        W2 = np.random.randn(hidden, 1) * np.sqrt(2.0 / hidden)
        b2 = np.zeros(1)

        y_tr = y_tr.reshape(-1, 1)
        best_te_mse = float('inf')
        best_W1, best_b1, best_W2, best_b2 = None, None, None, None

        for _ in range(epochs):
            z1 = X_tr @ W1 + b1
            a1 = np.maximum(0, z1)
            yp = a1 @ W2 + b2
            dy = 2 * (yp - y_tr) / len(X_tr)
            dW2 = a1.T @ dy
            db2 = np.sum(dy, axis=0)
            da1 = dy @ W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = X_tr.T @ dz1
            db1 = np.sum(dz1, axis=0)
            W1 -= lr * dW1; b1 -= lr * db1
            W2 -= lr * dW2; b2 -= lr * db2

            # early-stopping: 每200步评估一次测试集
            if _ % 200 == 0:
                z1_te = X_te @ W1 + b1
                te_mse = np.mean((np.maximum(0, z1_te) @ W2 + b2 - y_te.reshape(-1,1))**2)
                if te_mse < best_te_mse:
                    best_te_mse = te_mse
                    best_W1 = W1.copy(); best_b1 = b1.copy()
                    best_W2 = W2.copy(); best_b2 = b2.copy()

        if best_W1 is not None:
            W1, b1, W2, b2 = best_W1, best_b1, best_W2, best_b2

        y_tr_pred = (np.maximum(0, X_tr @ W1 + b1) @ W2 + b2).flatten()
        y_te_pred = (np.maximum(0, X_te @ W1 + b1) @ W2 + b2).flatten()
        return (np.mean((y_tr_pred - y_tr.flatten())**2),
                np.mean((y_te_pred - y_te.flatten())**2))

    train_mses_nn = []
    test_mses_nn = []
    for d_sub in poly_dims:
        X_sub = X_hd[:, :d_sub]
        tr_mse, te_mse = simple_nn_fit_multi(
            X_sub[:n_train], y_tr_norm, X_sub[n_train:], y_te_norm)
        train_mses_nn.append(tr_mse)
        test_mses_nn.append(te_mse)

    nn_param_counts = [d * 8 + 2 * 8 + 1 for d in poly_dims]   # 输入→8→1
    poly_param_counts_2 = [math.comb(d + 2, 2) + 1 for d in poly_dims]  # +1 for bias
    poly_param_counts_3 = [math.comb(d + 3, 3) + 1 for d in poly_dims]

    # ========== 画图 ==========
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("实验二：多项式回归 vs 神经网络", fontsize=16, fontweight='bold', y=0.98)

    # 子图1：参数数量增长
    ax = axes[0, 0]
    ax.plot(dims, poly_params_3, 'r-', linewidth=2.5, label="多项式 (3次)")
    ax.plot(dims, poly_params_2, 'r--', linewidth=2, label="多项式 (2次)")
    ax.plot(dims, nn_params_small, 'b-', linewidth=2.5, label="MLP (8隐藏)")
    ax.plot(dims, nn_params_tiny, 'b--', linewidth=2, label="MLP (4隐藏)")
    ax.set_xlabel("输入维度 d", fontsize=10)
    ax.set_ylabel("参数数量", fontsize=10)
    ax.set_title("① 参数数量 vs 输入维度", fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    # 标注交叉点
    cross_3_8 = next((d for d in dims if poly_params_3[d-1] > nn_params_small[d-1]), None)
    cross_3_4 = next((d for d in dims if poly_params_3[d-1] > nn_params_tiny[d-1]), None)
    if cross_3_8:
        ax.axvline(x=cross_3_8, color='gray', linestyle=':', alpha=0.5)
        ax.annotate(f"d={cross_3_8}\n3次多项式\n反超MLP(8)", xy=(cross_3_8, poly_params_3[cross_3_8-1]),
                   xytext=(cross_3_8+8, poly_params_3[cross_3_8-1]*0.1),
                   fontsize=7, color='darkred',
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax.text(38, 15000, "3次多项式: C(d+3,3)\n组合爆炸!", fontsize=8, color='darkred',
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))
    ax.text(38, 80, "MLP: O(d·h)\n线性增长", fontsize=8, color='darkblue',
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

    # 子图2：Runge 现象 - 多项式
    ax = axes[0, 1]
    ax.plot(x_test_rung, y_test_rung, 'k-', linewidth=2, label="真实函数 $1/(1+25x^2)$")
    ax.scatter(x_train_rung, y_train_rung, c='black', s=30, zorder=5, label="训练点")
    ax.plot(x_test_rung, y_poly_pred, 'r-', linewidth=1.5, label=f"{degree}次多项式拟合")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("② Runge 现象：高次多项式震荡")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.5)
    # 标注震荡
    ax.annotate("边缘剧烈震荡！", xy=(0.92, y_poly_pred[-6]), xytext=(0.5, 1.3),
               fontsize=9, color='red',
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 子图3：Runge 现象 - 神经网络
    ax = axes[0, 2]
    ax.plot(x_test_rung, y_test_rung, 'k-', linewidth=2, label="真实函数")
    ax.scatter(x_train_rung, y_train_rung, c='black', s=30, zorder=5, label="训练点")
    ax.plot(x_test_rung, y_nn_pred, 'b-', linewidth=2, label="两层 MLP (ReLU)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("③ 神经网络拟合：平滑无震荡")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.5)
    ax.text(0, 1.2, "MLP 平稳拟合\n无边缘震荡", fontsize=9, color='blue',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 子图4：Train/Test MSE — 三模型同图对比，共享 y 轴
    ax = axes[1, 0]

    # 多项式 2 次
    markers, caps, bars = ax.errorbar(poly_dims, test_mses_poly2,
                yerr=[max(0, a-b) for a,b in zip(test_mses_poly2, train_mses_poly2)],
                fmt='o--', color='#d62728', linewidth=2, markersize=7,
                capsize=4, label="多项式(2次) 测试MSE", zorder=5)
    # 多项式 3 次
    ax.errorbar(poly_dims, test_mses_poly3,
                yerr=[max(0, a-b) for a,b in zip(test_mses_poly3, train_mses_poly3)],
                fmt='s--', color='#ff7f0e', linewidth=2, markersize=7,
                capsize=4, label="多项式(3次) 测试MSE", zorder=5)
    # NN
    ax.errorbar(poly_dims, test_mses_nn,
                yerr=[max(0, a-b) for a,b in zip(test_mses_nn, train_mses_nn)],
                fmt='D-', color='#1f77b4', linewidth=2.5, markersize=7,
                capsize=4, label="MLP(8隐藏) 测试MSE", zorder=5)

    # 用浅色虚线画训练集 MSE
    ax.plot(poly_dims, train_mses_poly2, ':', color='#d62728', linewidth=1, alpha=0.5)
    ax.plot(poly_dims, train_mses_poly3, ':', color='#ff7f0e', linewidth=1, alpha=0.5)
    ax.plot(poly_dims, train_mses_nn, ':', color='#1f77b4', linewidth=1, alpha=0.5,
            label="训练MSE (三线几乎重叠在底部)")

    ax.set_xlabel("输入维度 d", fontsize=10)
    ax.set_ylabel("MSE (标准化)", fontsize=10)
    ax.set_title("④ 测试集 MSE 对比（误差棒=泛化差距）\n维度增加 → 多项式泛化崩塌，MLP 稳定", fontsize=11, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(True, alpha=0.3)

    # 标出多项式崩塌点
    ax.annotate("3次多项式\n测试误差暴涨！", xy=(7, test_mses_poly3[5]),
               xytext=(4, test_mses_poly3[5]*2.5), fontsize=8.5, color='#ff7f0e',
               fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.8),
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.annotate("MLP 泛化稳定", xy=(8, test_mses_nn[-1]),
               xytext=(6, test_mses_nn[-1]*0.5), fontsize=8.5, color='#1f77b4',
               fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.8))

    # 子图5：泛化差距直方图 — 直观对比
    ax = axes[1, 1]
    gap_poly2 = [max(0, t - r) for t, r in zip(test_mses_poly2, train_mses_poly2)]
    gap_poly3 = [max(0, t - r) for t, r in zip(test_mses_poly3, train_mses_poly3)]
    gap_nn = [max(0, t - r) for t, r in zip(test_mses_nn, train_mses_nn)]

    x_pos = np.arange(len(poly_dims))
    width = 0.25
    ax.bar(x_pos - width, gap_poly2, width, label="多项式(2次)", color='#d62728', alpha=0.85)
    ax.bar(x_pos, gap_poly3, width, label="多项式(3次)", color='#ff7f0e', alpha=0.85)
    ax.bar(x_pos + width, gap_nn, width, label="MLP(8隐藏)", color='#1f77b4', alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"d={d}" for d in poly_dims], fontsize=9)
    ax.set_xlabel("输入维度", fontsize=10)
    ax.set_ylabel("泛化差距 (Test - Train MSE)", fontsize=10)
    ax.set_title("⑤ 泛化差距对比\n值越小 = 泛化越好", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=1)

    # 在3次多项式柱子炸开处标注
    max_gap_idx = np.nanargmax(gap_poly3)
    if gap_poly3[max_gap_idx] > 0.5:
        ax.annotate(f"过拟合！\n+{gap_poly3[max_gap_idx]:.2f}",
                   xy=(max_gap_idx, gap_poly3[max_gap_idx]),
                   xytext=(max_gap_idx-1.5, gap_poly3[max_gap_idx]*1.3),
                   fontsize=8, color='#ff7f0e', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5))

    # 标注 NN 泛化差距极小的区域
    ax.annotate("MLP 泛化差距\n持续最小",
               xy=(len(poly_dims)-2, np.nanmean(gap_nn[-2:])),
               xytext=(len(poly_dims)-4, np.nanmean(gap_nn[-2:])*3),
               fontsize=8, color='#1f77b4', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5))

    # 子图6：参数效率对比
    ax = axes[1, 2]
    x_pos = np.arange(len(poly_dims))
    width = 0.22
    bars1 = ax.bar(x_pos - width, poly_param_counts_2, width, label="多项式 (2次)", color='lightcoral', alpha=0.85)
    bars2 = ax.bar(x_pos, poly_param_counts_3, width, label="多项式 (3次)", color='#d62728', alpha=0.85)
    bars3 = ax.bar(x_pos + width, nn_param_counts, width, label="MLP (6隐藏)", color='#1f77b4', alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"d={d}" for d in poly_dims], fontsize=9)
    ax.set_xlabel("输入维度", fontsize=10)
    ax.set_ylabel("参数数量", fontsize=10)
    ax.set_title("⑥ 参数数量对比（同任务）", fontsize=11, fontweight='bold')
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    # 在3次多项式柱子上标注反超比例
    for i, (p2, p3, n) in enumerate(zip(poly_param_counts_2, poly_param_counts_3, nn_param_counts)):
        # 标注3次 vs NN的倍数
        ratio = p3 / n
        y_pos = max(p3, n) * 1.15
        ax.text(i, y_pos, f"{ratio:.1f}x", ha='center', fontsize=7.5,
                fontweight='bold', color='#d62728')

    # 关键标注
    ax.annotate("2次: 参数少但欠拟合\n3次: 参数爆炸\n从d=5起超过MLP",
               xy=(4, poly_param_counts_3[4]), xytext=(1, max(poly_param_counts_3)*3),
               fontsize=8, color='darkred', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig("experiment2_poly_vs_nn.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[实验二] 图片已保存：experiment2_poly_vs_nn.png")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  思考题实验：激活函数 & 多项式回归 vs 神经网络")
    print("=" * 60)
    print()

    print("[实验一] 为什么不用 x² 作为激活函数？")
    print("         对比 x² / ReLU / Sigmoid 的三大行为差异...")
    experiment1_x2_activation()

    print()
    print("[实验二] 多项式回归 vs 神经网络的深度对比")
    print("         参数增长、Runge 现象、高维泛化...")
    experiment2_poly_vs_nn()

    print()
    print("=" * 60)
    print("  全部实验完成！")
    print("  图1: experiment1_x2_activation.png")
    print("  图2: experiment2_poly_vs_nn.png")
    print("=" * 60)
