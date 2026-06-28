"""生成固定 CSV 数据集（graduate_admission.csv）。

造数规则（教学用合成数据）：
  - 回归标签「综合评分」：五维特征的线性加权 + 高斯噪声，裁剪到 0~100
  - 分类标签「保研资格」：含非线性惩罚项的 latent 分数，按分位数切分约 35% 正样本
    （避免原先 logit 全为正导致 98% 正样本、指标虚高的问题）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FEATURE_NAMES = ["绩点GPA", "科研经历", "竞赛获奖", "英语成绩", "综合排名分位"]
LABEL_REG = "综合评分"
LABEL_CLS = "保研资格"
NUM_FEATURES = len(FEATURE_NAMES)
CSV_PATH = Path(__file__).resolve().parent / "graduate_admission.csv"

# 分类正样本目标比例（保研名额有限，刻意设为偏少）
POSITIVE_RATE = 0.35
N_SAMPLES = 800
RANDOM_SEED = 42


def generate_dataset(
    n_samples: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
    positive_rate: float = POSITIVE_RATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n_samples, NUM_FEATURES))

    # 回归：综合评分（0~100），与保研相关但允许不完全一致
    score = (
        35 * x[:, 0]
        + 25 * x[:, 1]
        + 20 * x[:, 2]
        + 10 * x[:, 3]
        + 10 * (1 - x[:, 4])
        + rng.normal(0, 5, n_samples)
    )
    y_reg = np.clip(score, 0, 100).astype(np.float32)

    # 分类：latent 分数 + 非线性边界（高绩点但科研弱会被扣分）
    latent = (
        2.0 * x[:, 0]
        + 1.8 * x[:, 1]
        + 1.4 * x[:, 2]
        + 0.7 * x[:, 3]
        - 2.2 * x[:, 4]
        - 1.6 * (x[:, 0] * (1 - x[:, 1]))
        + rng.normal(0, 0.45, n_samples)
    )
    cutoff = np.quantile(latent, 1 - positive_rate)
    y_cls = (latent >= cutoff).astype(np.int64)

    return x, y_reg, y_cls


def write_csv(
    path: Path = CSV_PATH,
    n_samples: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
    positive_rate: float = POSITIVE_RATE,
) -> tuple[int, float]:
    x, y_reg, y_cls = generate_dataset(n_samples, seed, positive_rate)
    header = FEATURE_NAMES + [LABEL_REG, LABEL_CLS]
    rows = np.column_stack([x, y_reg.reshape(-1, 1), y_cls.reshape(-1, 1)])

    lines = [",".join(header)]
    for row in rows:
        feat = ",".join(f"{v:.6f}" for v in row[:NUM_FEATURES])
        lines.append(f"{feat},{row[NUM_FEATURES]:.4f},{int(row[NUM_FEATURES + 1])}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return len(rows), float(y_cls.mean())


def main() -> None:
    n, pos_rate = write_csv()
    print(f"已写入 {CSV_PATH.name}，共 {n} 条样本，正样本比例 {pos_rate:.2%}")


if __name__ == "__main__":
    main()
