"""
第三节课训练模块：CLIP 对比学习训练（轻量投影头 + InfoNCE Loss）
==============================================================

【设计思路】
  完整训练 CLIP 需要 4 亿图文对 + 数百 GPU 天，不现实。
  本模块采用「冻结编码器 + 可训练投影头」的方案：
    - CLIP 图像/文本编码器冻结（保留预训练知识）
    - 各自接一个 2 层 MLP 投影头（总参数量 ≈ 100 万）
    - 用对称 InfoNCE 损失训练投影头，拉近匹配对、推远非匹配对
  这样在有限算力下完整演示了对比学习的训练流程。

【讲义对照】见 ../讲义.md「二、对比学习（对称 InfoNCE）」

【运行】
  python clip_train.py                    # 默认配置：500 对，5 epoch
  python clip_train.py --epochs 10 --lr 1e-4 --max-samples 2000

【硬件需求】
  - GPU (推荐): 2GB+ 显存，训练约 5-15 分钟
  - CPU: 可运行，训练约 30-60 分钟（500 对）

【输出文件】
  - outputs/train_loss_curve.png         训练损失曲线
  - outputs/sim_matrix_epoch*.png        每个 epoch 的相似度矩阵
  - cache/clip_projection_head.pt        训练好的投影头权重
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 路径 ----------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
IMAGE_DIR = DATA_DIR / "Flicker8k_Dataset"
TEXT_DIR = DATA_DIR / "Flickr8k_text"
CACHE_DIR = PROJECT_DIR / "cache"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 全局配置 ----------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 512          # CLIP ViT-B/32 输出维度
PROJ_DIM = 256           # 投影头中间层维度


# =============================================================================
# 第一部分：数据集准备（复用 clip_retrieval.py 的数据加载）
# =============================================================================

def load_train_pairs(max_samples: int = 500) -> list[tuple[str, str]]:
    """
    从 Flickr8k 训练集构建 (图片路径, 描述文本) 对。

    每张图片有 5 条描述，每条描述与图片组成一个训练对。
    若 max_samples 小于总数，则随机采样子集。

    返回 [(img_path, caption), ...]
    """
    import urllib.request
    import zipfile
    import random

    # ---------- 确保数据集已下载 ----------
    if not IMAGE_DIR.exists() or not TEXT_DIR.exists():
        print("数据集未找到，正在下载 Flickr8k (约 1.1 GB，仅首次)...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if not (DATA_DIR / "Flickr8k_Dataset.zip").exists():
            print("  下载图片包...")
            urllib.request.urlretrieve(
                "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
                str(DATA_DIR / "Flickr8k_Dataset.zip"),
            )
        if not IMAGE_DIR.exists():
            with zipfile.ZipFile(DATA_DIR / "Flickr8k_Dataset.zip") as z:
                z.extractall(DATA_DIR)

        if not (DATA_DIR / "Flickr8k_text.zip").exists():
            print("  下载文本包...")
            urllib.request.urlretrieve(
                "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
                str(DATA_DIR / "Flickr8k_text.zip"),
            )
        if not TEXT_DIR.exists():
            with zipfile.ZipFile(DATA_DIR / "Flickr8k_text.zip") as z:
                z.extractall(DATA_DIR)
        print("  数据集准备完成。")

    # ---------- 读训练集 ID 列表 ----------
    train_file = TEXT_DIR / "Flickr8k.trainImages.txt"
    train_ids = set(
        line.strip() for line in train_file.read_text().splitlines() if line.strip()
    )

    # ---------- 读所有标题 ----------
    token_file = TEXT_DIR / "Flickr8k.token.txt"
    captions: dict[str, list[str]] = {}
    for line in token_file.read_text(encoding="utf-8").strip().split("\n"):
        img_cap, text = line.split("\t")
        img_id = img_cap.split("#")[0]
        captions.setdefault(img_id, []).append(text.strip())

    # ---------- 构建训练对 ----------
    pairs: list[tuple[str, str]] = []
    for img_id in sorted(train_ids & captions.keys()):
        img_path = str(IMAGE_DIR / img_id)
        if not Path(img_path).exists():
            continue
        for cap in captions[img_id]:
            pairs.append((img_path, cap))

    # 随机采样子集
    if max_samples > 0 and max_samples < len(pairs):
        random.seed(42)
        pairs = random.sample(pairs, max_samples)

    print(f"训练集: {len(pairs)} 个图文对 (每张图片最多 5 条描述)")
    return pairs


class FlickrPairDataset(Dataset):
    """
    PyTorch Dataset：返回 (pixel_values, input_ids, attention_mask)。

    【讲义·三】CLIPProcessor 统一处理图像和文本：
      - 图像 → resize(224) → center_crop(224) → normalize
      - 文本 → tokenize → [1, 77] token ids
    """

    def __init__(self, pairs: list[tuple[str, str]], processor: CLIPProcessor):
        self.pairs = pairs
        self.processor = processor

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, caption = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        # 用 processor 分别处理，返回统一的 tensor
        img_inputs = self.processor(images=image, return_tensors="pt")
        txt_inputs = self.processor(
            text=caption, return_tensors="pt", padding="max_length",
            truncation=True, max_length=77,
        )
        return {
            "pixel_values": img_inputs["pixel_values"].squeeze(0),   # [3, 224, 224]
            "input_ids": txt_inputs["input_ids"].squeeze(0),          # [77]
            "attention_mask": txt_inputs["attention_mask"].squeeze(0),# [77]
        }


# =============================================================================
# 第二部分：可训练投影头
# 【讲义·二】CLIP 训练时，图像和文本编码器后各接一个线性投影层
# 本模块将编码器冻结，只训练这两个投影 MLP
# =============================================================================

class ProjectionHead(nn.Module):
    """
    2 层 MLP 投影头：512 → 256 → 512

    【讲义·二】
      CLIP 原文用的是单层线性投影 W_i ∈ R^{d_e × d_i}。
      这里用 2 层 MLP 增加一点表达能力，同时保持轻量。

    参数量: 512×256 + 256 + 256×512 + 512 ≈ 264K
    """

    def __init__(self, input_dim: int = EMBED_DIM, hidden_dim: int = PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# 第三部分：InfoNCE 损失
# 【讲义·二·公式】
#   L = 1/(2N) * Σ_i [ -log(exp(s_ii/τ) / Σ_j exp(s_ij/τ))     ← 图像→文本
#                      -log(exp(s_ii/τ) / Σ_j exp(s_ji/τ)) ]   ← 文本→图像
#   其中 s_ij = cosine_sim(img_i, txt_j)，τ 为温度系数
# =============================================================================

def contrastive_loss(
    img_emb: torch.Tensor,    # [B, D]  图像投影向量
    txt_emb: torch.Tensor,    # [B, D]  文本投影向量
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对称 InfoNCE 对比损失。

    假设 batch 内第 i 张图与第 i 条文本是匹配对（正样本），
    其余 (B-1) 个跨对为负样本。

    返回 (loss, accuracy)
      - loss: 标量
      - acc: 图像→文本匹配正确率（对角线 argmax 占比）
    """
    # L2 归一化 → cosine 空间
    img_emb = F.normalize(img_emb, dim=-1)  # [B, D]
    txt_emb = F.normalize(txt_emb, dim=-1)  # [B, D]

    # 相似度矩阵 [B, B]
    logits = img_emb @ txt_emb.T / temperature  # 除温度缩放

    # 标签：对角线为正样本
    labels = torch.arange(logits.shape[0], device=logits.device)

    # 图像 → 文本 方向
    loss_i2t = F.cross_entropy(logits, labels)
    # 文本 → 图像 方向（转置）
    loss_t2i = F.cross_entropy(logits.T, labels)

    loss = (loss_i2t + loss_t2i) / 2.0

    # 准确率（匹配正确即 argmax 在对角线）
    acc_i2t = (logits.argmax(dim=1) == labels).float().mean()
    acc_t2i = (logits.T.argmax(dim=1) == labels).float().mean()
    acc = (acc_i2t + acc_t2i) / 2.0

    return loss, acc


# =============================================================================
# 第四部分：训练循环
# =============================================================================

def train_one_epoch(
    model: CLIPModel,
    img_proj: ProjectionHead,
    txt_proj: ProjectionHead,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    temperature: float,
    epoch: int,
) -> tuple[float, float]:
    """
    一个 epoch 的训练。

    【流程】
      1. 冻结的 CLIP 编码器前向 → image_emb [B, 512], text_emb [B, 512]
      2. 可训练投影头 → img_proj [B, 512], txt_proj [B, 512]
      3. 对称 InfoNCE 损失
      4. 反向传播 → 只更新投影头
    """
    img_proj.train()
    txt_proj.train()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=90)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        # ---- 1. 冻结编码器前向（不计算梯度，节省显存/时间） ----
        with torch.no_grad():
            img_emb = model.get_image_features(pixel_values)               # [B, 512]
            txt_emb = model.get_text_features(input_ids, attention_mask)   # [B, 512]

        # ---- 2. 可训练投影头 ----
        img_proj_out = img_proj(img_emb)  # [B, 512]
        txt_proj_out = txt_proj(txt_emb)  # [B, 512]

        # ---- 3. 对比损失 ----
        loss, acc = contrastive_loss(img_proj_out, txt_proj_out, temperature)

        # ---- 4. 反向传播 ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.3f}")

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate_retrieval(
    model: CLIPModel,
    img_proj: ProjectionHead,
    txt_proj: ProjectionHead,
    dataloader: DataLoader,
    top_k: int = 5,
) -> float:
    """
    评估：在验证集上计算 Recall@K（Top-K 检索命中率）。

    对每张图，用投影后的向量检索 batch 内文本，
    若正确文本在 Top-K 内则命中。
    这模拟了对比学习的训练目标在检索场景下的效果。
    """
    img_proj.eval()
    txt_proj.eval()

    hits = 0
    total = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        img_emb = model.get_image_features(pixel_values)
        txt_emb = model.get_text_features(input_ids, attention_mask)

        img_out = F.normalize(img_proj(img_emb), dim=-1)
        txt_out = F.normalize(txt_proj(txt_emb), dim=-1)

        sim = img_out @ txt_out.T  # [B, B]
        _, top_idx = sim.topk(top_k, dim=1)          # [B, K]
        labels = torch.arange(sim.shape[0], device=sim.device).unsqueeze(1)  # [B, 1]
        hits += (top_idx == labels).any(dim=1).sum().item()
        total += sim.shape[0]

    return hits / total if total > 0 else 0.0


# =============================================================================
# 第五部分：可视化
# =============================================================================

def plot_training_curves(
    train_losses: list[float],
    train_accs: list[float],
    val_recalls: list[float],
    save_path: Path,
) -> None:
    """绘制训练损失 / 准确率 / 验证召回率曲线。"""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 左：损失
    ax1.plot(epochs, train_losses, "o-", color="#3498db", linewidth=2, markersize=6,
             label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("InfoNCE Loss")
    ax1.set_title("训练损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右：准确率 + 召回率
    ax2.plot(epochs, train_accs, "s-", color="#2ecc71", linewidth=2, markersize=6,
             label="Train Accuracy (i2t + t2i)")
    if val_recalls:
        ax2.plot(epochs, val_recalls, "D-", color="#e74c3c", linewidth=2, markersize=6,
                 label="Val Recall@5")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy / Recall")
    ax2.set_title("训练准确率 & 检索召回率")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("CLIP 对比学习训练 — InfoNCE Loss", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n训练曲线已保存: {save_path.relative_to(PROJECT_DIR)}")


def plot_similarity_matrix(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    epoch: int,
    save_dir: Path,
) -> None:
    """
    可视化相似度矩阵：对角线应为最高值（匹配对），非对角线较低。

    【讲义·二】对称 InfoNCE 让模型学会区分正负样本，
    训练前对角线不明显，训练后对角线应显著亮于周围。
    """
    img_emb = F.normalize(img_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)
    sim = (img_emb @ txt_emb.T).cpu().numpy()

    n = min(sim.shape[0], 30)  # 最多显示 30×30
    sim = sim[:n, :n]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Text idx")
    ax.set_ylabel("Image idx")
    ax.set_title(f"Similarity Matrix (Epoch {epoch})")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 标注对角线均值 vs 非对角线均值
    diag_mean = np.diag(sim).mean()
    off_diag = sim[~np.eye(sim.shape[0], dtype=bool)]
    off_diag_mean = off_diag.mean()
    ax.text(0.5, -0.12, f"对角线均值: {diag_mean:.3f}  |  非对角线均值: {off_diag_mean:.3f}",
            transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    save_path = save_dir / f"sim_matrix_epoch{epoch:02d}.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 第六部分：主流程
# =============================================================================

def main(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("第三节课训练模块：CLIP 对比学习（投影头 + InfoNCE）")
    print("=" * 60)
    print(f"设备: {DEVICE}  |  训练对: {args.max_samples}  |  Epoch: {args.epochs}")
    print(f"学习率: {args.lr}  |  温度: {args.temperature}  |  批次: {args.batch_size}")

    # ---- 1. 准备数据 ----
    print("\n[1/6] 加载训练数据...")
    train_pairs = load_train_pairs(max_samples=args.max_samples)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    train_dataset = FlickrPairDataset(train_pairs, processor)

    # 划分 80/20 训练/验证
    n_val = max(len(train_dataset) // 5, args.batch_size)
    n_train = len(train_dataset) - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, drop_last=False,
    )
    print(f"  训练: {len(train_subset)} 对  |  验证: {len(val_subset)} 对")

    # ---- 2. 加载冻结 CLIP ----
    print("\n[2/6] 加载预训练 CLIP (冻结编码器)...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"  CLIP 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M (已冻结)")

    # ---- 3. 初始化可训练投影头 ----
    print("\n[3/6] 初始化投影头...")
    img_proj = ProjectionHead().to(DEVICE)
    txt_proj = ProjectionHead().to(DEVICE)
    trainable_params = sum(p.numel() for p in img_proj.parameters()) + \
                       sum(p.numel() for p in txt_proj.parameters())
    print(f"  投影头参数量: {trainable_params / 1e3:.0f}K (可训练)")
    print(f"  训练/总参数比: {trainable_params / sum(p.numel() for p in model.parameters()) * 100:.2f}%")

    # ---- 4. 优化器 ----
    optimizer = torch.optim.AdamW(
        list(img_proj.parameters()) + list(txt_proj.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )

    # ---- 5. 训练循环 ----
    print(f"\n[4/6] 开始训练 ({args.epochs} epochs)...")
    print("-" * 40)
    t0 = time.time()

    train_losses: list[float] = []
    train_accs: list[float] = []
    val_recalls: list[float] = []

    # 训练前先保存初始相似度矩阵
    sample_batch = next(iter(train_loader))
    pixel_vals = sample_batch["pixel_values"].to(DEVICE)
    input_ids_s = sample_batch["input_ids"].to(DEVICE)
    attn_mask = sample_batch["attention_mask"].to(DEVICE)
    with torch.no_grad():
        init_img = model.get_image_features(pixel_vals)
        init_txt = model.get_text_features(input_ids_s, attn_mask)
    plot_similarity_matrix(init_img, init_txt, 0, OUTPUT_DIR)

    for epoch in range(1, args.epochs + 1):
        avg_loss, avg_acc = train_one_epoch(
            model, img_proj, txt_proj, train_loader, optimizer,
            args.temperature, epoch,
        )
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)

        # 验证
        recall = evaluate_retrieval(model, img_proj, txt_proj, val_loader, top_k=args.top_k)
        val_recalls.append(recall)

        print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.3f} | "
              f"Val Recall@{args.top_k}: {recall:.3f}")

        # 每个 epoch 保存相似度矩阵
        with torch.no_grad():
            epoch_img = img_proj(model.get_image_features(pixel_vals))
            epoch_txt = txt_proj(model.get_text_features(input_ids_s, attn_mask))
        plot_similarity_matrix(epoch_img, epoch_txt, epoch, OUTPUT_DIR)

    elapsed = time.time() - t0
    print("-" * 40)
    print(f"训练完成！总耗时: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ---- 6. 保存与可视化 ----
    print("\n[5/6] 保存模型 & 绘制曲线...")

    # 保存投影头权重
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "img_proj": img_proj.state_dict(),
        "txt_proj": txt_proj.state_dict(),
        "config": {"embed_dim": EMBED_DIM, "proj_dim": PROJ_DIM},
    }
    ckpt_path = CACHE_DIR / "clip_projection_head.pt"
    torch.save(ckpt, ckpt_path)
    print(f"  投影头权重已保存: {ckpt_path.relative_to(PROJECT_DIR)}")

    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, val_recalls,
                         OUTPUT_DIR / "train_loss_curve.png")

    # ---- 总结 ----
    print("\n[6/6] 训练结果总结")
    print("=" * 40)
    print(f"  初始 Loss:         {train_losses[0]:.4f}")
    print(f"  最终 Loss:         {train_losses[-1]:.4f}")
    print(f"  初始 Acc:          {train_accs[0]:.3f}")
    print(f"  最终 Acc:          {train_accs[-1]:.3f}")
    print(f"  初始 Val Recall@{args.top_k}: {val_recalls[0]:.3f}")
    print(f"  最终 Val Recall@{args.top_k}: {val_recalls[-1]:.3f}")

    print(f"\n输出文件:")
    print(f"  - {OUTPUT_DIR / 'train_loss_curve.png'}")
    print(f"  - {OUTPUT_DIR / 'sim_matrix_epoch*.png'} (共 {args.epochs + 1} 张)")
    print(f"  - {ckpt_path}")

    print(f"\n→ 请对照 ../讲义.md「二、对比学习」理解 InfoNCE 损失的含义。")
    print(f"→ 观察相似度矩阵变化：训练前对角线不明显 → 训练后对角线应显著亮于其他区域。")


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP 对比学习训练 (轻量投影头)")

    parser.add_argument("--epochs", type=int, default=5,
                        help="训练轮数 (默认 5)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="批次大小 (默认 16, CPU 建议 8)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="学习率 (默认 3e-4)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE 温度系数 (默认 0.07, 同 CLIP 论文)")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="最大训练样本数 (默认 500, 设 0 使用全部 ~30000 对)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="验证时 Recall@K 的 K 值 (默认 5)")

    args = parser.parse_args()

    if args.max_samples == 0:
        args.max_samples = 999999  # 使用全部

    if DEVICE == "cpu":
        print("⚠️  未检测到 GPU，将使用 CPU 训练（可能较慢）。")
        if args.batch_size > 8:
            print(f"   建议将 batch_size 从 {args.batch_size} 降至 8 或更小。")

    main(args)
