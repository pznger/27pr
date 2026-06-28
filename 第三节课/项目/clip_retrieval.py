"""
第三节课综合实战：基于 CLIP 的零样本图文检索
=========================================

【任务】在 Flickr8k 数据集上用预训练 CLIP 实现：
  - 图像检索：输入一段文本 → 返回最匹配的 Top-K 张图
  - 文本检索：输入一张图片 → 返回最相关的 Top-K 条描述

【核心原理】
  CLIP 将图像和文本分别编码到共享向量空间，
  同一语义的图文对在该空间中余弦相似度最高。
  只需计算相似度矩阵，无需任何微调 → 零样本检索。

【讲义对照】见 ../讲义.md「二、对比学习 / 三、图像编码器 / 四、零样本迁移」

【运行】python clip_retrieval.py
  首次运行会：
    1. 自动下载 Flickr8k 数据集（约 1.1 GB）
    2. 自动下载 CLIP ViT-B/32 模型（约 600 MB，仅一次）
    3. 编码全量图片（此后缓存到 .npy 文件，无需重复编码）

【调参/实验】
  修改 main() 底部的 QUERY_TEXT 变量即可检索不同内容
  修改 TOP_K 调整返回数量
  无需 GPU，CPU 约需 3-5 分钟编码全部图片
"""

from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import torch
import urllib.request
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------- 路径 -------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"               # 存放 Flickr8k
IMAGE_DIR = DATA_DIR / "Flicker8k_Dataset"    # 解压后的图片目录
TEXT_DIR = DATA_DIR / "Flickr8k_text"         # 解压后的文本文件
CACHE_DIR = PROJECT_DIR / "cache"             # 缓存的图像嵌入
OUTPUT_DIR = PROJECT_DIR / "outputs"

# ---------------------------- 全局配置 ----------------------------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # 轻量 CLIP，适合笔记本
TOP_K = 5                                          # 每次检索返回几条结果
BATCH_SIZE = 32                                    # 图像编码批大小（CPU 友好）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLOAT_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32  # CPU 用 float32


# =============================================================================
# 第一部分：数据集下载与加载
# 【讲义·二】对比学习需要成对的 (image, text) 数据
#
# Flickr8k：8,000 张自然图片，每张 5 条人工标注描述
# 本项目用它作为检索「素材库」，CLIP 模型不需要在此数据上训练
# =============================================================================

F8K_IMAGE_URL = (
    "https://github.com/jbrownlee/Datasets/releases/download/"
    "Flickr8k/Flickr8k_Dataset.zip"
)
F8K_TEXT_URL = (
    "https://github.com/jbrownlee/Datasets/releases/download/"
    "Flickr8k/Flickr8k_text.zip"
)


def _download_file(url: str, dest: Path) -> None:
    """显示进度条的下载函数。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  下载 {dest.name} ...")

    def _report(count: int, block_size: int, total_size: int) -> None:
        pct = count * block_size * 100 / total_size if total_size > 0 else 0
        sys.stdout.write(f"\r  {pct:.0f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest), reporthook=_report)
    print("")


def _ensure_dataset() -> None:
    """若本地缺少 Flickr8k，自动下载并解压。"""
    if IMAGE_DIR.exists() and TEXT_DIR.exists():
        print(f"  Flickr8k 数据已存在: {DATA_DIR}")
        return

    print("正在下载 Flickr8k 数据集（约 1.1 GB，仅首次需要）...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    img_zip = DATA_DIR / "Flickr8k_Dataset.zip"
    txt_zip = DATA_DIR / "Flickr8k_text.zip"

    if not IMAGE_DIR.exists():
        if not img_zip.exists():
            _download_file(F8K_IMAGE_URL, img_zip)
        print("  解压图片...")
        with zipfile.ZipFile(img_zip) as z:
            z.extractall(DATA_DIR)

    if not TEXT_DIR.exists():
        if not txt_zip.exists():
            _download_file(F8K_TEXT_URL, txt_zip)
        print("  解压文本...")
        with zipfile.ZipFile(txt_zip) as z:
            z.extractall(DATA_DIR)

    print("  数据集准备完成。")


def load_captions(split: str = "all") -> dict[str, list[str]]:
    """
    加载图片 → 描述列表的映射。

    Flickr8k 目录结构：
      Flicker8k_Dataset/    ← 图片文件夹
      Flickr8k_text/
        ├── Flickr8k.token.txt             ← 所有图片 id → 描述
        ├── Flickr8k.trainImages.txt   ← 训练集 id 列表
        ├── Flickr8k.devImages.txt     ← 验证集 id 列表
        └── Flickr8k.testImages.txt    ← 测试集 id 列表

    返回 { "1234567890.jpg": ["Two dogs in the snow.", ...] }
    """
    token_file = TEXT_DIR / "Flickr8k.token.txt"
    if not token_file.exists():
        raise FileNotFoundError(f"找不到 {token_file}，请确认数据集已下载")

    all_captions: dict[str, list[str]] = {}
    for line in token_file.read_text(encoding="utf-8").strip().split("\n"):
        # 格式："1000268201_693b08cb0e.jpg#0\tA child in a pink dress..."
        img_cap, text = line.split("\t")
        img_id = img_cap.split("#")[0]  # 去掉 #0～#4 后缀
        all_captions.setdefault(img_id, []).append(text.strip())

    if split == "all":
        return all_captions

    # 按官方划分过滤
    split_file = TEXT_DIR / f"Flickr8k.{'train' if split == 'train' else split}Images.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"找不到 {split_file}")
    ids = set(line.strip() for line in split_file.read_text().splitlines() if line.strip())
    return {img_id: caps for img_id, caps in all_captions.items() if img_id in ids}


# =============================================================================
# 第二部分：CLIP 模型加载与编码
# 【讲义·三】ViT 图像编码器 + Transformer 文本编码器
# 【讲义·四】零样本：无需在任何检索数据上微调
# =============================================================================

class CLIPRetriever:
    """
    封装 CLIP 模型，提供图像/文本编码与检索功能。

    【讲义·二/三】：
      - image_encoder (ViT-B/32): 将 224×224 图片映射到 512 维向量
      - text_encoder (Transformer): 将 token 序列映射到 512 维向量
      - 两个向量经 L2 归一化后做内积 = cosine similarity
    """

    def __init__(self, model_name: str = CLIP_MODEL_NAME) -> None:
        print(f"加载 CLIP 模型: {model_name} (首次运行将下载 ~600 MB)...")
        t0 = time.time()
        self.model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(torch.float32)  # CPU 推理用 float32 更稳定
        for param in self.model.parameters():
            param.requires_grad = False  # 冻结所有参数，仅做前向
        print(f"  模型加载完成，耗时 {time.time() - t0:.1f}s，设备: {DEVICE}")

    @torch.no_grad()
    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        """批量编码图片 → 512 维归一化向量 [N, 512]"""
        embeddings: list[np.ndarray] = []
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch = image_paths[i : i + BATCH_SIZE]
            images = [Image.open(p).convert("RGB") for p in batch]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            img_emb = self.model.get_image_features(**inputs)         # [B, 512]
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)    # L2 归一化
            embeddings.append(img_emb.cpu().numpy())
            sys.stdout.write(f"\r  编码图片: {min(i + BATCH_SIZE, len(image_paths))}/{len(image_paths)}")
            sys.stdout.flush()
        print("")
        return np.concatenate(embeddings, axis=0)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """批量编码文本 → 512 维归一化向量 [M, 512]"""
        embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            txt_emb = self.model.get_text_features(**inputs)          # [B, 512]
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)    # L2 归一化
            embeddings.append(txt_emb.cpu().numpy())
        return np.concatenate(embeddings, axis=0)


# =============================================================================
# 第三部分：检索逻辑
# 【讲义·二】cosine similarity 矩阵 → argsort 取 Top-K
# =============================================================================

def retrieve(
    query_emb: np.ndarray,           # [1, 512] 或 [N, 512]
    gallery_emb: np.ndarray,         # [M, 512]
    gallery_items: list[str],        # 候选池标识（路径或文本）
    top_k: int = TOP_K,
) -> list[tuple[str, float]]:
    """
    检索主函数：
      - sim = query_emb @ gallery_emb.T → 一个矩阵乘法完成全库比对
      - 按相似度降序取 top_k

    返回 [(item, score), ...]，score ∈ [0, 1]（cosine 范围约 0.2~0.5）
    """
    sim = query_emb @ gallery_emb.T  # [N, M]
    if sim.ndim == 1:
        sim = sim.reshape(1, -1)     # 单 query 统一成 [1, M]
    results = []
    for q_idx in range(sim.shape[0]):
        top_idx = np.argsort(sim[q_idx])[::-1][:top_k]
        for idx in top_idx:
            results.append((gallery_items[idx], float(sim[q_idx, idx])))
    return results


# =============================================================================
# 第四部分：可视化
# =============================================================================

def make_image_grid(image_paths: list[str], scores: list[float]) -> Path:
    """
    将检索到的 top_k 张图片拼接成一张大图，
    保存在 outputs/ 供查阅。
    """
    n = len(image_paths)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for i, (path, score) in enumerate(zip(image_paths, scores)):
        img = Image.open(path).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(f"Rank {i+1}: sim={score:.3f}", fontsize=10)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("CLIP 零样本图文检索结果", fontsize=14, y=1.02)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = OUTPUT_DIR / f"retrieval_results_{int(time.time())}.png"
    fig.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  检索结果图已保存: {filename.relative_to(PROJECT_DIR)}")
    return filename


# =============================================================================
# 第五部分：主流程
# =============================================================================

def main(query_text: str | None = None) -> None:
    """
    【讲义·十一】串联全流程：
      1. 下载/加载 Flickr8k 数据集
      2. 加载 CLIP ViT-B/32 预训练模型
      3. 编码全量图片 → 缓存到 cache/
      4. 编码 query 文本 → 检索 → 可视化 Top-K
    """
    print("=" * 60)
    print("第三节课 PyTorch 综合项目：CLIP 零样本图文检索")
    print("=" * 60)
    print(f"设备: {DEVICE} | CLIP 模型: {CLIP_MODEL_NAME}")
    print(f"数据集: Flickr8k | 返回 Top-{TOP_K}")

    # ------ 1. 准备数据 ------
    _ensure_dataset()
    captions = load_captions("all")
    image_ids = sorted(captions.keys())
    image_paths = [str(IMAGE_DIR / img_id) for img_id in image_ids]
    print(f"  已加载 {len(image_paths)} 张图片，每张 5 条描述")

    # ------ 2. 加载模型 ------
    retriever = CLIPRetriever()

    # ------ 3. 编码全量图片（含缓存） ------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "flickr8k_img_emb_vitb32.npy"

    if cache_file.exists():
        print("  加载缓存的图像嵌入...")
        gallery_emb = np.load(cache_file)
    else:
        print("  编码全量图片（约 3-5 分钟 / CPU）...")
        t0 = time.time()
        gallery_emb = retriever.encode_images(image_paths)
        np.save(cache_file, gallery_emb)
        print(f"  编码完成，耗时 {time.time() - t0:.1f}s；嵌入已缓存到 {cache_file.name}")

    # ------ 4. 文本检索 ------
    if query_text is None:
        query_text = "a dog running on a green grass field"
    print(f"\n检索 query: \"{query_text}\"")

    query_emb = retriever.encode_texts([query_text])
    results = retrieve(query_emb, gallery_emb, image_paths)

    # ------ 5. 展示结果 ------
    print(f"\n  Top-{TOP_K} 检索结果:")
    for rank, (path, score) in enumerate(results, 1):
        img_id = Path(path).name
        first_caption = captions.get(img_id, [""])[0]
        print(f"    {rank}. {img_id}  sim={score:.3f}")
        print(f"       描述: {first_caption}")

    # ------ 6. 可视化 ------
    result_paths = [p for p, _ in results]
    result_scores = [s for _, s in results]
    make_image_grid(result_paths, result_scores)

    # ------ 7. 双向检索实验 ------
    # 用第一张结果图做反问：找最匹配的描述
    print(f"\n{'='*40}")
    print("双向检索验证：用 Top-1 图片反问文本库")
    print(f"{'='*40}")
    top1_path = result_paths[0]
    top1_id = Path(top1_path).name
    img_emb = retriever.encode_images([top1_path])
    all_captions_flat = []
    caption_to_img = {}
    for img_id, caps in captions.items():
        for cap in caps:
            all_captions_flat.append(cap)
            caption_to_img[cap] = img_id
    text_emb = retriever.encode_texts(all_captions_flat)
    text_results = retrieve(img_emb, text_emb, all_captions_flat)

    print(f"  图 → 文本匹配结果:")
    for rank, (text, score) in enumerate(text_results[:3], 1):
        print(f"    {rank}. [{score:.3f}] {text}")

    print("\n完成！请对照 ../讲义.md「四、零样本迁移」理解检索机制。")


if __name__ == "__main__":
    # 修改 QUERY_TEXT 为任意自然语言描述即可检索不同内容
    # 示例见本文件底部注释
    QUERY_TEXT = "a boy in a red shirt playing soccer"

    # 也可以通过命令行传入：
    # python clip_retrieval.py "a sunset over the ocean"
    if len(sys.argv) > 1:
        QUERY_TEXT = " ".join(sys.argv[1:])

    main(QUERY_TEXT)
