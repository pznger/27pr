"""
第三节课 CLIP 图文检索 — Web 可视化界面
==========================================

使用 Streamlit 构建交互式前端，支持：
  - 模式一「以文搜图」：输入中文/英文描述 → 返回匹配的 Top-K 张图片
  - 模式二「以图搜文」：上传一张图片 → 返回最相关的 Top-K 条描述
  - 相似度直方图 + 图片网格展示

运行: streamlit run web_demo.py
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------- 环境配置 ----------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
IMAGE_DIR = DATA_DIR / "Flicker8k_Dataset"
CACHE_DIR = PROJECT_DIR / "cache"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLOAT_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ---------- 页面设置 ----------
st.set_page_config(
    page_title="CLIP 图文检索 DEMO",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 CLIP 零样本图文检索")
st.markdown("""
基于 OpenAI **CLIP (ViT-B/32)** 模型，无需任何微调即可实现图像与文本的跨模态检索。

- **以文搜图**：用自然语言描述你想找的画面
- **以图搜文**：上传一张图片，看 AI 如何用文字描述它
""")
st.markdown("---")

# ---------- 加载 CLIP 模型（缓存） ----------

@st.cache_resource
def load_clip_model(model_name: str):
    """加载 CLIP 模型（仅首次运行下载 ~600 MB），缓存到内存"""
    with st.spinner(f"正在加载 CLIP 模型 `{model_name}` (首次运行会下载约 600 MB)..."):
        model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
        processor = CLIPProcessor.from_pretrained(model_name)
        for param in model.parameters():
            param.requires_grad = False
    return model, processor

model, processor = load_clip_model(CLIP_MODEL_NAME)
st.sidebar.success(f"模型已就绪 | 设备: {DEVICE.upper()}")

# ---------- 加载数据集 ----------

@st.cache_resource
def load_dataset():
    """加载 Flickr8k 数据集：图片路径列表 + 图片向量 + 文本库"""
    import urllib.request
    import zipfile

    # 下载数据集（如果不存在）
    if not IMAGE_DIR.exists() or not (DATA_DIR / "Flickr8k_text").exists():
        with st.spinner("首次运行，正在下载 Flickr8k 数据集 (约 1.1 GB)..."):
            progress_bar = st.progress(0, text="下载图片...")
            image_zip = DATA_DIR / "Flickr8k_Dataset.zip"
            text_zip = DATA_DIR / "Flickr8k_text.zip"

            if not image_zip.exists():
                urllib.request.urlretrieve(
                    "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
                    str(image_zip),
                )
            progress_bar.progress(40, text="解压图片...")
            if not IMAGE_DIR.exists():
                with zipfile.ZipFile(image_zip) as z:
                    z.extractall(DATA_DIR)

            if not text_zip.exists():
                urllib.request.urlretrieve(
                    "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
                    str(text_zip),
                )
            progress_bar.progress(80, text="解压文本...")
            if not (DATA_DIR / "Flickr8k_text").exists():
                with zipfile.ZipFile(text_zip) as z:
                    z.extractall(DATA_DIR)
            progress_bar.progress(100, text="数据集就绪")

    # 加载图片路径
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    image_paths = [str(p) for p in image_paths]

    # 加载标题
    captions: dict[str, list[str]] = {}
    token_file = DATA_DIR / "Flickr8k_text" / "Flickr8k.token.txt"
    for line in token_file.read_text(encoding="utf-8").strip().split("\n"):
        img_cap, text = line.split("\t")
        img_id = img_cap.split("#")[0]
        captions.setdefault(img_id, []).append(text.strip())

    # 加载或构建图像嵌入缓存
    cache_file = CACHE_DIR / "image_embeddings.npy"
    cache_paths_file = CACHE_DIR / "image_paths.json"

    if cache_file.exists() and cache_paths_file.exists():
        import json
        with open(cache_paths_file, 'r') as f:
            cached_paths = json.load(f)
        if cached_paths == image_paths:
            img_emb = np.load(cache_file)
        else:
            img_emb = _encode_images(image_paths, cache_file, cache_paths_file)
    else:
        img_emb = _encode_images(image_paths, cache_file, cache_paths_file)

    # 构建文本库：每张图片的每条标题作为一个独立文本项
    all_texts: list[str] = []
    all_text_sources: list[str] = []  # 每条文本来源的图片名
    for img_name in sorted(captions.keys()):
        for cap in captions[img_name]:
            all_texts.append(cap)
            all_text_sources.append(img_name)

    return image_paths, img_emb, all_texts, all_text_sources, captions


def _encode_images(image_paths, cache_file, cache_paths_file):
    """编码所有图片并缓存"""
    import json
    all_emb: list[np.ndarray] = []
    total = len(image_paths)
    progress_bar_encode = st.progress(0, text=f"编码图片 0/{total}")

    for i in range(0, total, BATCH_SIZE):
        batch = image_paths[i : i + BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_emb.append(emb.cpu().numpy())
        progress_bar_encode.progress(
            min(i + BATCH_SIZE, total) / total,
            text=f"编码图片 {min(i + BATCH_SIZE, total)}/{total}",
        )

    img_emb = np.concatenate(all_emb, axis=0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, img_emb)
    with open(cache_paths_file, 'w') as f:
        json.dump(image_paths, f)
    progress_bar_encode.empty()
    return img_emb


# 加载数据
image_paths, img_emb, all_texts, all_text_sources, captions_dict = load_dataset()
st.sidebar.success(f"数据集: {len(image_paths)} 张图片, {len(all_texts)} 条描述")


# ---------- 工具函数 ----------

@torch.no_grad()
def encode_text(texts: list[str]) -> np.ndarray:
    """将文本编码为 512 维归一化向量"""
    embeddings: list[np.ndarray] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = processor(text=batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=77)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


@torch.no_grad()
def encode_image(image: Image.Image) -> np.ndarray:
    """将单张图片编码为 512 维归一化向量"""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()


def retrieve(query_emb: np.ndarray, gallery_emb: np.ndarray, top_k: int):
    """余弦相似度检索，返回 (索引列表, 相似度列表)"""
    sim = query_emb @ gallery_emb.T  # [1, M]
    top_idx = np.argsort(sim[0])[::-1][:top_k]
    return top_idx, sim[0, top_idx]


# ---------- 侧边栏配置 ----------

st.sidebar.header("⚙️ 参数设置")
mode = st.sidebar.radio(
    "检索模式",
    ["🎨 以文搜图 (Text → Image)", "📷 以图搜文 (Image → Text)"],
)
top_k = st.sidebar.slider("Top-K 返回数量", min_value=1, max_value=20, value=5)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**关于 CLIP**
- 模型: openai/clip-vit-base-patch32
- 数据集: Flickr8k (8000张)
- 原理: 图文对对比学习
- 参看: `../讲义.md`
""")

# ---------- 模式一：以文搜图 ----------

if "以文搜图" in mode:
    st.header("🎨 以文搜图 — 用自然语言描述你想找的画面")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "输入描述文字（中英文均可）",
            placeholder="例如: a dog running on the beach, 一个在草地上玩耍的小孩",
        )
    with col2:
        search_btn = st.button("🔍 搜索", type="primary", use_container_width=True)

    # 预设示例
    st.caption("💡 试试这些示例：")
    examples = st.columns(5)
    example_queries = [
        "a red car on the street",
        "a person climbing a mountain",
        "a dog playing with a ball",
        "a child on a bicycle",
        "birds flying in the sky",
    ]
    for i, (col, eq) in enumerate(zip(examples, example_queries)):
        with col:
            if st.button(eq[:30] + "…", key=f"ex_{i}"):
                query = eq
                search_btn = True

    if search_btn and query.strip():
        with st.spinner("正在检索..."):
            t0 = time.time()
            # 编码查询文本
            query_emb = encode_text([query.strip()])
            # 检索
            top_idx, scores = retrieve(query_emb, img_emb, top_k)
            elapsed = time.time() - t0

        st.success(f"检索完成！耗时 {elapsed:.2f}s，共命中 {len(top_idx)} 个结果")

        # 显示结果
        cols = min(top_k, 5)
        rows_count = (top_k + cols - 1) // cols

        for row_i in range(rows_count):
            cols_st = st.columns(cols)
            for col_i in range(cols):
                idx = row_i * cols + col_i
                if idx >= top_k:
                    break
                img_idx = top_idx[idx]
                score = scores[idx]
                with cols_st[col_i]:
                    try:
                        img = Image.open(image_paths[img_idx]).convert("RGB")
                        st.image(img, use_container_width=True)
                        st.caption(f"相似度: **{score:.4f}**")
                    except Exception as e:
                        st.error(f"图片加载失败: {e}")

        # 相似度柱状图
        st.markdown("---")
        st.markdown("**📊 相似度分布**")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 2.5))
        labels = [f"#{i+1}" for i in range(top_k)]
        colors = ["#2ecc71" if s == max(scores) else "#3498db" for s in scores]
        bars = ax.bar(range(top_k), scores, color=colors)
        ax.set_xticks(range(top_k))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f'Query: "{query.strip()}" — Top-{top_k} Results')
        ax.set_ylim(0, max(scores) * 1.2 if max(scores) > 0 else 0.5)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{score:.3f}", ha="center", fontsize=9)
        st.pyplot(fig)

# ---------- 模式二：以图搜文 ----------

else:
    st.header("📷 以图搜文 — 上传图片，让 AI 用文字描述它")

    uploaded_file = st.file_uploader(
        "上传一张图片",
        type=["jpg", "jpeg", "png", "webp"],
        help="支持 JPG/PNG/WEBP 格式",
    )

    if uploaded_file:
        # 显示上传的图片
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="你的查询图片", use_container_width=True)

        with col2:
            search_btn = st.button("🔍 搜索相似描述", type="primary")

            if search_btn:
                with st.spinner("正在检索文本库..."):
                    t0 = time.time()
                    # 编码图片
                    query_emb = encode_image(image)
                    # 检索
                    top_idx, scores = retrieve(query_emb, img_emb, top_k)
                    elapsed = time.time() - t0

                st.success(f"检索完成！耗时 {elapsed:.2f}s")

                # 显示匹配的图片 + 它们的描述
                results = []
                for i in range(top_k):
                    img_idx = top_idx[i]
                    img_name = os.path.basename(image_paths[img_idx])
                    img_caps = captions_dict.get(img_name, ["(无描述)"])
                    results.append({
                        "idx": i + 1,
                        "image": image_paths[img_idx],
                        "score": scores[i],
                        "captions": img_caps,
                        "img_name": img_name,
                    })

        if search_btn and results:
            st.markdown("---")
            st.markdown("### 📋 检索结果")

            for r in results:
                cols = st.columns([1, 3])
                with cols[0]:
                    st.image(r["image"], use_container_width=True)
                    st.caption(f"相似度: **{r['score']:.4f}**")
                with cols[1]:
                    st.markdown(f"**序号**: #{r['idx']}  |  **文件名**: `{r['img_name']}`")
                    st.markdown("**图片原始描述**:")
                    for c in r["captions"]:
                        st.markdown(f"- {c}")
                st.divider()

            # 相似度柱状图
            st.markdown("**📊 相似度分布**")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 2.5))
            r_scores = [r["score"] for r in results]
            labels = [f"#{r['idx']}" for r in results]
            colors = ["#2ecc71" if s == max(r_scores) else "#3498db" for s in r_scores]
            bars = ax.bar(range(len(r_scores)), r_scores, color=colors)
            ax.set_xticks(range(len(r_scores)))
            ax.set_xticklabels(labels)
            ax.set_ylabel("Cosine Similarity")
            ax.set_title(f"图片检索 — Top-{top_k} 最相似图片")
            ax.set_ylim(0, max(r_scores) * 1.2 if max(r_scores) > 0 else 0.5)
            for bar, score in zip(bars, r_scores):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{score:.3f}", ha="center", fontsize=9)
            st.pyplot(fig)

    else:
        st.info("👆 请上传一张图片来开始检索，或切换到「以文搜图」模式输入文字。")
