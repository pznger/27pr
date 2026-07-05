"""
推理：加载 best.pt，对验证集图像算嵌入；给定文本 query，用余弦相似度检索最相关的图并画 3×3。
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import gc
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPTokenizer
import matplotlib.pyplot as plt

import config as CFG
from train import build_loaders
from models import CLIPModel


def _unwrap(model):
    """取出被 DataParallel / DDP 包装的原始模型"""
    return getattr(model, "module", model)


def get_image_embeddings(valid_df, model_path, model=None):
    tokenizer = CLIPTokenizer.from_pretrained(CFG.clip_local_dir or CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    device = CFG.device

    if model is None:
        model = CLIPModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        device = next(model.parameters()).device
    model.eval()

    raw = _unwrap(model)
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = raw.image_encoder(batch["image"].to(device))
            image_embeddings = raw.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)


def find_matches(model, image_embeddings, query, image_filenames, n=9):
    device = next(model.parameters()).device
    tokenizer = CLIPTokenizer.from_pretrained(CFG.clip_local_dir or CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(device)
        for key, values in encoded_query.items()
    }
    raw = _unwrap(model)
    with torch.no_grad():
        text_features = raw.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = raw.text_projection(text_features)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    num_images = dot_similarity.squeeze(0).size(0)
    k = min(n * 5, num_images)
    _, indices = torch.topk(dot_similarity.squeeze(0), k)
    step = max(1, k // n)
    matches = [image_filenames[indices[i].item()] for i in range(0, k, step)][:n]

    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{CFG.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")

    plt.show()
