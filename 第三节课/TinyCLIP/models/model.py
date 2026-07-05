"""
CLIP 主模型：双塔（图像 ViT + 文本 CLIP Transformer）→ 投影头 → L2 归一化 → 对比损失。
论文 Section 2 里的双塔、对比学习、正负样本定义、对称损失、归一化、temperature 都在下面 forward 里，
对着这段代码看能把「CLIP 怎么训练、为什么这样设计」搞清楚。

多卡支持：forward 内部自动 all_gather embeddings，保证对比损失在全局 batch 上计算。
"""
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

import config as CFG
from .encoders import ImageEncoder, TextEncoder, ProjectionHead


# ==================== 多卡 all_gather（保留梯度） ====================

class _GatherWithGrad(torch.autograd.Function):
    """all_gather + cat，反向传播时只取本 GPU 对应的梯度切片。"""

    @staticmethod
    def forward(ctx, tensor):
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        ctx.rank = dist.get_rank()
        ctx.batch_size = tensor.shape[0]
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        start = ctx.rank * ctx.batch_size
        return grad_output[start : start + ctx.batch_size].contiguous()


def _gather(tensor, with_grad=True):
    """多卡时 all_gather 拼接；单卡时直接返回原 tensor。"""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return tensor
    if with_grad:
        return _GatherWithGrad.apply(tensor)
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


# ==================== CLIP 主模型 ====================

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # 获取图像和文本特征
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # 投影到共同维度空间
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # 论文：L2 归一化后点积 = 余弦相似度，有界且稳定
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # 多卡：聚合所有 GPU 的 embeddings 和 id，在全局 batch 上算对比损失
        image_embeddings = _gather(image_embeddings)
        text_embeddings = _gather(text_embeddings)

        # 论文公式：logits[i,j] = 第 i 个文本与第 j 张图的相似度 / temperature
        # temperature 控制 softmax 锐度，越小分布越尖
        logits = (text_embeddings @ image_embeddings.T) / self.temperature

        # 论文里「batch 内 N² 个 pair、同图为正」的落地：用 id 建 target，同 id 为正样本
        # 一图多 caption 时多条 caption 共享同一 id，均分 1（soft label）
        ids = batch["id"]
        if ids.ndim > 1:
            ids = ids.view(ids.size(0))
        ids = _gather(ids, with_grad=False)
        positive_mask = ids.unsqueeze(1) == ids.unsqueeze(0)
        positive_counts = positive_mask.sum(dim=-1, keepdim=True)
        targets = positive_mask.float() / positive_counts.clamp_min(1.0)

        # 对称损失：text→image 和 image→text 两个方向都算 CE 再平均，论文 Section 2
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    """软标签交叉熵，targets 可多正样本（一图多 caption 时同图均分 1）。"""
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
