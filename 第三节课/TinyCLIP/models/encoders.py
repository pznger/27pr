"""
图像编码器（timm ViT 或 本地 HF ViT）、文本编码器（HF CLIP，可本地）、投影头。
当 config.vit_local_dir / clip_local_dir 已设置时从 ModelScope 下载的本地目录加载。
"""
import torch
from torch import nn
import config as CFG
from transformers import CLIPTextModel, CLIPTextConfig, ViTModel as HfViTModel


class ImageEncoder(nn.Module):
    """
    图像编码器：vit_local_dir 存在时用 HF ViTModel 从本地加载，否则用 timm ViT-B/16。
    输入 (batch, 3, 224, 224) → 输出 (batch, 768)。
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        vit_local = getattr(CFG, "vit_local_dir", None)
        if vit_local and pretrained:
            self.model = HfViTModel.from_pretrained(vit_local)
            self._hf_vit = True
        else:
            import timm
            self.model = timm.create_model(
                model_name, pretrained, num_classes=0, global_pool="avg"
            )
            self._hf_vit = False
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        if self._hf_vit:
            out = self.model(pixel_values=x)
            return out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
        return self.model(x)


class TextEncoder(nn.Module):
    """
    文本编码器：clip_local_dir 存在时从本地加载，否则用 text_encoder_model。取 EOS 作为句子表示。
    """

    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        load_from = getattr(CFG, "clip_local_dir", None) or model_name
        if pretrained:
            self.model = CLIPTextModel.from_pretrained(load_from)
        else:
            self.model = CLIPTextModel(config=CLIPTextConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        # CLIP 论文：取 EOS token 作为句子表示（不是 BERT 的 CLS）
        # EOS 的 token id 是词表中最大值(49407)，所以 argmax 即可定位
        eos_idx = input_ids.argmax(dim=-1)
        return last_hidden_state[torch.arange(last_hidden_state.size(0)), eos_idx]


class ProjectionHead(nn.Module):
    """
    投影头：将图像/文本编码器输出投影到共同的低维空间，便于算相似度。
    论文里是单层线性投影；这里用 Linear→GELU→Linear + 残差 + LayerNorm，
    多了一层非线性，对理解「投影到共同空间」没影响，收敛通常更稳一点。
    """

    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x