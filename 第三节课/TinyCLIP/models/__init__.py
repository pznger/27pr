"""模型结构：图像/文本编码器与 CLIP 主模型。"""
from .encoders import ImageEncoder, TextEncoder, ProjectionHead
from .model import CLIPModel

__all__ = ["CLIPModel", "ImageEncoder", "TextEncoder", "ProjectionHead"]
