"""
本项目的超参与路径：ViT 图像编码（768 维）+ CLIP 文本编码（512 维）→ 投影 256 维，对比学习。

数据路径、训练超参、GPU、实验记录等由 run.sh + run_train.py 统一管理并覆盖本文件默认值。
直接运行 train/train.py 时使用下面默认值（默认 dataset/flickr8k）。
"""
import json
from pathlib import Path
import torch

_ROOT = Path(__file__).resolve().parent

# ==================== 模型 checkpoint（ModelScope 下载后从本地加载） ====================
checkpoint_dir = _ROOT / "checkpoint"
_paths_json = checkpoint_dir / "model_paths.json"
if _paths_json.exists():
    with open(_paths_json) as f:
        _paths = json.load(f)
    clip_local_dir = _paths.get("clip")
    vit_local_dir = _paths.get("vit")
else:
    clip_local_dir = None
    vit_local_dir = None

# ==================== 数据路径（默认 flickr8k；由 run.sh 覆盖） ====================
debug = False
dataset_name = "flickr8k"
captions_path = str(_ROOT / "dataset" / "flickr8k")
image_path = str(_ROOT / "dataset" / "flickr8k" / "Images")
gpu = "0"
mixed_precision_mode = "no"

# ==================== 训练超参（由 run.sh 覆盖） ====================
batch_size = 16
grad_accum_steps = 1
freeze_encoder_epochs = 0
num_workers = 4
lr = 1e-5
weight_decay = 1e-4
warmup_epochs = 5
min_lr_ratio = 0.05
epochs = 80
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_device_ids = None

# ==================== 实验记录（由 run.sh 覆盖） ====================
tracker_project = "CLIP"

# ==================== 图像编码器（timm ViT） ====================
model_name = 'vit_base_patch16_224'  # timm 中的 ViT-B/16
image_embedding = 768                # ViT-B 输出维度

# ==================== 文本编码器（优先本地 checkpoint，否则 HF 名） ====================
text_encoder_model = "openai/clip-vit-base-patch32"  # 未用本地时 HF 模型名
text_embedding = 512
text_tokenizer = "openai/clip-vit-base-patch32"      # 未用本地时 HF tokenizer 名
max_length = 77                                       # CLIP 论文规定的最大文本长度

# ==================== 编码器通用设置 ====================
#pretrained = True、trainable = True 就是控制“用预训练权重 + 全部参与训练”的开关；
# 若改成 trainable = False，就变成只训练投影头、冻结两个编码器。
pretrained = True    # 使用预训练权重（ViT 在小数据集上需要预训练才能收敛）
trainable = True     # 允许微调编码器参数
temperature = 0.07   # 更接近 CLIP 常用尺度，提升相似度分布区分度

# ==================== 图像预处理 ====================
size = 224

# ==================== 投影头（图像 768 / 文本 512 → 256） ====================
projection_dim = 256
dropout = 0.1
