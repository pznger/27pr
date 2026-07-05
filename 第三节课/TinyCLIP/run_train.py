#!/usr/bin/env python3
"""
统一训练入口：解析命令行参数 → 覆盖 config → 调用 train.main()。
由 run.sh 通过 accelerate launch 调用，单卡/多卡自动处理。
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="CLIP 训练统一入口")
    p.add_argument("--dataset", type=str, default="flickr8k",
                   choices=["flickr8k", "flickr30k"])
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--mixed_precision_mode", type=str, default="no",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--captions_path", type=str, default="")
    p.add_argument("--image_path", type=str, default="")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no_debug", action="store_true")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--freeze_encoder_epochs", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--swanlab_key", type=str, default="")
    p.add_argument("--project", type=str, default="CLIP")
    return p.parse_args()


def apply_config(args):
    """把命令行参数写入 config 模块，供 train / dataset / models 使用"""
    import config as CFG

    # 数据路径：优先显式指定，否则从 dataset 名推导
    if args.captions_path and args.image_path:
        CFG.captions_path = args.captions_path
        CFG.image_path = args.image_path
    else:
        base = ROOT / "dataset" / args.dataset
        CFG.captions_path = args.captions_path or str(base)
        CFG.image_path = args.image_path or str(base / "Images")

    if args.debug:
        CFG.debug = True
    elif args.no_debug:
        CFG.debug = False

    CFG.epochs = args.epochs
    CFG.dataset_name = args.dataset
    CFG.gpu = args.gpu
    CFG.mixed_precision_mode = args.mixed_precision_mode
    CFG.batch_size = args.batch_size
    CFG.grad_accum_steps = max(1, args.grad_accum_steps)
    CFG.freeze_encoder_epochs = max(0, args.freeze_encoder_epochs)
    CFG.num_workers = args.num_workers
    CFG.lr = args.lr
    CFG.weight_decay = args.weight_decay
    CFG.warmup_epochs = max(0, args.warmup_epochs)
    CFG.min_lr_ratio = float(args.min_lr_ratio)
    CFG.tracker_project = args.project


def main():
    args = parse_args()
    if args.swanlab_key:
        os.environ["SWANLAB_API_KEY"] = args.swanlab_key
    apply_config(args)

    from train.train import main as train_main
    train_main()


if __name__ == "__main__":
    main()
