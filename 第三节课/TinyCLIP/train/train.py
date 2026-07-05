"""
统一训练入口：基于 HuggingFace Accelerate，单卡/多卡同一代码路径。
单卡时 Accelerate 是透明包装，零开销；多卡时自动处理 DDP。
启动方式统一为：accelerate launch run_train.py --args...
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
from tqdm import tqdm
import math

import torch
from transformers import CLIPTokenizer
from accelerate import Accelerator, DistributedDataParallelKwargs

import config as CFG
from dataset import CLIPDataset, get_transforms
from models import CLIPModel
from utils import AvgMeter, get_lr


def set_encoder_trainable(model, trainable: bool):
    """两阶段训练开关：只冻结/解冻双编码器，投影头始终可训练。"""
    for p in model.image_encoder.parameters():
        p.requires_grad = trainable
    for p in model.text_encoder.parameters():
        p.requires_grad = trainable


def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{CFG.captions_path}/captions.csv")
    image_ids = dataframe["id"].drop_duplicates().to_numpy()
    if CFG.debug:
        image_ids = image_ids[:100]
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        dataframe["id"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=(mode == "train"),
    )
    return dataloader


def train_epoch(model, train_loader, optimizer, accelerator):
    loss_meter = AvgMeter()
    optimizer.zero_grad()
    tqdm_object = tqdm(train_loader, total=len(train_loader),
                       disable=not accelerator.is_main_process)
    for batch in tqdm_object:
        batch = {k: v.to(accelerator.device)
                 for k, v in batch.items() if k != "caption"}
        with accelerator.accumulate(model):
            loss = model(batch)
            # 多卡补偿：all_gather 后每 GPU 只贡献 1/N 梯度，DDP 再平均又除 N，
            # 乘回 num_processes 使梯度量级与单卡一致，不需要额外调学习率
            accelerator.backward(loss * accelerator.num_processes)
            optimizer.step()
            optimizer.zero_grad()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader, accelerator):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader),
                       disable=not accelerator.is_main_process)
    for batch in tqdm_object:
        batch = {k: v.to(accelerator.device)
                 for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=CFG.grad_accum_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    train_df, valid_df = make_train_valid_dfs()
    tokenizer = CLIPTokenizer.from_pretrained(CFG.clip_local_dir or CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    model = CLIPModel()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    total_epochs = max(1, CFG.epochs)
    warmup_epochs = min(max(0, CFG.warmup_epochs), total_epochs)
    min_lr_ratio = min(max(CFG.min_lr_ratio, 0.0), 1.0)

    def lr_lambda(epoch_idx: int) -> float:
        current = epoch_idx + 1
        if warmup_epochs > 0 and current <= warmup_epochs:
            return current / warmup_epochs
        if total_epochs == warmup_epochs:
            return min_lr_ratio
        progress = (current - warmup_epochs) / (total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )

    # 实验追踪（可选）：仅主进程初始化，没装 swanlab 也不影响训练
    tracker = None
    if accelerator.is_main_process:
        try:
            from experiment_tracker import ExperimentTracker
            tracker = ExperimentTracker(
                project=getattr(CFG, "tracker_project", "CLIP"),
                config={
                    "dataset": getattr(CFG, "dataset_name", "unknown"),
                    "captions_path": CFG.captions_path,
                    "image_path": CFG.image_path,
                    "debug": CFG.debug,
                    "gpu": getattr(CFG, "gpu", "unknown"),
                    "mixed_precision": getattr(CFG, "mixed_precision_mode", "no"),
                    "optimizer": "AdamW",
                    "scheduler": "warmup_cosine",
                    "model_name": CFG.model_name,
                    "text_encoder": CFG.text_encoder_model,
                    "lr": CFG.lr,
                    "weight_decay": CFG.weight_decay,
                    "batch_size": CFG.batch_size,
                    "grad_accum_steps": CFG.grad_accum_steps,
                    "two_stage_training": CFG.freeze_encoder_epochs > 0,
                    "freeze_encoder_epochs": CFG.freeze_encoder_epochs,
                    "effective_global_batch_size": (
                        CFG.batch_size * CFG.grad_accum_steps * accelerator.num_processes
                    ),
                    "num_workers": CFG.num_workers,
                    "epochs": CFG.epochs,
                    "warmup_epochs": CFG.warmup_epochs,
                    "min_lr_ratio": CFG.min_lr_ratio,
                    "image_embedding": CFG.image_embedding,
                    "text_embedding": CFG.text_embedding,
                    "projection_dim": CFG.projection_dim,
                    "temperature": CFG.temperature,
                    "num_processes": accelerator.num_processes,
                },
            )
        except Exception:
            pass

    best_loss = float("inf")
    last_encoder_trainable = None
    for epoch in range(CFG.epochs):
        encoder_trainable = epoch >= CFG.freeze_encoder_epochs
        phase = "stage2_full_finetune" if encoder_trainable else "stage1_head_only"
        base_model = accelerator.unwrap_model(model)
        set_encoder_trainable(base_model, encoder_trainable)
        if accelerator.is_main_process and last_encoder_trainable != encoder_trainable:
            print(
                f"[two-stage] epoch {epoch + 1}: "
                f"{'解冻编码器，全量微调' if encoder_trainable else '冻结编码器，仅训练投影头'}"
            )
        last_encoder_trainable = encoder_trainable

        if accelerator.is_main_process:
            print(f"Epoch: {epoch + 1}")

        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, accelerator)

        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader, accelerator)

        if tracker is not None:
            # phase_id: 0=仅投影头, 1=全量微调，SwanLab 图表只支持数值
            tracker.log({
                "train_loss": train_loss.avg,
                "valid_loss": valid_loss.avg,
                "lr": get_lr(optimizer),
                "epoch": epoch + 1,
                "phase": 1 if encoder_trainable else 0,
                "encoder_trainable": int(encoder_trainable),
            })

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), "best.pt")
                print("Saved Best Model!")

        lr_scheduler.step()

    if tracker is not None:
        tracker.finish()


if __name__ == "__main__":
    main()
