#!/usr/bin/env bash
# =============================================================================
# CLIP 训练统一入口：所有可配置项在下方显式写出。
# 单卡/多卡由 GPU 变量决定，accelerate 自动处理分布式。
# 用法：按需修改下面变量，然后执行 bash run.sh
# =============================================================================

# ------------------------------ 数据集 ------------------------------
DATASET="flickr30k"
CAPTIONS_PATH=""
IMAGE_PATH=""
DEBUG="no"

# ------------------------------ GPU ------------------------------
# 单卡："0"    多卡："0,1,2,3"（自动计算进程数）
GPU="0"
MIXED_PRECISION="bf16"

# ------------------------------ 训练超参 ------------------------------
# 第十次：模型效果多改合一（时间有限一次多调），见 优化记录.md 第十次试验
EPOCHS=40
BATCH_SIZE=128
GRAD_ACCUM_STEPS=2
FREEZE_ENCODER_EPOCHS=3
NUM_WORKERS=8
LR="1e-5"
WEIGHT_DECAY="1e-3"
WARMUP_EPOCHS=5
MIN_LR_RATIO="0.05"
# ------------------------------ SwanLab ------------------------------
# 去 https://swanlab.cn → 个人设置 → API Key 获取
SWANLAB_API_KEY="这里填写你自己的API KEY"
PROJECT="CLIP"

# =============================================================================
# 以下为调用逻辑，一般无需修改
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 从 GPU 字符串自动算进程数：单卡 "0" → 1，多卡 "0,1,2,3" → 4
NUM_PROCESSES=$(echo "$GPU" | tr ',' '\n' | wc -l)

DEBUG_ARG=""
if [ "$DEBUG" = "yes" ]; then
  DEBUG_ARG="--debug"
else
  DEBUG_ARG="--no_debug"
fi

EXTRA_ARGS=()
[ -n "$CAPTIONS_PATH" ] && EXTRA_ARGS+=(--captions_path "$CAPTIONS_PATH")
[ -n "$IMAGE_PATH" ]   && EXTRA_ARGS+=(--image_path "$IMAGE_PATH")

CUDA_VISIBLE_DEVICES="$GPU" accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision "$MIXED_PRECISION" \
  run_train.py \
  --dataset "$DATASET" \
  --gpu "$GPU" \
  --mixed_precision_mode "$MIXED_PRECISION" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" \
  --freeze_encoder_epochs "$FREEZE_ENCODER_EPOCHS" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --min_lr_ratio "$MIN_LR_RATIO" \
  --swanlab_key "$SWANLAB_API_KEY" \
  --project "$PROJECT" \
  $DEBUG_ARG \
  "${EXTRA_ARGS[@]}"
