"""
使用 ModelScope 将 CLIP 与 ViT 预训练权重下载到 checkpoint 目录。
运行: python download_models.py
"""
import json
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoint"
PATHS_JSON = CHECKPOINT_DIR / "model_paths.json"


def main():
    from modelscope import snapshot_download

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("下载 CLIP（openai-mirror/clip-vit-base-patch32）...")
    clip_dir = snapshot_download("openai-mirror/clip-vit-base-patch32", cache_dir=str(CHECKPOINT_DIR))
    print("  ->", clip_dir)

    print("下载 ViT（google/vit-base-patch16-224）...")
    vit_dir = snapshot_download("google/vit-base-patch16-224", cache_dir=str(CHECKPOINT_DIR))
    print("  ->", vit_dir)

    paths = {"clip": clip_dir, "vit": vit_dir}
    with open(PATHS_JSON, "w") as f:
        json.dump(paths, f, indent=2)
    print("路径已写入", PATHS_JSON)


if __name__ == "__main__":
    main()
