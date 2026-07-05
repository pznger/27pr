"""
多卡推理入口。通过 config.inference_device_ids 指定使用的 GPU，如 [0,1,2] 表示 3 卡。
单卡或未设置 inference_device_ids 时与 inference 行为一致。
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch

import config as CFG
from train import make_train_valid_dfs
from inference import get_image_embeddings, find_matches
from models import CLIPModel


def _wrap_model(model_path):
    device_ids = getattr(CFG, "inference_device_ids", None)
    model = CLIPModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    if device_ids and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.to(f"cuda:{device_ids[0]}")
    else:
        model = model.to(CFG.device)
    return model


def main():
    _, valid_df = make_train_valid_dfs()
    model_path = "best.pt"
    model = _wrap_model(model_path)
    model, image_embeddings = get_image_embeddings(valid_df, model_path, model=model)
    find_matches(model, image_embeddings, "a dog on the beach", valid_df["image"].values)


if __name__ == "__main__":
    main()
