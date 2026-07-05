"""
快速测试：造少量假数据，跑 1 个 epoch + 一次推理，验证脚本和 notebook 能跑通。
用法：在已安装依赖的环境（pip install -r requirements.txt）
      下，在项目根目录执行  python run_quick_test.py
"""
import os
import sys
import csv

# 无头环境里推理时不要弹窗
import matplotlib
matplotlib.use("Agg")

# 保证能 import 到项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def make_fake_data():
    base = os.path.join(os.path.dirname(__file__), "fake_data")
    img_dir = os.path.join(base, "Images")
    os.makedirs(img_dir, exist_ok=True)

    import numpy as np
    import cv2

    n_images = 20
    captions_per_image = 5
    rows = []
    for i in range(n_images):
        fname = f"{i}.png"
        # 224x224 RGB 随机图，存成 PNG
        img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(img_dir, fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for j in range(captions_per_image):
            rows.append({"image": fname, "caption": f"a photo of object {i} view {j}", "id": i})

    with open(os.path.join(base, "captions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "caption", "id"])
        w.writeheader()
        w.writerows(rows)
    return base


def test_scripts():
    base = make_fake_data()
    img_dir = os.path.join(base, "Images")

    import config as CFG
    CFG.image_path = img_dir
    CFG.captions_path = base
    CFG.debug = True
    CFG.epochs = 1
    CFG.batch_size = 4

    print(" [scripts] 开始训练 1 epoch ...")
    from train import main
    main()
    print(" [scripts] 训练完成")

    print(" [scripts] 跑一次推理 ...")
    from train import make_train_valid_dfs
    from inference import get_image_embeddings, find_matches
    _, valid_df = make_train_valid_dfs()
    model, image_embeddings = get_image_embeddings(valid_df, "best.pt")
    find_matches(model, image_embeddings, "a photo of object", valid_df["image"].values, n=3)
    print(" [scripts] 推理完成")
    return True


def test_notebook():
    root = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(root, "fake_data")
    if not os.path.isdir(base):
        make_fake_data()
    base_abs = os.path.abspath(base)
    img_dir_abs = os.path.join(base_abs, "Images")

    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    nb_path = os.path.join(os.path.dirname(__file__), "notebooks", "clip_tutorial.ipynb")
    if not os.path.isfile(nb_path):
        print(" [notebook] 未找到 notebooks/clip_tutorial.ipynb，跳过")
        return True

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # 把 CFG 里的路径改成 fake_data（绝对路径），epochs 改成 1
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if "image_path" in src and "captions_path" in src and "class CFG" in src:
            src = src.replace("/your/path/to/Flicker-8k/Images", img_dir_abs.replace("\\", "/"))
            src = src.replace("/your/path/to/Flicker-8k", base_abs.replace("\\", "/"))
            src = src.replace("epochs = 5", "epochs = 1")
            cell.source = src
            break

    print(" [notebook] 执行 notebook ...")
    ep = ExecutePreprocessor(timeout=180)
    ep.preprocess(nb, {"metadata": {"path": root}})
    print(" [notebook] 执行完成")
    return True


if __name__ == "__main__":
    print("=== 快速测试（假数据） ===\n")
    ok_script = False
    ok_nb = False
    try:
        ok_script = test_scripts()
    except Exception as e:
        print(f" [scripts] 失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        ok_nb = test_notebook()
    except Exception as e:
        print(f" [notebook] 失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== 结果 ===")
    print("  脚本:", "通过" if ok_script else "失败")
    print("  Notebook:", "通过" if ok_nb else "失败")
    sys.exit(0 if (ok_script and ok_nb) else 1)
