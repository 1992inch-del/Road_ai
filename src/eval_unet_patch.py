from pathlib import Path
import numpy as np
import torch

from dataset_massroad_patch import MassRoadPatchDataset
from unet import UNet


# ================== 指标函数 ==================
def compute_metrics(pred: np.ndarray, gt: np.ndarray):
    """
    pred, gt: (H,W) binary {0,1}
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    eps = 1e-6
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return iou, dice, precision, recall


# ================== 主流程 ==================
def main():
    # -------- 路径 & RUN_NAME（必须和 train / predict 一致） --------
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "Massachusetts"

    RUN_NAME = "run1"   # ←←← 与 train_unet.py / predict_unet.py 完全一致
    ckpt_path = project_root / "results" / RUN_NAME / "best_unet.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # -------- device --------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # -------- 数据集（patch，与训练一致） --------
    ds = MassRoadPatchDataset(
        str(data_root),
        split="test",        # 若想评 val，改成 "val"
        patch_size=512,
        use_rgb=True
    )

    print(f"Evaluating on {len(ds)} patches")

    # -------- 模型加载 --------
    model = UNet(in_channels=3, out_channels=1, base=32).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded checkpoint from {ckpt_path}")

    # -------- 评估 --------
    ious, dices, precisions, recalls = [], [], [], []

    with torch.no_grad():
        for i in range(len(ds)):
            img, gt, _ = ds[i]          # img:(3,H,W), gt:(1,H,W)
            img = img.unsqueeze(0).to(device)

            logits = model(img)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)

            gt = gt[0].numpy().astype(np.uint8)

            iou, dice, prec, rec = compute_metrics(pred, gt)

            ious.append(iou)
            dices.append(dice)
            precisions.append(prec)
            recalls.append(rec)

    # -------- 汇总输出 --------
    print("\n===== Patch-level Evaluation =====")
    print(f"RUN_NAME   : {RUN_NAME}")
    print(f"Num patches: {len(ds)}")
    print(f"Mean IoU   : {np.mean(ious):.4f}")
    print(f"Mean Dice  : {np.mean(dices):.4f}")
    print(f"Precision  : {np.mean(precisions):.4f}")
    print(f"Recall     : {np.mean(recalls):.4f}")


if __name__ == "__main__":
    main()
