from pathlib import Path
import numpy as np
import torch
import tifffile as tiff

from dataset_massroad_patch import MassRoadPatchDataset
from unet import UNet


def save_mask_png_like(path: Path, mask01: np.ndarray):
    # mask01: (H,W) float 0/1
    out = (np.clip(mask01, 0, 1) * 255).astype(np.uint8)
    tiff.imwrite(str(path), out)


def save_prob_tif(path: Path, prob: np.ndarray):
    # prob: (H,W) float32, range 0~1
    tiff.imwrite(str(path), prob.astype(np.float32))


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "Massachusetts"

    RUN_NAME = "run1"   # 必须和 train_unet.py 一样
    ckpt_path = project_root / "results" / RUN_NAME / "best_unet.pth"

    out_dir = project_root / "results" / RUN_NAME / "pred_masks"
    out_dir.mkdir(parents=True, exist_ok=True)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 选 test 集做推理（patch，与训练一致）
    ds = MassRoadPatchDataset(str(data_root), split="test", patch_size=512, use_rgb=True)

    model = UNet(in_channels=3, out_channels=1, base=32).to(device)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from {ckpt_path}")
    model.eval()

    # 只先跑前 20 张看效果
    n = min(20, len(ds))
    with torch.no_grad():
        for i in range(n):
            img, _, name = ds[i]
            stem = Path(str(name)).stem

            img = img.unsqueeze(0).to(device)  # (1,3,512,512)

            logits = model(img)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (512,512)
            pred = (prob > 0.5).astype(np.float32)

            out_prob = out_dir / f"{stem}_prob.tif"
            out_pred = out_dir / f"{stem}_pred.tif"


            save_prob_tif(out_prob, prob)        # 概率图（0~1）
            save_mask_png_like(out_pred, pred)   # 二值图（0/255）

            print("saved:", out_pred)

    print("Done. Output dir:", out_dir)


if __name__ == "__main__":
    main()
