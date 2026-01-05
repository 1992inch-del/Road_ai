print("RUNNING FILE:", __file__)

from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unet import UNet
from dataset_massroad_patch import MassRoadPatchDataset


def dice_iou_from_logits(logits, targets, thr=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()

    # targets: (B,1,H,W)
    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3))
    iou = (inter + 1e-6) / (union + 1e-6)

    dice = (2 * inter + 1e-6) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-6)
    return dice.mean().item(), iou.mean().item()


def main():
    # === 路径 ===
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "Massachusetts"

    RUN_NAME = "run1"   # ←←← 想要在上一个的基础上继续训练时不要改名；只有做“全新实验”才改名。这个只要改名，predict_unet.py和eval_unet_patch.py都要改名。
    out_dir = project_root / "results" / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)


    # === 超参数（先用最稳的）===
    epochs = 5 #这个是总轮数，比如第一次进行到5，第二次运行时就会从6开始，直到运行到规定的次数结束
    batch_size = 2
    lr = 1e-3
    num_workers = 0  # Windows 先用0，稳定后再改2/4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = MassRoadPatchDataset(
        str(data_root),
        split="train",
        patch_size=512,
        use_rgb=True
    )

    val_ds = MassRoadPatchDataset(
        str(data_root),
        split="val",
        patch_size=512,  #Windows稳定后可以更改为768
        use_rgb=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )


    x, y, name = train_ds[0]
    print("FINAL CHECK:", name, x.shape, y.shape)


    # === 模型 ===
    model = UNet(in_channels=3, out_channels=1, base=32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path = out_dir / "best_unet.pth"
    best_iou = -1.0
    start_epoch = 1

    # ====== Resume training if checkpoint exists ======
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])

        best_iou = float(ckpt.get("best_iou", best_iou))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from epoch {start_epoch-1}, best_iou={best_iou:.4f}")
    else:
        print("No checkpoint found, training from scratch.")


    for ep in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = criterion(logits, masks)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # === 验证 ===
        model.eval()
        val_loss = 0.0
        dices, ious = [], []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)

                loss = criterion(logits, masks)
                val_loss += loss.item()

                d, i = dice_iou_from_logits(logits, masks)
                dices.append(d)
                ious.append(i)

        val_loss /= max(len(val_loader), 1)
        val_dice = sum(dices) / max(len(dices), 1)
        val_iou = sum(ious) / max(len(ious), 1)

        dt = time.time() - t0
        print(f"Epoch {ep}/{epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
              f"dice={val_dice:.4f} iou={val_iou:.4f} | {dt:.1f}s")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "epoch": ep,
                    "best_iou": best_iou,
                },
                ckpt_path
            )

            print(f"  Saved: {ckpt_path} (best_iou={best_iou:.4f})")

    print("Done. Best checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()