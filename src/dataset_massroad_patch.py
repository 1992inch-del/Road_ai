from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff


def _read_tiff_hwc(path: Path) -> np.ndarray:
    arr = tiff.imread(str(path))
    if arr.ndim == 2:
        return arr[..., None]  # (H,W,1)
    if arr.ndim != 3:
        raise RuntimeError(f"Unsupported TIFF shape {arr.shape} from {path}")
    # (C,H,W) -> (H,W,C) if looks like channels-first
    if arr.shape[0] <= 8 and arr.shape[2] > 8:
        arr = np.transpose(arr, (1, 2, 0))
    return arr  # (H,W,C)


def _to_float01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    # 更快的归一化：按最大最小（比 percentile 快很多）
    mx = float(img.max()) if img.max() > 0 else 1.0
    mn = float(img.min())
    img = (img - mn) / (mx - mn + 1e-6)
    return np.clip(img, 0.0, 1.0)


def _make_pairs(img_dir: Path, mask_dir: Path):
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")])
    masks = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")])
    mask_map = {m.stem: m for m in masks}

    pairs = []
    for im in imgs:
        m = mask_map.get(im.stem)
        if m is not None:
            pairs.append((im, m))
    if not pairs:
        raise RuntimeError(f"No pairs found. img_dir={img_dir}, mask_dir={mask_dir}")
    return pairs


def _random_crop_pair(img_t: torch.Tensor, mask_t: torch.Tensor, size: int):
    # img_t: (C,H,W), mask_t: (1,H,W)
    _, H, W = img_t.shape
    if H < size or W < size:
        # 不足则中心裁到 min(H,W) 的 multiple，或者直接 pad 再裁
        pad_h = max(0, size - H)
        pad_w = max(0, size - W)
        if pad_h > 0 or pad_w > 0:
            img_t = torch.nn.functional.pad(img_t, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
            mask_t = torch.nn.functional.pad(mask_t, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
            _, H, W = img_t.shape

    top = torch.randint(0, H - size + 1, (1,)).item()
    left = torch.randint(0, W - size + 1, (1,)).item()
    img_t = img_t[:, top:top+size, left:left+size]
    mask_t = mask_t[:, top:top+size, left:left+size]
    return img_t, mask_t


class MassRoadPatchDataset(Dataset):
    def __init__(self, root: str, split: str, patch_size: int = 512, use_rgb: bool = True):
        root = Path(root)
        assert split in ("train", "val", "test")

        self.img_dir = root / split
        self.mask_dir = root / f"{split}_labels"
        self.pairs = _make_pairs(self.img_dir, self.mask_dir)

        self.patch_size = patch_size
        self.use_rgb = use_rgb

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = _read_tiff_hwc(img_path)
        mask = _read_tiff_hwc(mask_path)

        img = _to_float01(img)

        # 通道：默认转 RGB(3)
        if self.use_rgb:
            if img.shape[2] >= 3:
                img = img[:, :, :3]
            else:
                img = np.repeat(img, 3, axis=2)

        # mask -> 二值 (H,W)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.float32)

        img_t = torch.from_numpy(np.transpose(img, (2, 0, 1)).copy()).float()  # (C,H,W)
        mask_t = torch.from_numpy(mask[None, ...].copy()).float()              # (1,H,W)

        # 核心：随机裁 512×512，保证 img/mask 永远同尺寸
        img_t, mask_t = _random_crop_pair(img_t, mask_t, size=self.patch_size)

        return img_t, mask_t, img_path.name
