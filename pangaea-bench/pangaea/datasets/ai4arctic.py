import numpy as np
import rasterio
import torch
from pathlib import Path
from torch.utils.data import Dataset


class AI4ArcticDataset(Dataset):
    def __init__(self, root_path, split, img_size=512, ignore_index=255, is_train=False, **kwargs):
        self.root = Path(root_path)
        self.split = split
        self.img_size = int(img_size)
        self.ignore_index = int(ignore_index)
        self.is_train = bool(is_train)

        # Required by SegEvaluator
        self.dataset_name = kwargs.get("dataset_name", "ai4arctic")
        self.classes = kwargs.get(
            "classes",
            ["open_water", "new_ice", "young_ice", "thin_first_year_ice", "thick_first_year_ice", "old_ice"],
        )
        self.num_classes = int(kwargs.get("num_classes", len(self.classes)))

        self.img_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Missing images dir: {self.img_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing masks dir: {self.mask_dir}")

        self.samples = sorted(self.img_dir.glob("*.tif"))
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No .tif found in {self.img_dir}")

    def __len__(self):
        return len(self.samples)

    def _crop(self, img, mask):
        # img: (C,H,W), mask: (H,W)
        h, w = img.shape[1], img.shape[2]
        crop = self.img_size

        if h <= crop or w <= crop:
            # If smaller than crop, take top-left (or you can pad)
            return img[:, :crop, :crop], mask[:crop, :crop]

        if self.is_train:
            top = np.random.randint(0, h - crop + 1)
            left = np.random.randint(0, w - crop + 1)
        else:
            # deterministic center crop for val/test
            top = (h - crop) // 2
            left = (w - crop) // 2

        img = img[:, top : top + crop, left : left + crop]
        mask = mask[top : top + crop, left : left + crop]
        return img, mask

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        mask_path = self.mask_dir / img_path.name

        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)  # (C,H,W)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)  # (H,W)

        img = np.nan_to_num(img, nan=0.0)

        img, mask = self._crop(img, mask)

        img = torch.from_numpy(img)   # (C,H,W)
        mask = torch.from_numpy(mask) # (H,W)

        # PANGAEA expects (T,C,H,W) per modality; set T=1
        return {
            "image": {"sar": img.unsqueeze(1)},  # (1,C,H,W)
            "target": mask,
            "metadata": {},
        }
