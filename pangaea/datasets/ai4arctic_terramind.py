import numpy as np
import rasterio
import torch
from pathlib import Path
from torch.utils.data import Dataset


class AI4ArcticTerraMindDataset(Dataset):
    """
    TerraMind-only AI4Arctic loader.

    Key difference vs your standard loader:
      - Adds an empty "optical" modality so BandFilter/BandPadding won't KeyError.
      - Does NOT change your SAR layout convention (keeps (C, T, H, W) with T=1).
        This avoids impacting the other encoders that already work for you.
    """

    def __init__(self, root_path, split, img_size=512, ignore_index=255, is_train=False, **kwargs):
        self.root = Path(root_path)
        self.split = split
        self.img_size = int(img_size)
        self.ignore_index = int(ignore_index)
        self.is_train = bool(is_train)

        # Common metadata fields expected by evaluators
        self.dataset_name = kwargs.get("dataset_name", "ai4arctic")
        self.classes = kwargs.get(
            "classes",
            ["open_water", "new_ice", "young_ice", "thin_first_year_ice", "thick_first_year_ice", "old_ice"],
        )
        self.num_classes = int(kwargs.get("num_classes", len(self.classes)))
        self.bands = kwargs.get("bands", {"sar": ["VV", "VH"], "optical": []})

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
            return img[:, :crop, :crop], mask[:crop, :crop], 0, 0

        if self.is_train:
            top = np.random.randint(0, h - crop + 1)
            left = np.random.randint(0, w - crop + 1)
        else:
            top = (h - crop) // 2
            left = (w - crop) // 2

        img = img[:, top: top + crop, left: left + crop]
        mask = mask[top: top + crop, left: left + crop]
        return img, mask, int(top), int(left)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        mask_path = self.mask_dir / img_path.name

        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)  # (C,H,W)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)  # (H,W)

        img = np.nan_to_num(img, nan=0.0)
        img, mask, top, left = self._crop(img, mask)

        img_t = torch.from_numpy(img)   # (C,H,W)
        mask_t = torch.from_numpy(mask) # (H,W)

        H, W = img_t.shape[1], img_t.shape[2]

        # KEEP your existing convention used by other working runs:
        # (C, T, H, W) with T=1
        sar = img_t.unsqueeze(1)  # (C,1,H,W)

        # Add empty optical modality (0 channels) in same convention
        optical_empty = torch.zeros((0, 1, H, W), dtype=sar.dtype)

        return {
            "image": {
                "sar": sar,
                "optical": optical_empty,
            },
            "target": mask_t,
            "metadata": {
                "filename": img_path.name,
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "crop_top": top,
                "crop_left": left,
            },
        }
