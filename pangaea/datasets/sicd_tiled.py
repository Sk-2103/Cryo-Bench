import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from glob import glob
from pangaea.datasets.base import RawGeoFMDataset


class SICDTiled(RawGeoFMDataset):
    """SICD / AI4Arctic with full-coverage evaluation.

    The original `ai4arctic.AI4ArcticDataset` takes a single centre crop of every val/test
    scene, so with img_size=512 on ~5200x5000 scenes only ~1% of the test pixels were ever
    scored. This class follows the same approach as the CaFFe `ZONE` dataset:

      TRAIN    random crop of size img_size (matching the previous behaviour)
      VAL/TEST deterministic edge-aligned tiling covering the whole scene

    so every test pixel is predicted, which is what the Methods section claims.

    `modality` selects which key the image is returned under ("sar" or "optical"); the
    number of channels taken is len(bands[modality]), read from the front of the file.
    SICD scenes are 3-band; the SAR view uses VV,VH and the RGB-proxy view uses all three.
    """

    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[float]],
        data_std: dict[str, list[float]],
        data_min: dict[str, list[float]],
        data_max: dict[str, list[float]],
        # defaulted: configs saved by the old ai4arctic runs omit these keys
        download_url: str = None,
        auto_download: bool = False,
        modality: str = "sar",
        stride: int | None = None,
        pad_value: float = 0.0,
    ):
        super(SICDTiled, self).__init__(
            split=split, dataset_name=dataset_name, multi_modal=multi_modal,
            multi_temporal=multi_temporal, root_path=root_path, classes=classes,
            num_classes=num_classes, ignore_index=ignore_index, img_size=img_size,
            bands=bands, distribution=distribution, data_mean=data_mean,
            data_std=data_std, data_min=data_min, data_max=data_max,
            download_url=download_url, auto_download=auto_download,
        )

        self.split = split
        self.tile_size = int(img_size)
        self.stride = int(stride) if stride is not None else int(img_size)
        self.pad_value = float(pad_value)
        self.modality = modality
        self.n_channels = len(bands[modality])

        self.image_dir = os.path.join(root_path, split, "images")
        self.mask_dir = os.path.join(root_path, split, "masks")
        self.image_list = sorted(glob(os.path.join(self.image_dir, "*.tif")))
        self.mask_list = sorted(glob(os.path.join(self.mask_dir, "*.tif")))
        assert len(self.image_list) == len(self.mask_list), "images and masks must match"
        assert len(self.image_list) > 0, f"no .tif found under {self.image_dir}"

        self.mode = "random_crop" if split == "train" else "tiles"

        self.tiles = []
        if self.mode == "tiles":
            ts, st = self.tile_size, self.stride
            for img_idx, img_path in enumerate(self.image_list):
                with rasterio.open(img_path) as src:
                    H, W = src.height, src.width
                if H <= ts and W <= ts:
                    self.tiles.append((img_idx, 0, 0))
                    continue
                rows = list(range(0, max(H - ts + 1, 1), st))
                cols = list(range(0, max(W - ts + 1, 1), st))
                if H > ts and rows[-1] != H - ts:
                    rows.append(H - ts)
                if W > ts and cols[-1] != W - ts:
                    cols.append(W - ts)
                for r in rows:
                    for c in cols:
                        self.tiles.append((img_idx, r, c))

    def __len__(self):
        return len(self.image_list) if self.mode == "random_crop" else len(self.tiles)

    def _pad(self, img, mask):
        ts = self.tile_size
        _, h, w = img.shape
        ph, pw = max(ts - h, 0), max(ts - w, 0)
        if ph == 0 and pw == 0:
            return img, mask
        img = F.pad(img, (0, pw, 0, ph), mode="constant", value=self.pad_value)
        mask = F.pad(mask, (0, pw, 0, ph), mode="constant", value=int(self.ignore_index))
        return img, mask

    def _read(self, img_path, mask_path, row_off, col_off, h, w):
        window = Window(col_off=col_off, row_off=row_off, width=w, height=h)
        with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_msk:
            img = src_img.read(window=window).astype(np.float32)
            mask = src_msk.read(1, window=window).astype(np.int64)
        img = img[: self.n_channels]
        img = np.nan_to_num(img, nan=self.pad_value, posinf=self.pad_value,
                            neginf=self.pad_value)
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()

    def __getitem__(self, index):
        ts = self.tile_size
        if self.mode == "random_crop":
            img_path, mask_path = self.image_list[index], self.mask_list[index]
            with rasterio.open(img_path) as src:
                H, W = src.height, src.width
            ch, cw = min(ts, H), min(ts, W)
            top = np.random.randint(0, H - ch + 1) if H > ch else 0
            left = np.random.randint(0, W - cw + 1) if W > cw else 0
            img, target = self._read(img_path, mask_path, top, left, ch, cw)
        else:
            img_idx, top, left = self.tiles[index]
            img_path, mask_path = self.image_list[img_idx], self.mask_list[img_idx]
            with rasterio.open(img_path) as src:
                H, W = src.height, src.width
            ch, cw = min(ts, H - top), min(ts, W - left)
            img, target = self._read(img_path, mask_path, top, left, ch, cw)

        img, target = self._pad(img, target)
        img = img.unsqueeze(1)  # (C, T=1, H, W)
        return {"image": {self.modality: img}, "target": target, "metadata": {}}

    @staticmethod
    def download(self, silent=False):
        pass
