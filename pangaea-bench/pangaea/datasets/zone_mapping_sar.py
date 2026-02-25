import os
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from glob import glob
from pangaea.datasets.base import RawGeoFMDataset


class ZONE(RawGeoFMDataset):
    """
    ZONE dataset rewritten with:
      - TRAIN: random crop (image + mask together), returning fixed square tiles
      - VAL/TEST: deterministic tiling (sliding tiles), returning fixed square tiles

    This is designed to avoid evaluator's sliding-window stitching issues by ensuring
    the evaluator always receives square inputs of size img_size x img_size.

    IMPORTANT for evaluation:
      - Use stride == tile_size (non-overlap) to avoid double-counting pixels in metrics.
      - Set evaluator inference_mode="whole" (recommended) since dataset already returns tiles.
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
        download_url: str,
        auto_download: bool,
        stride: int | None = None,       # used for VAL/TEST tiling; default = img_size (non-overlap)
        train_crops_per_image: int = 1,  # how many random crops per big image per epoch (train only)
        pad_mode: str = "constant",      # padding mode if an image is smaller than tile size
        pad_value: float = 0.0,
    ):
        super(ZONE, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
        )

        self.split = split
        self.tile_size = int(img_size)
        self.stride = int(stride) if stride is not None else int(img_size)
        self.train_crops_per_image = int(train_crops_per_image)
        self.pad_mode = pad_mode
        self.pad_value = float(pad_value)

        # Paths
        self.image_dir = os.path.join(root_path, "images", split)
        self.mask_dir = os.path.join(root_path, "labels", split)

        self.image_list = sorted(glob(os.path.join(self.image_dir, "*.png")))
        self.mask_list = sorted(glob(os.path.join(self.mask_dir, "*.png")))
        assert len(self.image_list) == len(self.mask_list), "Number of images and masks must match!"

        # Mode selection
        self.mode = "random_crop" if split == "train" else "tiles"

        # Precompute tiles for val/test
        self.tiles = []
        if self.mode == "tiles":
            for img_idx, (img_path, msk_path) in enumerate(zip(self.image_list, self.mask_list)):
                with rasterio.open(img_path) as src_img, rasterio.open(msk_path) as src_msk:
                    H, W = src_img.height, src_img.width
                    if (H != src_msk.height) or (W != src_msk.width):
                        raise ValueError(
                            f"Image/mask size mismatch:\n"
                            f"  image: {img_path} ({W}x{H})\n"
                            f"  mask : {msk_path} ({src_msk.width}x{src_msk.height})"
                        )

                    ts = self.tile_size
                    st = self.stride

                    # If image is smaller than tile, we will return a single tile at (0,0) and pad later
                    if H <= ts and W <= ts:
                        self.tiles.append((img_idx, 0, 0))
                        continue

                    # Generate offsets (edge-aligned coverage)
                    row_offsets = list(range(0, max(H - ts + 1, 1), st))
                    col_offsets = list(range(0, max(W - ts + 1, 1), st))

                    # Ensure last tile hits bottom/right edge (so full coverage)
                    if H > ts:
                        last_r = H - ts
                        if len(row_offsets) == 0:
                            row_offsets = [0]
                        if row_offsets[-1] != last_r:
                            row_offsets.append(last_r)
                    else:
                        row_offsets = [0]

                    if W > ts:
                        last_c = W - ts
                        if len(col_offsets) == 0:
                            col_offsets = [0]
                        if col_offsets[-1] != last_c:
                            col_offsets.append(last_c)
                    else:
                        col_offsets = [0]

                    for r in row_offsets:
                        for c in col_offsets:
                            self.tiles.append((img_idx, r, c))

    def __len__(self):
        if self.mode == "random_crop":
            # present each big image multiple times per epoch with different random crops
            return len(self.image_list) * self.train_crops_per_image
        return len(self.tiles)

    def _pad_to_tile(self, img_chw: torch.Tensor, mask_hw: torch.Tensor):
        """
        img_chw: (C, h, w)
        mask_hw: (h, w)
        returns padded to (C, ts, ts) and (ts, ts)
        """
        ts = self.tile_size
        _, h, w = img_chw.shape
        pad_h = max(ts - h, 0)
        pad_w = max(ts - w, 0)
        if pad_h == 0 and pad_w == 0:
            return img_chw, mask_hw

        # pad format: (left, right, top, bottom)
        img_chw = F.pad(img_chw, (0, pad_w, 0, pad_h), mode=self.pad_mode, value=self.pad_value)
        mask_hw = F.pad(mask_hw, (0, pad_w, 0, pad_h), mode="constant", value=int(self.ignore_index))
        return img_chw, mask_hw

    def __getitem__(self, index):
        ts = self.tile_size

        if self.mode == "random_crop":
            # Map "index" to a big image index
            img_idx = index // self.train_crops_per_image
            img_path = self.image_list[img_idx]
            mask_path = self.mask_list[img_idx]

            with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_msk:
                H, W = src_img.height, src_img.width
                if (H != src_msk.height) or (W != src_msk.width):
                    raise ValueError(
                        f"Image/mask size mismatch:\n"
                        f"  image: {img_path} ({W}x{H})\n"
                        f"  mask : {mask_path} ({src_msk.width}x{src_msk.height})"
                    )

                crop_h = min(ts, H)
                crop_w = min(ts, W)

                # random top-left (valid)
                top = np.random.randint(0, H - crop_h + 1) if H > crop_h else 0
                left = np.random.randint(0, W - crop_w + 1) if W > crop_w else 0

                window = Window(col_off=left, row_off=top, width=crop_w, height=crop_h)

                img = src_img.read(window=window).astype(np.float32)     # (C, crop_h, crop_w)
                mask = src_msk.read(1, window=window).astype(np.int64)   # (crop_h, crop_w)

            img = torch.from_numpy(img).float()
            target = torch.from_numpy(mask).long()

            # pad if needed (when big image smaller than tile)
            img, target = self._pad_to_tile(img, target)

            # add temporal dimension T=1 -> (C, 1, ts, ts)
            img = img.unsqueeze(1)

            return {
                "image": {"sar": img},
                "target": target,
                "metadata": {},
            }

        # VAL/TEST: deterministic tiling
        img_idx, row_off, col_off = self.tiles[index]
        img_path = self.image_list[img_idx]
        mask_path = self.mask_list[img_idx]

        with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_msk:
            H, W = src_img.height, src_img.width
            if (H != src_msk.height) or (W != src_msk.width):
                raise ValueError(
                    f"Image/mask size mismatch:\n"
                    f"  image: {img_path} ({W}x{H})\n"
                    f"  mask : {mask_path} ({src_msk.width}x{src_msk.height})"
                )

            # When image smaller than tile, read what exists and pad
            crop_h = min(ts, H - row_off) if H > row_off else 0
            crop_w = min(ts, W - col_off) if W > col_off else 0
            crop_h = max(crop_h, 0)
            crop_w = max(crop_w, 0)

            # If something odd happens, fall back to 0,0
            if crop_h == 0 or crop_w == 0:
                row_off, col_off = 0, 0
                crop_h = min(ts, H)
                crop_w = min(ts, W)

            window = Window(col_off=col_off, row_off=row_off, width=crop_w, height=crop_h)

            img = src_img.read(window=window).astype(np.float32)     # (C, crop_h, crop_w)
            mask = src_msk.read(1, window=window).astype(np.int64)   # (crop_h, crop_w)

        img = torch.from_numpy(img).float()
        target = torch.from_numpy(mask).long()

        # pad to tile size if crop touches image boundary or image smaller than tile
        img, target = self._pad_to_tile(img, target)

        img = img.unsqueeze(1)  # (C, 1, ts, ts)

        return {
            "image": {"sar": img},
            "target": target,
            "metadata": {},
        }

    @staticmethod
    def download(self, silent=False):
        pass