import os
from glob import glob

import numpy as np
import rasterio
import torch

from pangaea.datasets.base import RawGeoFMDataset


class GLID(RawGeoFMDataset):
    """
    Minimal single-temporal RGB semantic segmentation dataset.
    Expects:
      root_path/{train,val,test}/images/*
      root_path/{train,val,test}/masks/*
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
        bands: dict,
        distribution: list,
        data_mean: dict,
        data_std: dict,
        data_min: dict,
        data_max: dict,
        download_url: str,
        auto_download: bool,
    ):
        super().__init__(
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
        self.root_path = root_path

        img_dir = os.path.join(root_path, split, "images")
        msk_dir = os.path.join(root_path, split, "masks")

        self.images = sorted(glob(os.path.join(img_dir, "*")))
        self.masks  = sorted(glob(os.path.join(msk_dir, "*")))

        # basic sanity: assume same stems
        assert len(self.images) == len(self.masks), "images/masks count mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        msk_path = self.masks[idx]

        with rasterio.open(img_path) as src:
            img = src.read()  # (C,H,W) for tif; (H,W,3) won't happen for rasterio
        if img.ndim == 2:
            img = img[None, ...]
        # keep only first 3 bands for safety
        img = img[:3, :, :]
        img = torch.from_numpy(img).float()

        with rasterio.open(msk_path) as src:
            mask = src.read(1)
        mask = torch.from_numpy(mask.astype(np.int64))

        # Optional: if masks are {0,255} convert to {0,1}
        # mask = (mask > 0).long()

        # PANGAEA expects (C,T,H,W) for each modality, even for single-temporal datasets
        # (as seen in other dataset implementations)
        img = img.unsqueeze(1)  # -> (C,1,H,W)

        return {
            "image": {"optical": img},   # RGB is treated as "optical"
            "target": mask,
            "metadata": {},
        }

    @staticmethod
    def download(self, silent=False):
        raise NotImplementedError("GLID is a local dataset; no auto-download.")
