import os
import torch
import rasterio
from glob import glob
from pangaea.datasets.base import RawGeoFMDataset
import numpy as np

class GSSD(RawGeoFMDataset):

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
        #temp: int,
    ):

        super(GSSD, self).__init__(
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

        #self.temp = temp

        # ✅ Define paths
        self.image_dir = os.path.join(root_path, split, "image")
        self.mask_dir = os.path.join(root_path, split, "mask")

        # ✅ Collect all tif files
        self.image_list = sorted(glob(os.path.join(self.image_dir, "*.tif")))
        self.mask_list = sorted(glob(os.path.join(self.mask_dir, "*.tif")))

        assert len(self.image_list) == len(self.mask_list), \
            "Number of images and masks must match!"

        print(f"Loaded {len(self.image_list)} samples for split = {split}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        # -------------------------------------------------
        # ✅ Load Image
        # -------------------------------------------------
        img_path = self.image_list[index]

        with rasterio.open(img_path) as src:
            img = src.read()   # shape = (C, H, W)
        
        
        img = torch.tensor(img, dtype=torch.float32)

       

        # Add temporal dimension (T=1)
        img= img.unsqueeze(1)   # (C, T, H, W)
        

        # -------------------------------------------------
        # ✅ Load Mask
        # -------------------------------------------------
        mask_path = self.mask_list[index]

        with rasterio.open(mask_path) as src:
            mask = src.read(1)   # (H, W)

        target = torch.tensor(mask, dtype=torch.long)

        # -------------------------------------------------
        # ✅ Return Format Required by Pangaea
        # -------------------------------------------------
        return {
            "image": {
                "optical": img
            },
            "target": target,
            "metadata": {}
        }

    @staticmethod
    def download(self, silent=False):
        pass

