import os
from glob import glob

import numpy as np
import rasterio
import torch

from pangaea.datasets.base import RawGeoFMDataset


class GLBRandom(RawGeoFMDataset):
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
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
        image_dirname: str = "images",
        mask_dirname: str = "masks",
    ):
        super(GLBRandom, self).__init__(
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

        split_root = os.path.join(self.root_path, split)
        image_root = os.path.join(split_root, image_dirname)
        mask_root = os.path.join(split_root, mask_dirname)

        self.image_list = sorted(glob(os.path.join(image_root, "*.tif")))
        if not self.image_list:
            raise FileNotFoundError(f"No TIFF images found in {image_root}")

        self.target_list = []
        for image_path in self.image_list:
            mask_path = os.path.join(mask_root, os.path.basename(image_path))
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Missing mask for {image_path}. Expected {mask_path}"
                )
            self.target_list.append(mask_path)

        self.optical_count = len(self.bands.get("optical", []))
        self.sar_count = len(self.bands.get("sar", []))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        with rasterio.open(self.image_list[index]) as src:
            image = src.read().astype(np.float32)

        expected_count = self.optical_count + self.sar_count
        if image.shape[0] != expected_count:
            raise ValueError(
                f"{self.image_list[index]} has {image.shape[0]} bands; "
                f"expected {expected_count}"
            )

        with rasterio.open(self.target_list[index]) as src:
            target = src.read(1).astype(np.int64)

        optical = torch.from_numpy(image[: self.optical_count])
        sar = torch.from_numpy(image[self.optical_count :])

        output = {
            "image": {
                "optical": optical.unsqueeze(1),
                "sar": sar.unsqueeze(1),
            },
            "target": torch.from_numpy(target).long(),
            "metadata": {
                "image_path": self.image_list[index],
                "target_path": self.target_list[index],
            },
        }
        return output

    @staticmethod
    def download(self, silent=False):
        if not os.path.exists(self.root_path):
            raise FileNotFoundError(f"{self.root_path} does not exist.")
