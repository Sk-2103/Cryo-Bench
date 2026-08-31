import os

import numpy as np
import torch
from PIL import Image

from pangaea.datasets.base import RawGeoFMDataset


class ShelfBench(RawGeoFMDataset):
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
        super(ShelfBench, self).__init__(
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

        self.image_list = [
            entry.path
            for entry in os.scandir(image_root)
            if entry.is_file() and entry.name.endswith(".png")
        ]
        if not self.image_list:
            raise FileNotFoundError(f"No PNG images found in {image_root}")

        self.target_list = []
        for image_path in self.image_list:
            mask_path = os.path.join(mask_root, os.path.basename(image_path))
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Missing mask for {image_path}. Expected {mask_path}"
                )
            self.target_list.append(mask_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.image_list[index]).convert("L"), dtype=np.float32)
        image = image / 255.0
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(1)

        target = np.asarray(Image.open(self.target_list[index]).convert("L"), dtype=np.uint8)
        target = torch.from_numpy((target > 0).astype(np.int64)).long()

        return {
            "image": {
                "optical": image.clone(),
                "sar": image,
            },
            "target": target,
            "metadata": {
                "image_path": self.image_list[index],
                "target_path": self.target_list[index],
            },
        }

    @staticmethod
    def download(self, silent=False):
        if not os.path.exists(self.root_path):
            raise FileNotFoundError(f"{self.root_path} does not exist.")
