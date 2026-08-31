#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import rasterio
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from pangaea.datasets.base import GeoFMDataset
from pangaea.engine.evaluator import Evaluator
from pangaea.utils.collate_fn import get_collate_fn
from pangaea.utils.utils import get_best_model_ckpt_path, get_final_model_ckpt_path


DEFAULT_CKPT_DIR = Path(
    "/media/turtle-ssd/users/skaushik/Flood_Data_compilation/"
    "psfloods_rgvt_ufo_htx_val/output/"
    "20260415_104205_f33977_prithvi_eo_v2_300_seg_upernet_psfloods_rgvt"
)


def load_model(cfg, ckpt_dir: Path, device: torch.device, use_final: bool):
    encoder = instantiate(cfg.encoder)
    encoder.load_encoder_weights(logger=logging.getLogger())
    decoder = instantiate(cfg.decoder, encoder=encoder)
    decoder.to(device)

    ckpt_path = (
        get_final_model_ckpt_path(ckpt_dir)
        if use_final
        else get_best_model_ckpt_path(ckpt_dir)
    )
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in state:
        state = state["model"]
    decoder.load_state_dict(state)
    decoder.eval()
    return decoder, Path(ckpt_path)


def build_loader(cfg, split: str, batch_size: int, num_workers: int):
    preprocessor = instantiate(
        cfg.preprocessing.test,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )
    raw_dataset = instantiate(cfg.dataset, split=split)
    dataset = GeoFMDataset(raw_dataset, preprocessor)
    modalities = list(cfg.encoder.input_bands.keys())
    collate_fn = get_collate_fn(modalities)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return loader


def save_prediction(mask: torch.Tensor, src_path: Path, dst_path: Path) -> None:
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=255)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(mask.detach().cpu().numpy().astype("uint8"), 1)


def save_probability(prob: torch.Tensor, src_path: Path, dst_path: Path) -> None:
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="float32", nodata=None)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(prob.detach().cpu().numpy().astype("float32"), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-subdir", default="predictions_test_best")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-final", action="store_true")
    parser.add_argument("--save-probability", action="store_true")
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir.resolve()
    cfg = OmegaConf.load(ckpt_dir / "configs" / "config.yaml")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loader = build_loader(cfg, args.split, args.batch_size, args.num_workers)
    model, ckpt_path = load_model(cfg, ckpt_dir, device, args.use_final)

    output_root = ckpt_dir / args.output_subdir
    mask_root = output_root / "mask"
    prob_root = output_root / "probability"

    inference_mode = cfg.task.evaluator.inference_mode
    sliding_batch = cfg.task.evaluator.sliding_inference_batch

    with torch.no_grad():
        for data in tqdm(loader, desc=f"Inference on {args.split}"):
            image = {k: v.to(device) for k, v in data["image"].items()}
            metadata = data["metadata"]

            if inference_mode == "sliding":
                target_shape = tuple(data["target"].shape[-2:])
                logits = Evaluator.sliding_inference(
                    model,
                    image,
                    model.encoder.input_size,
                    output_shape=target_shape,
                    max_batch=sliding_batch,
                )
            elif inference_mode == "whole":
                logits = model(image, output_shape=data["target"].shape[-2:])
            else:
                raise NotImplementedError(f"Unsupported inference mode: {inference_mode}")

            if logits.shape[1] == 1:
                water_prob = torch.sigmoid(logits).squeeze(1)
                pred = (water_prob > 0.5).to(torch.uint8)
            else:
                probs = torch.softmax(logits, dim=1)
                water_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                pred = torch.argmax(logits, dim=1).to(torch.uint8)

            for i, sample_meta in enumerate(metadata):
                src_path = Path(sample_meta["image_path"])
                stem = src_path.name
                save_prediction(pred[i], src_path, mask_root / stem)
                if args.save_probability:
                    save_probability(water_prob[i], src_path, prob_root / stem)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Saved masks to: {mask_root}")
    if args.save_probability:
        print(f"Saved probabilities to: {prob_root}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
