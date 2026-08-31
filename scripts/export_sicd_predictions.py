#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pangaea.engine.data_preprocessor import Preprocessor
from pangaea.engine.evaluator import Evaluator
from pangaea.utils.collate_fn import get_collate_fn


AI4ARCTIC_PALETTE = [
    (0, 0, 0),
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
]


class PreprocessedDataset(Dataset):
    def __init__(self, raw_dataset: Dataset, preprocessor: Preprocessor | None, max_samples: int | None = None) -> None:
        self.raw_dataset = raw_dataset
        self.preprocessor = preprocessor
        self.max_samples = max_samples

        for attr in ["classes", "ignore_index", "num_classes", "split", "dataset_name"]:
            if hasattr(raw_dataset, attr):
                setattr(self, attr, getattr(raw_dataset, attr))

    def __len__(self) -> int:
        raw_len = len(self.raw_dataset)
        return min(raw_len, self.max_samples) if self.max_samples is not None else raw_len

    def __getitem__(self, idx: int):
        sample = self.raw_dataset[idx]
        if self.preprocessor is not None:
            sample = self.preprocessor(sample)
        return sample


def flatten_palette(palette):
    flat = []
    for rgb in palette:
        flat.extend(rgb)
    flat.extend([0] * (768 - len(flat)))
    return flat[:768]


def save_mask(mask: np.ndarray, index_path: Path, color_path: Path) -> None:
    indexed = Image.fromarray(mask.astype(np.uint8), mode="P")
    indexed.putpalette(flatten_palette(AI4ARCTIC_PALETTE))
    indexed.save(index_path)

    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for idx, rgb in enumerate(AI4ARCTIC_PALETTE):
        color[mask == idx] = rgb
    Image.fromarray(color, mode="RGB").save(color_path)


def inspect_sicd_channels(dataset_root: Path) -> int:
    sample = sorted((dataset_root / "test" / "images").glob("*.tif"))[0]
    with rasterio.open(sample) as src:
        return int(src.count)


def load_cfg(run_dir: Path, dataset_root: Path):
    cfg = OmegaConf.load(run_dir / "configs" / "config.yaml")
    cfg.dataset.root_path = str(dataset_root)
    OmegaConf.resolve(cfg)
    return cfg


def is_compatible(cfg, available_channels: int) -> tuple[bool, str]:
    input_bands = OmegaConf.to_container(cfg.encoder.get("input_bands"), resolve=True)
    if not input_bands:
        return False, "missing encoder.input_bands"

    optical = input_bands.get("optical", [])
    sar = input_bands.get("sar", [])

    if optical and len(optical) > 3:
        return False, f"requires {len(optical)} optical bands, SICD currently exposes 3 channels"
    if sar and len(sar) > 2:
        return False, f"requires {len(sar)} sar bands, exporter only supports 2-channel SICD SAR runs"
    if optical and sar:
        if len(optical) != 0:
            return False, "mixed optical+sar run not supported for SICD export"
    if not optical and not sar:
        return False, "no supported optical/sar input bands declared"
    if available_channels < max(len(optical), len(sar)):
        return False, f"dataset has {available_channels} channels, run needs {max(len(optical), len(sar))}"

    return True, "ok"


def build_model(cfg, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    encoder = instantiate(cfg.encoder)
    decoder = instantiate(cfg.decoder, encoder=encoder)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    decoder.load_state_dict(state_dict)
    decoder.to(device)
    decoder.eval()
    return decoder


def infer_mask(model, batch, cfg, device: torch.device) -> np.ndarray:
    image = {k: v.to(device) for k, v in batch["image"].items()}
    target = batch["target"].to(device)
    inference_mode = str(cfg.task.evaluator.inference_mode)
    max_batch = cfg.task.evaluator.get("sliding_inference_batch")

    with torch.no_grad():
        if inference_mode == "sliding":
            logits = Evaluator.sliding_inference(
                model,
                image,
                model.encoder.input_size,
                output_shape=target.shape[-2:],
                max_batch=max_batch,
            )
        elif inference_mode == "whole":
            logits = model(image, output_shape=target.shape[-2:])
        else:
            raise ValueError(f"Unsupported inference mode: {inference_mode}")

    if logits.shape[1] == 1:
        pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
    else:
        pred = torch.argmax(logits, dim=1)
    return pred[0].detach().cpu().numpy().astype(np.uint8)


def export_run(
    run_dir: Path,
    dataset_root: Path,
    output_root: Path,
    device: torch.device,
    max_samples: int | None,
) -> dict:
    cfg = load_cfg(run_dir, dataset_root)
    compatible, reason = is_compatible(cfg, inspect_sicd_channels(dataset_root))
    run_name = run_dir.name
    model_name = f"{cfg.encoder._target_.split('.')[-1]}__{run_name}"
    run_output_dir = output_root / model_name
    if not compatible:
        return {
            "run_dir": str(run_dir),
            "model_name": model_name,
            "saved": 0,
            "skipped": True,
            "reason": reason,
        }

    raw_dataset = instantiate(cfg.dataset, split="test")
    preprocessor: Preprocessor = instantiate(
        cfg.preprocessing.test,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )
    dataset = PreprocessedDataset(raw_dataset, preprocessor, max_samples=max_samples)

    ckpt_path = run_dir / "checkpoint__best.pth"
    model = build_model(cfg, ckpt_path, device)

    index_dir = run_output_dir / "index"
    color_dir = run_output_dir / "color"
    index_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=get_collate_fn(list(cfg.encoder.input_bands.keys())),
    )

    saved = 0
    for batch in tqdm(loader, desc=run_name, leave=False):
        metadata = batch["metadata"][0]
        stem = Path(metadata["filename"]).stem
        mask = infer_mask(model, batch, cfg, device)
        save_mask(mask, index_dir / f"{stem}.png", color_dir / f"{stem}.png")
        saved += 1

    manifest = {
        "run_dir": str(run_dir),
        "model_name": model_name,
        "saved": saved,
        "skipped": False,
        "reason": "ok",
        "checkpoint": str(ckpt_path),
        "dataset_target": str(cfg.dataset._target_),
        "encoder_target": str(cfg.encoder._target_),
        "input_bands": OmegaConf.to_container(cfg.encoder.input_bands, resolve=True),
    }
    with open(run_output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--match", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_dirs = sorted(
        p for p in args.runs_root.iterdir()
        if p.is_dir() and (p / "configs" / "config.yaml").exists() and (p / "checkpoint__best.pth").exists()
    )
    if args.match:
        run_dirs = [run for run in run_dirs if args.match in run.name]

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for run_dir in tqdm(run_dirs, desc="Runs"):
        manifests.append(
            export_run(
                run_dir,
                args.dataset_root,
                args.output_root,
                torch.device(args.device),
                args.max_samples,
            )
        )

    with open(args.output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"runs": manifests, "total_runs": len(manifests)}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
