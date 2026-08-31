#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

GLID_PALETTE = [
    (0, 0, 0),
    (220, 20, 60),
]

DEFAULT_PALETTE = [
    (0, 0, 0),
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


@dataclass
class SampleRecord:
    image_path: Path
    mask_path: Path


class FullSceneSegmentationDataset(Dataset):
    def __init__(
        self,
        cfg,
        split: str,
        dataset_root: Path,
        preprocessor: Preprocessor | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.dataset_root = dataset_root
        self.preprocessor = preprocessor
        self.max_samples = max_samples
        self.image_dir = dataset_root / split / "images"
        self.mask_dir = dataset_root / split / "masks"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing mask directory: {self.mask_dir}")

        self.samples = self._discover_samples()
        if not self.samples:
            raise FileNotFoundError(f"No test samples found in {self.image_dir}")
        if self.max_samples is not None:
            self.samples = self.samples[: self.max_samples]

        dataset_cfg = cfg.dataset
        self.dataset_name = str(dataset_cfg.get("dataset_name", dataset_root.name))
        self.classes = list(dataset_cfg.classes)
        self.num_classes = int(dataset_cfg.num_classes)
        self.ignore_index = int(dataset_cfg.ignore_index)

        self.expected_modalities = {
            key: list(value) for key, value in cfg.encoder.input_bands.items()
        }
        self.palette = palette_for_dataset(self.dataset_name, self.num_classes)

    def _discover_samples(self) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        for image_path in sorted(self.image_dir.iterdir()):
            if not image_path.is_file():
                continue
            mask_path = self.mask_dir / image_path.name
            if not mask_path.exists():
                continue
            records.append(SampleRecord(image_path=image_path, mask_path=mask_path))
        return records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        image = self._load_image(sample.image_path)
        target = self._load_mask(sample.mask_path)
        output = {
            "image": image,
            "target": target,
            "metadata": {
            "filename": sample.image_path.name,
            "stem": sample.image_path.stem,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
            },
        }
        if self.preprocessor is not None:
            output = self.preprocessor(output)
        return output

    def _load_image(self, image_path: Path) -> dict[str, torch.Tensor]:
        suffix = image_path.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            with rasterio.open(image_path) as src:
                arr = src.read().astype(np.float32)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            arr = np.array(Image.open(image_path))
            if arr.ndim == 2:
                arr = arr[..., None]
            arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported image format: {image_path}")

        arr = np.nan_to_num(arr, nan=0.0)
        tensors: dict[str, torch.Tensor] = {}
        channel_count = arr.shape[0]

        for modality, bands in self.expected_modalities.items():
            if len(bands) == 0:
                tensors[modality] = torch.zeros(
                    (0, 1, arr.shape[1], arr.shape[2]), dtype=torch.float32
                )
                continue

            if modality == "sar":
                if channel_count < len(bands):
                    raise ValueError(
                        f"{image_path.name} has {channel_count} channels, "
                        f"but run expects {len(bands)} SAR channels."
                    )
                modality_arr = arr[: len(bands)]
            elif modality == "optical":
                if channel_count < len(bands):
                    raise ValueError(
                        f"{image_path.name} has {channel_count} channels, "
                        f"but run expects {len(bands)} optical channels."
                    )
                modality_arr = arr[: len(bands)]
            else:
                raise ValueError(f"Unsupported modality '{modality}' in run config.")

            tensors[modality] = torch.from_numpy(modality_arr).float().unsqueeze(1)

        return tensors

    def _load_mask(self, mask_path: Path) -> torch.Tensor:
        suffix = mask_path.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]
        else:
            raise ValueError(f"Unsupported mask format: {mask_path}")
        return torch.from_numpy(mask.astype(np.int64))


def palette_for_dataset(dataset_name: str, num_classes: int) -> list[tuple[int, int, int]]:
    name = dataset_name.lower()
    if "ai4arctic" in name or "sicd" in name:
        palette = AI4ARCTIC_PALETTE
    elif "glid" in name:
        palette = GLID_PALETTE
    else:
        palette = DEFAULT_PALETTE

    if len(palette) >= num_classes:
        return palette[:num_classes]

    extended = list(palette)
    while len(extended) < num_classes:
        base = DEFAULT_PALETTE[len(extended) % len(DEFAULT_PALETTE)]
        extended.append(base)
    return extended


def flatten_palette(palette: list[tuple[int, int, int]]) -> list[int]:
    flat: list[int] = []
    for rgb in palette:
        flat.extend(rgb)
    flat.extend([0] * (768 - len(flat)))
    return flat[:768]


def save_palette_png(mask: np.ndarray, out_path: Path, palette: list[tuple[int, int, int]]) -> None:
    image = Image.fromarray(mask.astype(np.uint8), mode="P")
    image.putpalette(flatten_palette(palette))
    image.save(out_path)


def save_color_png(mask: np.ndarray, out_path: Path, palette: list[tuple[int, int, int]]) -> None:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for idx, rgb in enumerate(palette):
        color[mask == idx] = rgb
    Image.fromarray(color, mode="RGB").save(out_path)


def find_run_dirs(runs_root: Path, ckpt_name: str, top_level_only: bool) -> list[Path]:
    run_dirs: list[Path] = []
    if top_level_only:
        for child in sorted(runs_root.iterdir()):
            if not child.is_dir():
                continue
            config_path = child / "configs" / "config.yaml"
            if config_path.exists() and (child / ckpt_name).exists():
                run_dirs.append(child)
    else:
        for config_path in runs_root.rglob("configs/config.yaml"):
            run_dir = config_path.parent.parent
            if (run_dir / ckpt_name).exists():
                run_dirs.append(run_dir)
    return sorted(run_dirs)


def load_run_config(run_dir: Path, dataset_root: Path | None) -> object:
    cfg = OmegaConf.load(run_dir / "configs" / "config.yaml")
    if dataset_root is not None:
        cfg.dataset.root_path = str(dataset_root)
    OmegaConf.resolve(cfg)
    return cfg


def encoder_signature(cfg) -> str:
    signature = {
        "target": str(cfg.encoder.get("_target_", "")),
        "weights": str(cfg.encoder.get("encoder_weights", "")),
        "input_bands": OmegaConf.to_container(cfg.encoder.get("input_bands"), resolve=True),
        "modalities": OmegaConf.to_container(cfg.encoder.get("modalities"), resolve=True),
        "input_size": cfg.encoder.get("input_size"),
    }
    return json.dumps(signature, sort_keys=True)


def select_latest_per_encoder(run_dirs: list[Path], dataset_root: Path | None) -> list[Path]:
    latest: dict[str, tuple[str, Path]] = {}
    for run_dir in run_dirs:
        try:
            cfg = load_run_config(run_dir, dataset_root=dataset_root)
        except Exception:
            continue
        sig = encoder_signature(cfg)
        stamp = run_dir.name
        if sig not in latest or stamp > latest[sig][0]:
            latest[sig] = (stamp, run_dir)
    return sorted(path for _, path in latest.values())


def build_model(cfg, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    encoder = instantiate(cfg.encoder)
    decoder = instantiate(cfg.decoder, encoder=encoder)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    decoder.load_state_dict(state_dict)
    decoder.to(device)
    decoder.eval()
    return decoder


def run_inference(
    model: torch.nn.Module,
    batch: dict[str, object],
    device: torch.device,
    inference_mode: str,
    sliding_inference_batch: int | None,
) -> np.ndarray:
    image = {
        key: value.to(device)
        for key, value in batch["image"].items()
    }
    target = batch["target"].to(device)

    with torch.no_grad():
        if inference_mode == "sliding":
            logits = Evaluator.sliding_inference(
                model,
                image,
                model.encoder.input_size,
                output_shape=target.shape[-2:],
                max_batch=sliding_inference_batch,
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
    cfg,
    dataset_root: Path,
    output_root: Path,
    split: str,
    ckpt_name: str,
    device: torch.device,
    overwrite: bool,
    max_samples: int | None,
) -> dict[str, object]:
    run_name = run_dir.name
    model_name = f"{cfg.encoder._target_.split('.')[-1]}__{run_name}"
    run_output_dir = output_root / model_name
    raw_dir = run_output_dir / "index"
    color_dir = run_output_dir / "color"
    raw_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)

    preprocessor: Preprocessor = instantiate(
        cfg.preprocessing.test,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )
    dataset = FullSceneSegmentationDataset(
        cfg,
        split=split,
        dataset_root=dataset_root,
        preprocessor=preprocessor,
        max_samples=max_samples,
    )
    model = build_model(cfg, run_dir / ckpt_name, device)
    collate_fn = get_collate_fn(list(cfg.encoder.input_bands.keys()))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    saved = 0
    skipped = 0
    errors: list[str] = []

    progress = tqdm(loader, desc=run_name, leave=False)
    for batch in progress:
        metadata = batch["metadata"][0]
        stem = Path(str(metadata["filename"])).stem
        raw_path = raw_dir / f"{stem}.png"
        color_path = color_dir / f"{stem}.png"

        if not overwrite and raw_path.exists() and color_path.exists():
            skipped += 1
            continue

        try:
            prediction = run_inference(
                model=model,
                batch=batch,
                device=device,
                inference_mode=str(cfg.task.evaluator.inference_mode),
                sliding_inference_batch=cfg.task.evaluator.get("sliding_inference_batch"),
            )
            save_palette_png(prediction, raw_path, dataset.palette)
            save_color_png(prediction, color_path, dataset.palette)
            saved += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{metadata['filename']}: {exc}")

    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / ckpt_name),
        "dataset_root": str(dataset_root),
        "model_name": model_name,
        "split": split,
        "saved": saved,
        "skipped_existing": skipped,
        "errors": errors,
        "classes": list(cfg.dataset.classes),
    }
    with open(run_output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def manifests_to_summary(manifests: Iterable[dict[str, object]], output_path: Path) -> None:
    manifests = list(manifests)
    summary = {
        "total_runs": len(manifests),
        "runs": manifests,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export test-set segmentation predictions for saved PANGAEA runs."
    )
    parser.add_argument("--runs-root", type=Path, required=True, help="Root directory containing run subdirectories.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root containing test/images and test/masks.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory where model-wise predictions will be written.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to export.")
    parser.add_argument("--ckpt-name", type=str, default="checkpoint__best.pth", help="Checkpoint filename inside each run directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--match", type=str, default=None, help="Only export runs whose directory path contains this substring.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs.")
    parser.add_argument("--max-samples", type=int, default=None, help="Only export the first N samples per run.")
    parser.add_argument("--top-level-only", action="store_true", help="Only consider direct child run directories under --runs-root.")
    parser.add_argument("--latest-per-encoder", action="store_true", help="Keep only the latest run for each encoder signature.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    device = torch.device(args.device)

    run_dirs = find_run_dirs(runs_root, args.ckpt_name, top_level_only=args.top_level_only)
    if args.match:
        run_dirs = [run for run in run_dirs if args.match in str(run)]
    if args.latest_per_encoder:
        run_dirs = select_latest_per_encoder(run_dirs, dataset_root=dataset_root)
    if not run_dirs:
        raise FileNotFoundError(f"No runs with {args.ckpt_name} found under {runs_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, object]] = []

    for run_dir in tqdm(run_dirs, desc="Runs"):
        try:
            cfg = load_run_config(run_dir, dataset_root=dataset_root)
            manifest = export_run(
                run_dir=run_dir,
                cfg=cfg,
                dataset_root=dataset_root,
                output_root=output_root,
                split=args.split,
                ckpt_name=args.ckpt_name,
                device=device,
                overwrite=args.overwrite,
                max_samples=args.max_samples,
            )
            manifests.append(manifest)
        except Exception as exc:  # noqa: BLE001
            manifests.append(
                {
                    "run_dir": str(run_dir),
                    "saved": 0,
                    "skipped_existing": 0,
                    "errors": [str(exc)],
                }
            )

    manifests_to_summary(manifests, output_root / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
