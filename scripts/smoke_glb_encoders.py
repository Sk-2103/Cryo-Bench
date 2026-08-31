#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import os
import traceback
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from pangaea.datasets.base import GeoFMDataset


DEFAULT_ENCODERS = [
    "croma_joint",
    "croma_optical",
    "croma_sar",
    "dofa",
    "gfmswin",
    "prithvi",
    "prithvi_eo_v2_300",
    "remoteclip",
    "satlasnet_mi",
    "satlasnet_si",
    "scalemae",
    "spectralgpt",
    "ssl4eo_data2vec",
    "ssl4eo_dino",
    "ssl4eo_mae_optical",
    "ssl4eo_mae_sar",
    "ssl4eo_moco",
    "terramind_large",
    "unet_encoder",
    "vit_scratch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test GLB encoder configs.")
    parser.add_argument("--encoders", nargs="*", default=DEFAULT_ENCODERS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", default="glb_random")
    parser.add_argument("--split", default="train")
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--results-dir", default="/tmp/glb_encoder_smoke")
    return parser.parse_args()


def batch_to_device(sample: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.unsqueeze(0).to(device, non_blocking=True)
        for key, value in sample["image"].items()
    }


def smoke_encoder(name: str, args: argparse.Namespace, logger: logging.Logger) -> None:
    decoder = "seg_unet" if name in {"unet_encoder", "unet_encoder_mi"} else "seg_upernet"
    finetune = "true" if name in {"unet_encoder", "unet_encoder_mi", "vit_scratch"} else "false"
    overrides = [
        f"dataset={args.dataset}",
        f"encoder={name}",
        f"decoder={decoder}",
        "preprocessing=seg_default",
        "criterion=cross_entropy",
        "task=segmentation",
        f"work_dir={args.results_dir}",
        f"finetune={finetune}",
        "batch_size=1",
        "test_batch_size=1",
        "num_workers=0",
        "test_num_workers=0",
        "use_wandb=false",
    ]
    if args.skip_weights:
        overrides.extend(["encoder.encoder_weights=null", "encoder.download_url=null"])

    cfg = compose(config_name="train", overrides=overrides)
    logger.info("[%s] composed config", name)

    encoder = instantiate(cfg.encoder)
    if not args.skip_weights:
        encoder.load_encoder_weights(logger)
    decoder = instantiate(cfg.decoder, encoder=encoder)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    decoder.to(device)
    decoder.eval()
    logger.info("[%s] built decoder on %s", name, device)

    preprocessor = instantiate(
        cfg.preprocessing.train,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )
    raw_dataset = instantiate(cfg.dataset, split=args.split)
    dataset = GeoFMDataset(raw_dataset, preprocessor)
    sample = dataset[0]
    image = batch_to_device(sample, device)
    target_shape = sample["target"].shape[-2:]

    with torch.no_grad():
        output = decoder(image, output_shape=target_shape)
    logger.info("[%s] PASS output_shape=%s", name, tuple(output.shape))


def main() -> int:
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("glb-smoke")

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    failures = []
    with initialize(version_base=None, config_path="../configs"):
        for name in args.encoders:
            logger.info("[%s] START", name)
            try:
                smoke_encoder(name, args, logger)
            except Exception as exc:
                failures.append(name)
                logger.error("[%s] FAIL %s", name, exc)
                logger.error("[%s] TRACE\n%s", name, traceback.format_exc())
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = {"passed": [e for e in args.encoders if e not in failures], "failed": failures}
    logger.info("SUMMARY %s", OmegaConf.to_yaml(summary).strip())
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
