#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap


OUTPUT_ROOT = Path(
    "/media/turtle-ssd/users/skaushik/Flood_Data_compilation/"
    "psfloods_rgvt_ufo_htx_val/output"
)
TEST_IMAGE_DIR = Path(
    "/media/turtle-ssd/users/skaushik/Flood_Data_compilation/"
    "psfloods_rgvt_ufo_htx_val/test/image"
)
TEST_MASK_DIR = Path(
    "/media/turtle-ssd/users/skaushik/Flood_Data_compilation/"
    "psfloods_rgvt_ufo_htx_val/test/mask"
)


MODEL_RUNS = [
    ("prithvi_eo_v2_300", "20260415_104205_f33977_prithvi_eo_v2_300_seg_upernet_psfloods_rgvt", "predictions_test_best"),
    ("terramind_large", "20260415_045006_6d46a9_terramind_large_seg_upernet_psfloods_rgvt", "predictions_test_best_v2"),
    ("spectralgpt", "20260414_191327_2d142a_spectralgpt_seg_upernet_psfloods_rgvt", "predictions_test_best"),
    ("unet_encoder+seg_unet", "20260415_083830_d490d3_unet_encoder_seg_unet_psfloods_rgvt", "predictions_test_best_v2"),
    ("croma_optical", "20260414_130113_84f1ec_croma_optical_seg_upernet_psfloods_rgvt", "predictions_test_best"),
    ("prithvi", "20260415_103300_998035_prithvi_seg_upernet_psfloods_rgvt", "predictions_test_best_v2"),
    ("ssl4eo_moco", "20260415_010829_8e8694_ssl4eo_moco_seg_upernet_psfloods_rgvt", "predictions_test_best_v2"),
    ("ssl4eo_mae_optical", "20260415_004525_e38fe1_ssl4eo_mae_optical_seg_upernet_psfloods_rgvt", "predictions_test_best"),
    ("ssl4eo_dino", "20260415_001103_c76fe1_ssl4eo_dino_seg_upernet_psfloods_rgvt", "predictions_test_best"),
    ("dofa", "20260414_130113_f06996_dofa_seg_upernet_psfloods_rgvt", "predictions_test_best"),
    ("ssl4eo_data2vec", "20260414_234918_5988fb_ssl4eo_data2vec_seg_upernet_psfloods_rgvt", "predictions_test_best"),
    ("scalemae", "20260414_190230_bbed52_scalemae_seg_upernet_psfloods_rgvt", "predictions_test_best"),
    ("resnet50_pretrained", "20260414_184622_9fb315_resnet50_pretrained_seg_upernet_psfloods_rgvt", "predictions_test_best"),
    ("gfmswin", "20260414_130113_2362d5_gfmswin_seg_upernet_psfloods_rgvt", "predictions_test_best"),
    ("remoteclip", "20260414_184610_160d0f_remoteclip_seg_upernet_psfloods_rgvt", "predictions_test_best"),
]


MASK_CMAP = ListedColormap(
    [
        "#f3efe6",  # class 0: non-water
        "#1f78b4",  # class 1: water
    ]
)


def read_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    rgb = arr[[2, 1, 0], :, :]
    rgb = np.transpose(rgb, (1, 2, 0))

    lo = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    return rgb


def read_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def prettify(name: str) -> str:
    return name.replace("_", "\n")


def build_panels(image_name: str):
    panels = [
        ("RGB (B4/B3/B2)", TEST_IMAGE_DIR / image_name, "rgb"),
        ("Ground Truth", TEST_MASK_DIR / image_name, "mask"),
    ]
    for model_name, run_dir, pred_subdir in MODEL_RUNS:
        pred_path = OUTPUT_ROOT / run_dir / pred_subdir / "mask" / image_name
        panels.append((prettify(model_name), pred_path, "mask"))
    return panels


def plot_sample(image_name: str, save_dir: Path, ncols: int = 5) -> None:
    panels = build_panels(image_name)
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.4 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for ax, (title, path, mode) in zip(axes.flat, panels):
        if not path.exists():
            ax.text(0.5, 0.5, f"Missing\n{title}", ha="center", va="center", fontsize=12)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.axis("off")
            continue

        if mode == "rgb":
            ax.imshow(read_rgb(path))
        else:
            ax.imshow(read_mask(path), cmap=MASK_CMAP, vmin=0, vmax=1, interpolation="nearest")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    for ax in axes.flat[len(panels):]:
        ax.axis("off")

    legend_handles = [
        mpatches.Patch(color=MASK_CMAP.colors[0], label="0: Non-water"),
        mpatches.Patch(color=MASK_CMAP.colors[1], label="1: Water"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle(image_name.replace(".tif", ""), fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{Path(image_name).stem}.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=OUTPUT_ROOT / "visualizations_all_models",
    )
    parser.add_argument("--ncols", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    image_names = sorted(p.name for p in TEST_IMAGE_DIR.glob("*.tif"))
    if args.limit is not None:
        image_names = image_names[: args.limit]

    for image_name in image_names:
        out_path = args.save_dir / f"{Path(image_name).stem}.png"
        if args.skip_existing and out_path.exists():
            continue
        plot_sample(image_name, args.save_dir, ncols=args.ncols)

    print(f"Saved {len(image_names)} visualizations to {args.save_dir}")


if __name__ == "__main__":
    main()
