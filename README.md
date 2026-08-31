# Cryo-Bench 🧊

> **A Benchmark for Evaluating Geospatial Foundation Models on Cryosphere Applications**
[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Sk-21/Cryo-Bench)
[![PANGAEA](https://img.shields.io/badge/Built%20on-PANGAEA-blue?style=flat-square)](https://arxiv.org/abs/2412.04204)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

![Cryo-Bench Overview](assets/overview.png)

![Cryo-Bench Overview](assets/cryo-bench_data.png)

**Cryo-Bench** is a community benchmark that evaluates geospatial foundation models (GFMs) on six cryosphere segmentation datasets spanning five components: supraglacial debris, glacial lakes (two sensing configurations), sea ice, calving fronts, and Antarctic ice-shelf extent. It is built on top of the [PANGAEA](https://arxiv.org/abs/2412.04204) evaluation protocol using multi-sensor satellite imagery from Sentinel-1/2, Landsat-8, WorldView-2, and historical SAR missions.

---

## 📋 Tasks & Datasets

Cryo-Bench includes six benchmark tasks covering five components of the cryosphere:

| Dataset | Component | Location | Sensors | Classes | Ancillary Data | Paper | Download |
|---------|-----------|----------|---------|---------|----------------|-------|----------|
| **GSDD** | Supraglacial Debris | Global | Sentinel-2 | Binary | Slope, Elevation, Velocity | [Article](https://www.sciencedirect.com/science/article/pii/S2666017225001257) | [Zenodo](https://zenodo.org/records/17161810) |
| **GLID** | Glacial Lakes | Himalayas | WorldView-2, Sentinel-2, Landsat-8, Gaofen-2 | Binary | — | [Article](https://www.sciencedirect.com/science/article/pii/S002216942500410X) | [Zenodo](https://zenodo.org/records/14838695) |
| **GLB** | Glacial Lakes (multi-source) | High Mountain Asia | Sentinel-2, Sentinel-1, terrain (11 bands) | Binary | Slope, Elevation | [Article](https://essd.copernicus.org/preprints/essd-2026-474/) | [Zenodo](https://zenodo.org/records/17917359) |
| **SICD** | Sea Ice | Canadian & Greenlandic Arctic | Sentinel-1 | Multiclass | Incidence Angle | [Article](https://egusphere.copernicus.org/preprints/2023/egusphere-2023-2648/) | [HuggingFace](https://huggingface.co/datasets/torchgeo/ai4artic-sea-ice-challenge) |
| **CaFFe** | Calving Fronts | Greenland, Alaska, Antarctic Peninsula | ERS-1/2, Envisat, RADARSAT-1, ALOS PALSAR, TSX, TDX, Sentinel-1 | Multiclass | — | [Article](https://essd.copernicus.org/articles/14/4287/2022/) | [PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.940950) |
| **Shelf-Bench** | Ice-Shelf Extent | Antarctica | SAR,  Sentinel-1 | Binary |  | [Article](https://essd.copernicus.org/preprints/essd-2025-758/) | [Zenodo](https://zenodo.org/records/20430768) |

> GLB and Shelf-Bench replace the earlier GLD dataset in this benchmark. GLB's paper is in review; its download link and Shelf-Bench's source citation are not yet finalized -- both are tracked as open items rather than guessed at here.

---

## 🏆 Benchmark Results

Table below reports mIoU (↑) for all models evaluated with **frozen encoders** and **100% training data** using the UPerNet decoder, across all six datasets. Rank (↓) is by six-dataset average mIoU. Baseline models (U-Net, ViT) are trained from scratch. Every value here is parsed directly from run logs -- see the paper for bootstrap confidence intervals on the top of this table (the #1/#2 gap is not statistically resolved).

> **Bold** = best performance · *Italic* = second best

| Model | GSDD | GLID | GLB | SICD | CaFFe | Shelf-Bench | Avg. mIoU ↑ | Rank ↓ |
|-------|:----:|:---:|:---:|:----:|:-----:|:-----------:|:-----------:|:------:|
| **U-Net** | 73.89 | *91.58* | **80.46** | 20.61 | **59.82** | 82.97 | **68.22** | 1 |
| TerraMind | **74.63** | 88.26 | *79.91* | **33.27** | 46.64 | 84.46 | *67.86* | 2 |
| DOFA | 72.96 | **92.61** | 79.39 | 20.41 | 50.71 | **87.08** | 67.19 | 3 |
| RemoteCLIP | 73.42 | 90.89 | 78.09 | 14.51 | 56.64 | 83.59 | 66.19 | 4 |
| GFM-Swin | 73.00 | 89.69 | 78.42 | 11.55 | 58.12 | *85.63* | 66.07 | 5 |
| Scale-MAE | 73.47 | 90.91 | 79.16 | 6.02 | *58.19* | 83.28 | 65.17 | 6 |
| CROMA | 74.15 | 78.52 | 77.68 | *21.81* | 42.03 | 81.15 | 62.56 | 7 |
| **ViT** | *74.41* | 78.14 | 79.81 | 15.98 | 40.25 | 81.86 | 61.74 | 8 |
| S12-MoCo | 73.03 | 75.51 | 75.45 | 20.66 | 36.21 | 74.97 | 59.31 | 9 |
| S12-DINO | 71.19 | 75.69 | 75.17 | 20.59 | 35.58 | 75.28 | 58.92 | 10 |
| S12-Data2Vec | 73.68 | 75.19 | 74.59 | 17.25 | 35.96 | 75.79 | 58.74 | 11 |
| SatlasNet | 73.70 | 77.03 | 74.74 | 19.93 | 33.96 | 73.00 | 58.73 | 12 |
| S12-MAE | 73.51 | 75.71 | 75.43 | 11.21 | 36.99 | 75.93 | 58.13 | 13 |
| Prithvi | 70.52 | 71.11 | 69.08 | 15.37 | 32.01 | 74.41 | 55.42 | 14 |
| SpectralGPT | 73.22 | 70.87 | 72.82 | 12.57 | 32.70 | 66.75 | 54.82 | 15 |

> Encoders are kept **frozen** for all GFMs. U-Net and ViT are trained from scratch. GLB and Shelf-Bench are new in this update, replacing GLD; RAMEN (an earlier author-model entry) has been removed -- its runs did not complete cleanly and it is no longer part of the reported roster.

<!-- Radar chart removed: the previous assets/radar.png showed the old 5-dataset roster
     (GLD, no GLB/Shelf-Bench) and a stale SICD number. Re-add once a replacement is built
     from current data. -->

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---


---

## 🙏 Acknowledgements

Cryo-Bench builds on the [PANGAEA benchmark](https://github.com/yurujaja/pangaea-bench) and the [RAMEN](https://github.com/nicolashoudre/RAMEN) framework. We thank the developers of DOFA, TerraMind, Prithvi, SatlasNet, and all other foundation models included in this benchmark. We also thank the dataset authors of GSDD, GLID, GLB, SICD, CaFFe, and Shelf-Bench for making their data publicly available.
