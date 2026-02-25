# Cryo-Bench 🧊

**Cryo-Bench** is a comprehensive benchmark for evaluating geospatial foundation models on cryosphere remote sensing tasks. It is built on top of the [PANGAEA](https://arxiv.org/abs/2412.04204) evaluation protocol and extends it with five cryosphere-specific tasks using multi-sensor satellite imagery.




## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/nicolashoudre/RAMEN
cd RAMEN/pangaea-bench
```

### Option 1: Conda / Mamba (Recommended)

```bash
conda env create -f environment.yaml
conda activate pangaea-bench
```

For faster dependency resolution, use [Mamba](https://github.com/conda-forge/miniforge/releases/):

```bash
wget https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Mambaforge-24.3.0-0-Linux-x86_64.sh
sh ./Mambaforge-24.3.0-0-Linux-x86_64.sh

mamba env create -f environment.yaml
mamba activate pangaea-bench
```

### Option 2: pip + virtualenv

```bash
export PANGAEA_PATH=/path/to/venv/pangaea-bench   # set your desired path
python3 -m venv ${PANGAEA_PATH}
source ${PANGAEA_PATH}/bin/activate

pip install -r requirements.txt
pip install --no-build-isolation --no-deps -e .
```

---

## 🚀 Running Experiments

All experiments use `torchrun` with the PANGAEA `run.py` entry point. Set your output directory before running:

```bash
export BASE=/path/to/your/results
```

---

### 1. Glacial Lake Image Segmentation (GLID)

#### RAMEN Encoder

```bash
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
  --config-name=train \
  dataset=glid \
  encoder=ramen_monotemporal \
  encoder.input_res=30.0 \
  encoder.input_size=40 \
  encoder.res=80.0 \
  decoder=seg_upernet \
  preprocessing=seg_default \
  criterion=cross_entropy \
  task=segmentation
```

> **Note:** `encoder.input_size` and `encoder.res` may need to be tuned depending on your data configuration.

#### Baseline U-Net

```bash
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
  --config-name=train \
  dataset=glid \
  encoder=unet_encoder \
  decoder=seg_unet \
  preprocessing=seg_default \
  criterion=cross_entropy \
  finetune=true \
  task=segmentation
```

#### All Other Encoders (batch sweep)

```bash
BASE=/path/to/results/GLID

for ENC in \
  vit_scratch croma_optical dofa gfmswin prithvi remoteclip satlasnet_si scalemae \
  spectralgpt ssl4eo_moco ssl4eo_dino ssl4eo_mae_optical ssl4eo_data2vec terramind_large
do
  torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
    --config-name=train \
    dataset=glid task=segmentation \
    encoder=${ENC} decoder=seg_upernet preprocessing=seg_default criterion=cross_entropy \
    batch_size=4 num_workers=4 \
    +n_epochs=2 \
    work_dir=${BASE} \
    |& tee ${BASE}/${ENC}.log
done
```

---

### 2. Sea Ice Classification (AI4Arctic)

The AI4Arctic dataset supports two encoder configurations depending on whether the model has been pre-trained on SAR data.

#### RGB / Optical Encoders

```bash
BASE=/path/to/results/SeaIce

for ENC in \
  vit_scratch gfmswin prithvi remoteclip satlasnet_si scalemae \
  spectralgpt ssl4eo_moco ssl4eo_dino ssl4eo_mae_optical ssl4eo_data2vec terramind_large
do
  torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
    --config-name=train \
    dataset=ai4arctic_rgb task=segmentation \
    encoder=${ENC} decoder=seg_upernet preprocessing=seg_default criterion=cross_entropy \
    batch_size=4 num_workers=4 \
    +n_epochs=2 \
    work_dir=${BASE} \
    |& tee ${BASE}/${ENC}.log
done
```

#### SAR-Aware Encoders (DOFA, TerraMind, RAMEN, CROMA)

For models pre-trained on or compatible with SAR imagery, use the `ai4arctic_terramind` dataset config:

```bash
for ENC in dofa terramind_large croma_sar ramen_monotemporal
do
  torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
    --config-name=train \
    dataset=ai4arctic_terramind task=segmentation \
    encoder=${ENC} decoder=seg_upernet preprocessing=seg_default criterion=cross_entropy \
    batch_size=4 num_workers=4 \
    +n_epochs=2 \
    work_dir=${BASE} \
    |& tee ${BASE}/${ENC}_sar.log
done
```

---

## ⚡ FLOPs & Inference Time Profiling

To evaluate computational cost and performance trade-offs, Cryo-Bench includes a profiler task using [fvcore](https://github.com/facebookresearch/fvcore).

```bash
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
  --config-name=train \
  dataset=glid \
  encoder=ramen \
  encoder.input_res=30.0 \
  encoder.input_size=40 \
  encoder.res=80.0 \
  decoder=seg_upernet \
  preprocessing=seg_default \
  criterion=cross_entropy \
  task=profiler \
  task.trainer.n_epochs=1 \
  batch_size=1
```

> Adjust `encoder.input_size` and `encoder.res` to match your target encoder and dataset resolution.

---


## 📂 Repository Structure

```
Cryo-Bench/
└── pangaea-bench/
    ├── pangaea/
    │   ├── run.py              # Main entry point
    │   ├── datasets/           # Dataset configs and loaders
    │   ├── encoders/           # Encoder model definitions
    │   ├── decoders/           # Decoder heads (UPerNet, U-Net, etc.)
    │   └── ...
    ├── configs/
    │   ├── train.yaml          # Base training config
    │   ├── dataset/            # Per-dataset configs
    │   └── encoder/            # Per-encoder configs
    ├── environment.yaml
    ├── requirements.txt
    └── README.md
```

---

## 📖 Citation

If you use Cryo-Bench or the underlying PANGAEA framework, please cite:

```bibtex
@misc{marsocci2024pangaeaglobalinclusivebenchmark,
  title     = {PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models},
  author    = {Valerio Marsocci and Yuru Jia and Georges Le Bellier and David Kerekes and
               Liang Zeng and Sebastian Hafner and Sebastian Gerard and Eric Brune and
               Ritu Yadav and Ali Shibli and Heng Fang and Yifang Ban and
               Maarten Vergauwen and Nicolas Audebert and Andrea Nascetti},
  year      = {2024},
  eprint    = {2412.04204},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2412.04204}
}
```



> If you use Cryo-Bench specifically, please also cite our paper (coming soon).

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

Cryo-Bench builds on the [PANGAEA benchmark](https://github.com/yurujaja/pangaea-bench) and the [RAMEN](https://github.com/nicolashoudre/RAMEN) framework. We thank the developers of DOFA, TerraMind, Prithvi, SatlasNet, and all other foundation models included in this benchmark.
