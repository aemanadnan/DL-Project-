# Benchmarking Noise-Robust Training Strategies for Liver Tumor Segmentation on LiTS

A controlled cross-strategy comparison of four common training-side improvements (Focal loss, 2.5D context, dropout regularisation, two-stage cascade) under reproducible 50% slice-level label corruption on the LiTS dataset.

> **Headline finding.** At 50% combined label noise, **three of four strategies preserve clean-label tumor Dice to within run-to-run variance**. The strongest 2.5D variant (Stack-5) actually *improves* by +0.022 under noise, consistent with noise acting as implicit regularisation against an overfitting baseline. Conventional regularisation (cosine LR + dropout) did not address Stack-5's persistent overfitting; controlled label noise did.

**Authors:** Aeman Adnan (26100098), Zainab Nabi (26100392) — Group 7

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Dataset](#dataset)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Strategy Summary](#strategy-summary)
- [Clean-Label Results](#clean-label-results)
- [Noise Robustness Results](#noise-robustness-results)
- [Key Implementation Files](#key-implementation-files)
- [Limitations](#limitations)
- [Authors and Acknowledgments](#authors-and-acknowledgments)

---

## Overview

### Research Question

Among common training-side improvements for liver tumor segmentation on LiTS, **which strategy degrades least when training labels are systematically corrupted?**

### Approach

Rather than propose a new architecture, we built a controlled benchmark:

1. Established a clean-label 2D U-Net baseline.
2. Implemented and evaluated several training-side improvements on clean labels.
3. Built a reproducible slice-level label-noise injection framework with three noise types (`missing`, `jitter`, `combined`).
4. Re-trained four representative strategies under 50% combined noise with **byte-equivalent corruption across strategies** (same noise seed → same corrupted labels for every strategy).
5. Compared degradation, evaluating always on clean validation labels.

### Why This Matters

Prior LiTS work proposes individual training strategies in isolation, on different baselines, with different setups. No prior work performs this systematic cross-strategy comparison under matched noise conditions on LiTS. The infrastructure built for this project ([`lits_core.py`](lits_core.py)) is reusable for follow-on studies at additional rates, types, and strategies.

---

## Repository Structure

```
.
├── README.md                          ← you are here
├── lits_core.py                       ← shared module (datasets, U-Net, training, noise injection)
├── lits_cascade.py                    ← cascade-specific helpers (Stage-1 inference, cropped dataset)
│
├── notebooks/
│   ├── clean_baselines/
│   │   ├── d3_baseline.ipynb          ← Baseline 2D U-Net (Dice + CE)
│   │   ├── d4_focal_loss.ipynb        ← Focal loss + capped class weights
│   │   ├── d5_curriculum.ipynb        ← Curriculum sampling (failed; excluded from D7)
│   │   ├── d6_25d_unet.ipynb          ← 2.5D U-Net, stack=3
│   │   ├── d6b_25d_augmented.ipynb    ← 2.5D + data augmentation
│   │   ├── d6c_stack5.ipynb           ← 2.5D U-Net, stack=5 (best clean single-stage)
│   │   ├── d6c_v2_cosine_dropout.ipynb ← Stack-5 + cosine LR + bottleneck dropout
│   │   ├── d6d_attention.ipynb        ← Stack-5 + attention gates (exploratory)
│   │   ├── d6e_mild_augment.ipynb     ← Mild augmentation variant
│   │   └── d7b_cascade_oversample.ipynb ← Two-stage cascade + Tversky-only
│   │
│   └── noise_robustness/
│       ├── d7_D4_noise.ipynb          ← D4 under 50% combined noise
│       ├── d7_D6c_noise.ipynb         ← D6c under 50% combined noise
│       ├── d7_D6cV2_noise.ipynb       ← D6cV2 under 50% combined noise
│       └── d7_D7b_cascade_noise.ipynb ← D7b under 50% combined noise (binary case)
│
├── results/                           ← per-strategy JSON result files
│   ├── results_d7_D4.json
│   ├── results_d7_D6c.json
│   ├── results_d7_D6cV2.json
│   └── results_d7_D7b.json
│
├── plots/                             ← result visualisations
│   ├── result_clean_vs_noisy.png
│   ├── result_epoch_shift.png
│   └── result_training_curves.png
│
├── report/
│   └── lits_d7_final_report.pdf       ← detailed project writeup
│
└── requirements.txt
```

---

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.0+ (with MPS for Apple Silicon, or CUDA for NVIDIA GPUs)
- See [`requirements.txt`](requirements.txt) for the full dependency list.

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/lits-noise-robustness.git
cd lits-noise-robustness

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Core dependencies:

```
torch>=2.0
numpy
nibabel
scipy
matplotlib
jupyter
```

---

## Dataset

We use the **LiTS (Liver Tumor Segmentation) dataset**. Download from Kaggle:

> https://www.kaggle.com/datasets/andrewmvd/lits-png  
> *(Or the original NIfTI distribution from the LiTS challenge.)*

The dataset is **not included in this repo** due to its size (~50 GB).

### What we used

- **51 of 131 publicly-released training volumes** (parts pt3, pt4, pt5 from the Kaggle archive split). The pt1–pt5 partitioning is a download convenience, not a stratified split — see report §3.2 for the bias analysis.
- **70/15/15 volume-level split** with `seed=42`: 35 train, 8 validation, 8 test volumes.
- **Preprocessing:** HU windowing to [-100, 400], rescale to [0, 1], resize to 128×128, drop background-only slices.

After unzipping, point each notebook's `DATA_DIR` to your local data location:

```python
DATA_DIR = '/path/to/your/LiTS/archive'
```

The preprocessed `.npy` cache is created on the first pass and reused thereafter.

---

## Reproducing the Experiments

### Clean Baselines

Run the notebooks in [`notebooks/clean_baselines/`](notebooks/clean_baselines/) in order. Each is self-contained — open in Jupyter, set `DATA_DIR`, and run all cells.

```bash
cd notebooks/clean_baselines
jupyter notebook d3_baseline.ipynb
```

Recommended order:

1. `d3_baseline.ipynb` — Reference baseline (Dice + CE, no extras)
2. `d4_focal_loss.ipynb` — Focal loss + capped class weights
3. `d6_25d_unet.ipynb` — 2.5D Stack-3
4. `d6c_stack5.ipynb` — 2.5D Stack-5 (best clean single-stage)
5. `d6c_v2_cosine_dropout.ipynb` — Stack-5 + cosine + dropout
6. `d7b_cascade_oversample.ipynb` — Two-stage cascade

Each notebook trains for 18–30 epochs depending on strategy (M4 Mac: ~30–60 min/run; CUDA GPU: ~10–20 min/run).

### Noise Robustness Experiments

Each noise notebook in [`notebooks/noise_robustness/`](notebooks/noise_robustness/) is a self-contained re-run of one strategy under label noise.

**Workflow per notebook:**

1. **Cell 1** — Set `NOISE_TYPE` and `NOISE_RATE`. Default is `combined` at `0.50`.
2. **Cells 2–5** — Imports, configuration, data loading (auto, no edits needed).
3. **Cell 6** — Builds `train_dataset` with noise applied; `val_dataset` stays clean.
4. **Cell 7** — Visual sanity check: shows clean vs corrupted slices side-by-side. **Verify this looks right before training.**
5. **Cells 8–11** — Build model, train, save checkpoint.
6. **Cell 12** — Append result entry to `results/results_d7_<strategy>.json`.
7. **Cell 13** — Auto-plot degradation curve (when ≥2 rates exist).

**Run all four:**

```bash
cd notebooks/noise_robustness
jupyter notebook d7_D4_noise.ipynb
# Run, repeat for D6c, D6cV2, D7b
```

To replicate our exact noise pattern, use `NOISE_SEED = 2026`. With this seed held constant, all four strategies see byte-equivalent corrupted labels.

---

## Strategy Summary

| Strategy | Notebook | What it Adds | Addresses |
|---|---|---|---|
| **D3** Baseline | `d3_baseline.ipynb` | 2D U-Net + Dice + CE | Reference |
| **D4** Focal | `d4_focal_loss.ipynb` | Replace CE with Focal (γ=2.0) + capped 5:1 class weights | Class imbalance |
| **D5** Curriculum | `d5_curriculum.ipynb` | Sample slices weighted by tumor pixel count | Easy→hard ordering *(failed)* |
| **D6** 2.5D Stack-3 | `d6_25d_unet.ipynb` | Stack 3 adjacent slices as input channels | Spatial context |
| **D6b** 2.5D + Aug | `d6b_25d_augmented.ipynb` | Add random flips, rotations, brightness/contrast jitter | Overfitting |
| **D6c** 2.5D Stack-5 | `d6c_stack5.ipynb` | Widen stack to 5 slices | More spatial context |
| **D6cV2** + Cosine + Dropout | `d6c_v2_cosine_dropout.ipynb` | Cosine LR + bottleneck Dropout2d(p=0.3) | Regularisation |
| **D6d** Attention | `d6d_attention.ipynb` | Add attention gates on skip connections | Skip-connection refinement |
| **D7b** Cascade | `d7b_cascade_oversample.ipynb` | Two-stage: D6cV2 frozen for liver → binary U-Net for tumor inside crop, with 3:1 oversample + Tversky-only loss | Task decomposition + empty-prediction failure mode |

---

## Clean-Label Results

Validation Dice on the held-out 8 validation volumes. All experiments share the same architecture (2D U-Net, base_features=64), data split (seed=42), and optimiser (Adam lr=1e-4).

| Strategy | Liver Dice | Tumor Dice | Best Epoch | Δ Tumor vs D3 |
|---|---|---|---|---|
| D3 Baseline | 0.8885 | 0.5011 | 8 | — |
| D4 Focal + class weights | 0.8901 | 0.4961 | 7 | −0.005 |
| D5 Curriculum sampling | 0.8886 | 0.4774 | 13 | −0.024 |
| D6 2.5D Stack-3 | 0.8917 | 0.5314 | 8 | +0.030 |
| **D6c 2.5D Stack-5** | **0.8917** | **0.5749** | **7** | **+0.074** |
| D6cV2 + Cosine + Dropout | ~0.89 | 0.5649 | 14 | +0.064 |
| D7b Cascade (binary metric)* | N/A | ~0.65 | ~20 | N/A |

*\* D7b reports binary tumor Dice on oracle-cropped patches — different metric from the multi-class tumor Dice in the other rows. Not directly comparable.*

**Takeaways:**
- 2.5D context is the single biggest lever (Stack-3 +0.030, Stack-5 +0.074).
- Loss-level reweighting (Focal) saturates quickly — Dice already handles imbalance.
- Curriculum sampling actively hurts.
- Conventional regularisation (cosine + dropout) didn't address Stack-5's overfitting on this setup.

---

## Noise Robustness Results

Tumor Dice under **combined 50% slice-level noise** (validation evaluated on clean labels).

| Strategy | Clean Tumor | Noisy Tumor | Δ | Best Ep (clean / noisy) |
|---|---|---|---|---|
| D4 Focal | 0.4961 | 0.5057 | **+0.010** | 7 / 4 |
| **D6c Stack-5** | 0.5749 | **0.5971** | **+0.022** | 7 / 11 |
| D6cV2 + Cosine + Dropout | 0.5649 | 0.5528 | −0.012 | 14 / 11 |
| D7b Cascade* | ~0.650 | 0.5636 | −0.086 | 20 / 7 |

*\* Different metric — see report §10.3 for the cascade-noise interaction analysis.*

**Three of four strategies preserve clean-label performance under 50% noise.** D6c actually improves, with the best epoch shifting later (7 → 11) — direct evidence that noise acts as implicit regularisation against the overfitting baseline.

See [`plots/result_clean_vs_noisy.png`](plots/result_clean_vs_noisy.png) and [`plots/result_epoch_shift.png`](plots/result_epoch_shift.png) for visualisations.

### Why the Robustness?

Three concurrent mechanisms (full discussion in report §11):

1. **Implicit regularisation** — clean-label setup overfit heavily (10× train/val gap); noise breaks memorisation.
2. **Dice-loss noise tolerance** — ratio formulation bounds per-pixel error impact.
3. **Volumetric redundancy** — for 2.5D models, neighbouring slices retain correct labels and provide corrective signal.

---

## Key Implementation Files

### `lits_core.py`

Single-import module covering the entire pipeline:

```python
from lits_core import (
    get_device,
    load_lits_paths,
    make_split,
    LiTS25DDataset,        # built-in noise injection via noise_type / noise_rate / noise_seed
    UNet2D,
    dice_loss,
    combined_loss,
    compute_dice_per_class,
    train_model,
)
```

Key features:

- `LiTS25DDataset` supports `stack_size` (1, 3, 5, ...), augmentation, and slice-level label noise injection in a single class.
- Noise injection accepts `noise_type ∈ {'missing', 'jitter', 'combined', None}` and a `noise_seed` for full reproducibility.
- The same `noise_seed` produces byte-equivalent corrupted labels across all strategy notebooks — this is what makes the cross-strategy comparison fair.

### `lits_cascade.py`

Cascade-specific helpers used by D7b:

- `LiTSLiverCroppedDataset` — crops volumes around the liver bounding box for Stage-2 training.
- `stage1_predict_liver_mask` — runs Stage-1 inference on a CT volume.
- `make_tumor_oversampler` — `WeightedRandomSampler` for the 3:1 tumor oversampling.
- `tversky_loss`, `combined_tversky_loss` — Stage-2 loss functions.

The `NoisyBinaryWrapper` for D7b's binary mask noise is defined inline in `d7_D7b_cascade_noise.ipynb` for transparency.

---

## Limitations

We document these openly so results are interpreted correctly:

- **Subset of LiTS** — 51 of 131 volumes due to compute. Internal validity preserved; absolute Dice slightly lower than full-data SoTA.
- **128×128 input resolution** — native LiTS is 512×512. Downsampling costs small-tumor sensitivity.
- **2D / 2.5D, not full 3D** — 3D nnU-Net would integrate inter-slice context natively.
- **Single noise rate** — Combined at 50% only. No degradation curve across rates.
- **Single random seed** — Differences ±0.02 are within run-to-run variance.
- **D7b uses a different metric** — Binary tumor Dice on oracle-cropped patches. Not directly comparable to multi-class numbers; flagged repeatedly in the report.
- **D6cV2 and D7b each apply two changes** — Cosine + dropout and oversampling + Tversky-only respectively. We can't cleanly attribute observed effects to one component.

The relative comparisons across strategies remain valid regardless of these limitations — that's the controlled-experiment design.

---

## Authors and Acknowledgments

**Group 7**
- Aeman Adnan (26100098)
- Zainab Nabi (26100392)

Built for the Deep Learning course project. Detailed report in [`report/lits_d7_final_report.pdf`](report/lits_d7_final_report.pdf).

### Citations

If you use this infrastructure or the noise injection mechanism, please cite the following:

- Bilic, P. et al. *The Liver Tumor Segmentation Benchmark (LiTS).* arXiv:1901.04056, 2019.
- Ronneberger, O., Fischer, P., Brox, T. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015.
- Lin, T.-Y. et al. *Focal Loss for Dense Object Detection.* ICCV 2017.
- Salehi, S. et al. *Tversky loss function for image segmentation.* MLMI 2017.
- Loshchilov, I., Hutter, F. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017.
- Szegedy, C. et al. *Rethinking the Inception Architecture for Computer Vision.* CVPR 2016.

---

## License

This project is released for academic use. The LiTS dataset itself is governed by its own license terms — see the [LiTS challenge page](https://competitions.codalab.org/competitions/17094).
