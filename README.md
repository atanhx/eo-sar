# Binary Change Detection on EO-SAR Image Pairs

Pixel-level binary change detection on co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) satellite image pairs for identifying damaged or destroyed buildings following disaster events.

---

## Key Results

Seven experiments progressed from a vanilla CNN baseline through domain-specific mitigations to a satellite-domain foundation model. The dominant failure mode is **geographic domain shift** between validation (scenes 01/02) and test (scenes 09/10) — not class imbalance or model capacity.

| Exp | Architecture/Method | Val F1 | Test F1 | Notes |
|:---:|:-------------|:------:|:-------:|:------|
| 1 | ResNet-34 (α=0.25) | 0.4325 | 0.0206 | Baseline |
| 2 | MiT-B2 (α=0.75) | **0.5686** | 0.0095 | Best validation |
| 3 | ResNet-34 (α=0.75) | 0.5491 | 0.0363 | Improved training |
| 4 | MiT-B2 + Domain Aug | 0.5637 | 0.0194 | Standard augmentation |
| 5 | TTA + Sliding Window | — | — | Re-evaluation only (no training) |
| 6 ★ | ResNet-34 + SAR IN + Dropout | 0.5110 | **0.0921** | **Best test generalization** |
| 7 | SatlasPretrain Swin-V2-B | 0.4728 | 0.0392 | Foundation model |

**★ Recommended checkpoint** — Experiment 6 achieves the best test F1 by directly attacking the domain-shift mechanism (SAR instance normalisation + channel dropout) rather than adding architectural complexity.

---

## Requirements

- **Python:** 3.10+ (tested on 3.12, 3.13)
- **PyTorch:** 2.x with CUDA support
- **GPU:** NVIDIA GPU with 16 GB VRAM (Free T4 Google Colab, Free L4 Lightning Studio or better recommended)

## Environment Setup

```bash
# Clone the repository
git clone <repository_url>
cd eo-sar


# UV (Recommended)
uv sync

# OR with venv + pip

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset Structure

Download the dataset and place it under `<root>/data/`:

```text
data/
├── train/
│   ├── pre-event/       # 3-channel RGB optical images (.tif)
│   ├── post-event/      # 1-channel SAR backscatter images (.tif)
│   └── target/          # 4-class annotation masks (.tif)
├── val/
│   ├── pre-event/ / post-event/ / target/
└── test/
    ├── pre-event/ / post-event/ / target/
```

> Labels are automatically remapped to binary: {Background, Intact} >> 0 (No-Change), {Damaged, Destroyed} >> 1 (Change).

---

## Usage

### Training

Training is configured via YAML files in `configs/`. 
<br>Each experiment has its own configuration:

```bash
# Experiment 6 (recommended — best test generalisation)
python src/train.py --config configs/config_exp6.yaml

# Experiment 7 (SatlasPretrain Swin-V2-B foundation model)
python src/train.py --config configs/config_exp7.yaml

# Any experiment N ∈ {1, 2, 3, 4, 6, 7}
python src/train.py --config configs/config_expN.yaml
```

Outputs (checkpoints, logs) are saved to `outputs/expN/`.

### Evaluation

```bash
# Evaluate with validation-optimal threshold on test split
python src/eval.py \
    --config configs/config_exp6.yaml \
    --weights outputs/exp6/checkpoints/exp6_best_checkpoint.pth \
    --split test --tune_threshold

# With test-time augmentation (8-fold: 4 rotations × flip)
python src/eval.py \
    --config configs/config_exp6.yaml \
    --weights outputs/exp6/checkpoints/exp6_best_checkpoint.pth \
    --split test --tune_threshold --tta

# Generate qualitative visualisations
python src/eval.py \
    --config configs/config_exp6.yaml \
    --weights outputs/exp6/checkpoints/exp6_best_checkpoint.pth \
    --split test --tune_threshold --visualize

# Oracle threshold diagnostic (analysis only — not valid for reporting)
python src/eval.py \
    --config configs/config_exp7.yaml \
    --weights outputs/exp7/checkpoints/exp6_best_checkpoint.pth \
    --split test --tune_threshold --oracle_threshold
```

### Utility Scripts

```bash
# Dataset exploration and statistics
python scripts/exploration.py

# Generate combined training curves for all experiments
python scripts/plot_all_experiments.py

# Format raw logs into structured markdown
python scripts/format_logs.py --all
python scripts/format_logs.py --exp 7
```



---

## Experimental Overview

| Exp | Motivation | Key Change | Finding |
|:---:|:-----------|:-----------|:--------|
| 1 | CNN baseline | ResNet-34, α=0.25 | Global metric accumulation changed F1 from 0.23>>0.43 |
| 2 | Transformer encoder | MiT-B2, weighted sampler | Best val F1 (0.57) but worst test F1 (0.01) — overfitting |
| 3 | Ablation | ResNet-34 + Exp 2 training | Smaller capacity limits overfitting >> better test F1 |
| 4 | Domain augmentation | ±45° rotation, SAR noise | Marginal test improvement; augmentation alone insufficient |
| 5 | Inference techniques | TTA + sliding window | TTA *amplifies* domain-shift failure on MiT-B2 |
| 6 | Domain-shift mitigation | SAR instance norm, channel dropout | **Best generalization on test**  (~153% over Exp 3) |
| 7 | Foundation model | SatlasPretrain Swin-V2-B, 2-phase training | Optical-only pretraining limits cross-modal transfer |

---

## Citation / References

- Daudt et al. (2018). *Fully Convolutional Siamese Networks for Change Detection.* IEEE ICIP.
- Xie et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS.
- Bastani et al. (2023). *SatlasPretrain: A Large-Scale Dataset for Remote Sensing Image Understanding.* IEEE ICCV.
- Liu et al. (2022). *Swin Transformer V2: Scaling Up Capacity and Resolution.* IEEE CVPR.
- Lin et al. (2017). *Focal Loss for Dense Object Detection.* IEEE ICCV.
- Milletari et al. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* IEEE 3DV.
- Ulyanov et al. (2016). *Instance Normalization: The Missing Ingredient for Fast Stylization.* arXiv.
- Iakubovskii, P. (2019). *Segmentation Models PyTorch.* [GitHub](https://github.com/qubvel/segmentation_models.pytorch).

## License

See [LICENSE](./LICENSE).
