# Workflows & Troubleshooting


## Quick Start

```bash
# 1. Set up environment (see README.md)
# 2. Place dataset under data/
# 3. Train the recommended model
python src/train.py --config configs/config_exp6.yaml

# 4. Evaluate on test set with threshold tuning
python src/eval.py \
    --config configs/config_exp6.yaml \
    --weights outputs/exp6/checkpoints/best_checkpoint.pth \
    --split test --tune_threshold --visualize
```

---

## Training a New Experiment

### 1. Create a configuration file

Copy an existing config and modify it:

```bash
cp configs/config_exp6.yaml configs/config_exp8.yaml
```

Key sections to adjust:
- `model.encoder_name` — backbone identifier
- `training.epochs`, `training.batch_size`, `training.learning_rate`
- `augmentation.*` — domain-shift mitigations
- `paths.*` — output directories (use `outputs/exp8/`)

### 2. Run training

```bash
python src/train.py --config configs/config_exp8.yaml
```

Training logs are saved to `outputs/exp8/logs/train_<timestamp>.log`.
Checkpoints are saved every 10 epochs and whenever validation F1 improves.

### 3. Monitor progress

```bash
# Watch the log in real-time
tail -f outputs/exp8/logs/train_*.log

# Quick check of best F1
grep "Best validation" outputs/exp8/logs/train_*.log
```

---

## Evaluation Workflows

### 1. Standard evaluation (val-optimal threshold >> test)

```bash
python src/eval.py \
    --config configs/config_exp6.yaml \
    --weights outputs/exp6/checkpoints/best_checkpoint.pth \
    --split test --tune_threshold
```

This:
1. Sweeps thresholds {0.1, 0.2, …, 0.9} on the **validation** set
2. Selects the threshold maximising val F1
3. Applies that threshold to the **test** set
4. Reports F1, IoU, precision, recall, and confusion matrix

### 2. With test-time augmentation (TTA)

```bash
python src/eval.py --config configs/config_exp6.yaml \
    --weights outputs/exp6/checkpoints/best_checkpoint.pth \
    --split test --tune_threshold --tta
```

TTA applies 8-fold augmentation (4 rotations × horizontal flip) and averages
the de-augmented logit maps. Note: TTA can *amplify* failures under severe
domain shift (see Exp 5 results).

### 3. Qualitative visualisations

```bash
python src/eval.py --config configs/config_exp6.yaml \
    --weights outputs/exp6/checkpoints/best_checkpoint.pth \
    --split test --tune_threshold --visualize
```

Generates 5-panel PNG images (EO | SAR | Ground truth | Prediction | Error
overlay) in `outputs/exp6/evals/test/`.

### 4. Oracle threshold diagnostic

```bash
python src/eval.py --config configs/config_exp7.yaml \
    --weights outputs/exp7/checkpoints/best_checkpoint.pth \
    --split test --tune_threshold --oracle_threshold
```

> **Diagnostic only:** This sweeps thresholds on the test set directly,
> which constitutes information leakage and is not valid for reporting. It
> quantifies how much of the val→test F1 gap is calibration vs. representation.

---

## Generating Training Curves

```bash
# Generate combined comparison chart for all experiments
python scripts/plot_all_experiments.py

# Output: report/figures/training_curves.png
```

---

## Log Formatting

Raw log files can be converted to structured markdown:

```bash
# Format all logs for experiment 7
python scripts/format_logs.py --exp 7

# Format all experiments
python scripts/format_logs.py --all

# Write to file
python scripts/format_logs.py --exp 7 --output outputs/exp7/logs/summary.md
```

---

## SatlasPretrain (Experiment 7)

Experiment 7 uses a Swin-V2-B backbone with weights from the
[SatlasPretrain](https://github.com/allenai/satlaspretrain_models) project.

**First run:** Weights are automatically downloaded from HuggingFace and cached in `weights_cache/`.

**2-phase training schedule:**
- Phase 1 (epochs 1–10): Backbone + FPN frozen; only decoder trains
- Phase 2 (epochs 11–60): Full network fine-tuned at 10× lower backbone LR

**Memory requirements:** The 90M-parameter model uses gradient accumulation
(batch 4 × accum 4 = effective batch 16) to fit within 24 GB VRAM.


