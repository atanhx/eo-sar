"""Binary change detection pipeline for EO-SAR image pairs.

This package implements an early-fusion UNet-based binary change detection
system for co-registered Electro-Optical (EO) and Synthetic Aperture Radar
(SAR) satellite imagery.

Modules:
    dataset:  Data loading, augmentation, and change-aware sampling.
    model:    UNet (smp) and SatlasUNet architectures with Dice+Focal loss.
    metrics:  Globally-accumulated F1, IoU, precision, recall.
    train:    Training loop with 2-phase scheduling and gradient accumulation.
    eval:     Inference with TTA, threshold tuning, and visualisation.
    utils:    Logging and timestamp utilities.
"""
