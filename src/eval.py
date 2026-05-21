"""Inference, threshold tuning, and visualisation for change detection."""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import get_dataloader, load_tif, remap_labels
from metrics import aggregate_metrics, compute_metrics
from model import build_model, build_satlas_model
from utils import Tee, run_timestamp


def sliding_window_inference(
    model: torch.nn.Module,
    images: torch.Tensor,
    patch_size: int = 512,
    stride: int = 256,
) -> torch.Tensor:
    """Run inference with overlapping sliding windows and average logits.

    Args:
        model: Segmentation model producing ``(B, 1, H, W)`` logits.
        images: Input tensor of shape ``(B, C, H, W)``.
        patch_size: Side length of each inference patch.
        stride: Step size between adjacent patches.

    Returns:
        Averaged logit tensor of shape ``(B, 1, H, W)``.
    """
    B, C, H, W = images.shape
    if H == patch_size and W == patch_size:
        return model(images)

    logits = torch.zeros((B, 1, H, W), device=images.device)
    counts = torch.zeros((B, 1, H, W), device=images.device)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = images[:, :, y : y + patch_size, x : x + patch_size]
            out = model(patch)
            logits[:, :, y : y + patch_size, x : x + patch_size] += out
            counts[:, :, y : y + patch_size, x : x + patch_size] += 1

    return logits / counts


# 8-fold TTA: 4 rotations × {no-flip, horizontal-flip}
_TTA_AUGS: list[tuple[int, bool]] = [
    (0, False),
    (90, False),
    (180, False),
    (270, False),
    (0, True),
    (90, True),
    (180, True),
    (270, True),
]


def _apply_tta_aug(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Apply a rotation + optional horizontal flip to a BCHW tensor."""
    if flip:
        x = torch.flip(x, dims=[3])
    if k > 0:
        x = torch.rot90(x, k=k // 90, dims=[2, 3])
    return x


def _invert_tta_aug(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Invert the TTA augmentation applied by :func:`_apply_tta_aug`."""
    if k > 0:
        x = torch.rot90(x, k=-(k // 90), dims=[2, 3])
    if flip:
        x = torch.flip(x, dims=[3])
    return x


def tta_inference(
    model: torch.nn.Module,
    images: torch.Tensor,
    patch_size: int = 512,
    stride: int = 256,
) -> torch.Tensor:
    """8-fold test-time augmentation with sliding-window inference.

    Applies 4 rotations × {no-flip, horizontal-flip}, runs sliding-window
    inference on each augmented version, inverts the spatial transform, then
    averages the de-augmented logit maps.

    Args:
        model: Segmentation model producing ``(B, 1, H, W)`` logits.
        images: Input tensor of shape ``(B, C, H, W)``.
        patch_size: Side length of each inference patch.
        stride: Step size between adjacent patches.

    Returns:
        Averaged logit tensor of shape ``(B, 1, H, W)``.
    """
    accumulated = torch.zeros(
        (images.shape[0], 1, images.shape[2], images.shape[3]),
        device=images.device,
        dtype=torch.float32,
    )
    for k, flip in _TTA_AUGS:
        aug = _apply_tta_aug(images, k, flip)
        logits = sliding_window_inference(model, aug, patch_size, stride)
        accumulated += _invert_tta_aug(logits, k, flip)
    return accumulated / len(_TTA_AUGS)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "test",
    threshold: float = 0.5,
    use_tta: bool = False,
) -> dict[str, float]:
    """Evaluate model on a full split with sliding-window inference.

    Args:
        model: Trained segmentation model.
        loader: Data loader for the evaluation split.
        device: Compute device.
        split_name: Name of the split (for logging).
        threshold: Binary decision threshold after sigmoid.
        use_tta: If ``True``, apply 8-fold test-time augmentation.

    Returns:
        Aggregated metrics dictionary.
    """
    model.eval()
    batch_metrics = []
    infer_fn = tta_inference if use_tta else sliding_window_inference
    tag = "TTA+SW" if use_tta else "SW"
    print(f"\nEvaluating on {split_name} split [{tag}] with threshold={threshold}...")

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = infer_fn(model, images)
        batch_metrics.append(compute_metrics(logits, masks, threshold=threshold))

    return aggregate_metrics(batch_metrics)


def print_metrics_table(metrics: dict[str, float], split_name: str) -> None:
    """Pretty-print a metrics summary table to stdout.

    Args:
        metrics: Metrics dictionary from :func:`evaluate`.
        split_name: Name of the evaluated split.
    """
    print(f"\n{'=' * 50}")
    print(f"RESULTS: {split_name.upper()} SPLIT")
    print(f"{'=' * 50}")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print("\n  Confusion Matrix:")
    print("                 Predicted")
    print("                 No-Change  Change")
    print(f"  Actual No-Change   {metrics['tn']:>8}  {metrics['fp']:>6}")
    print(f"  Actual Change      {metrics['fn']:>8}  {metrics['tp']:>6}")

    total_actual = metrics["tp"] + metrics["fn"]
    total_pred = metrics["tp"] + metrics["fp"]

    if total_actual > 0:
        miss_rate = metrics["fn"] / total_actual
        print("\n  Error Profile:")
        print(f"  Miss rate (FN/actual change): {miss_rate:.2%}")
        print(
            f"  Of actual change pixels, model caught {metrics['tp'] / total_actual:.2%}"
        )

    if total_pred > 0:
        print(
            f"  False alarm rate (FP/predicted change): {metrics['fp'] / total_pred:.2%}"
        )


def _normalize_display(arr: np.ndarray) -> np.ndarray:
    """Percentile-based contrast stretch for visualisation."""
    arr = arr.astype(float)
    low, high = np.percentile(arr, 2), np.percentile(arr, 98)
    if high > low:
        arr = np.clip((arr - low) / (high - low), 0, 1)
    return arr


def visualize_predictions(
    model: torch.nn.Module,
    data_root: str,
    split: str,
    device: torch.device,
    output_dir: Path,
    n_samples: int = 8,
    threshold: float = 0.5,
    sar_instance_norm: bool = False,
) -> None:
    """Generate side-by-side prediction panels and save as PNGs.

    Panels: EO | SAR | Ground truth | Prediction | Error overlay.

    Args:
        model: Trained segmentation model.
        data_root: Dataset root directory.
        split: Split to visualise.
        device: Compute device.
        output_dir: Directory to save output PNGs.
        n_samples: Maximum number of samples.
        threshold: Binary decision threshold.
        sar_instance_norm: If ``True``, replicate instance normalisation used
            in Experiment 6 training instead of plain ``/255`` scaling.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(data_root) / split
    pre_dir, post_dir, target_dir = (
        split_dir / "pre-event",
        split_dir / "post-event",
        split_dir / "target",
    )

    with_change, without_change = [], []
    for stem in sorted(f.stem for f in pre_dir.glob("*.tif")):
        mask = remap_labels(load_tif(target_dir / f"{stem}.tif"))
        (with_change if mask.mean() > 0.02 else without_change).append(stem)

    selected = (with_change[:5] + without_change[:3])[:n_samples]
    print(f"\nGenerating {len(selected)} prediction visualizations...")
    model.eval()

    for idx, stem in enumerate(selected):
        eo_raw = load_tif(pre_dir / f"{stem}.tif")
        sar_raw = load_tif(post_dir / f"{stem}.tif")
        mask_bin = remap_labels(load_tif(target_dir / f"{stem}.tif"))

        # Replicate dataset preprocessing
        eo_norm = (eo_raw.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [
            0.229,
            0.224,
            0.225,
        ]
        sar_f = sar_raw.astype(np.float32) / 255.0
        if sar_f.ndim == 2:
            sar_f = sar_f[:, :, np.newaxis]
        if sar_instance_norm:
            from dataset import instance_norm_sar  # noqa: PLC0415

            sar_norm = instance_norm_sar(sar_f)
        else:
            sar_norm = sar_f

        eo_t = torch.from_numpy(eo_norm.transpose(2, 0, 1)).float()
        sar_t = torch.from_numpy(sar_norm.transpose(2, 0, 1)).float()
        input_tensor = torch.cat([eo_t, sar_t], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = sliding_window_inference(model, input_tensor)

        prob = torch.sigmoid(logit).squeeze().cpu().numpy()
        pred = (prob > threshold).astype(np.uint8)
        m = compute_metrics(logit.cpu(), torch.from_numpy(mask_bin).long().unsqueeze(0))

        fig, axes = plt.subplots(1, 5, figsize=(22, 5))
        change_pct = 100 * mask_bin.mean()

        axes[0].imshow(_normalize_display(eo_raw))
        axes[0].set_title("EO Pre-Event\n(Optical RGB)", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(_normalize_display(sar_raw), cmap="gray")
        axes[1].set_title("SAR Post-Event\n(Radar Backscatter)", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(mask_bin, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[2].set_title(f"Ground Truth\nChange: {change_pct:.2f}%", fontsize=10)
        axes[2].axis("off")

        axes[3].imshow(pred, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[3].set_title(
            f"Prediction\nPredicted: {100 * pred.mean():.2f}%", fontsize=10
        )
        axes[3].axis("off")

        overlay = np.zeros((*mask_bin.shape, 3), dtype=np.float32)
        overlay[(mask_bin == 1) & (pred == 1)] = [0.0, 0.8, 0.0]
        overlay[(mask_bin == 0) & (pred == 1)] = [0.8, 0.0, 0.0]
        overlay[(mask_bin == 1) & (pred == 0)] = [0.0, 0.0, 0.8]

        axes[4].imshow(eo_raw)
        axes[4].imshow(overlay, alpha=0.6)
        axes[4].set_title(
            f"Error Overlay\nF1:{m['f1']:.3f} IoU:{m['iou']:.3f}", fontsize=10
        )
        axes[4].axis("off")
        axes[4].legend(
            handles=[
                mpatches.Patch(color=[0, 0.8, 0], label="TP"),
                mpatches.Patch(color=[0.8, 0, 0], label="FP"),
                mpatches.Patch(color=[0, 0, 0.8], label="FN"),
            ],
            loc="lower right",
            fontsize=7,
            framealpha=0.8,
        )

        plt.suptitle(
            f"Sample {idx + 1}: {stem}\n"
            f"Prec: {m['precision']:.3f}  Rec: {m['recall']:.3f}  "
            f"F1: {m['f1']:.3f}  IoU: {m['iou']:.3f}",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / f"prediction_{idx + 1:02d}_{stem}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        print(f"  Saved: prediction_{idx + 1:02d}_{stem}.png  (F1: {m['f1']:.3f})")

    print(f"\nAll visualizations saved to {output_dir}")


def find_best_threshold(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool = False,
) -> float:
    """Sweep thresholds on the validation set to maximise F1.

    Args:
        model: Trained segmentation model.
        loader: Validation data loader.
        device: Compute device.
        use_tta: If ``True``, use TTA during the threshold sweep.

    Returns:
        Threshold value maximising validation F1.
    """
    model.eval()
    infer_fn = tta_inference if use_tta else sliding_window_inference

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_metrics = {t: [] for t in thresholds}

    print("\nThreshold sweep on validation set:")
    print(f"{'Threshold':>10}  {'F1':>8}  {'Precision':>10}  {'Recall':>8}  {'IoU':>8}")
    print("=" * 55)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = infer_fn(model, images)
            for t in thresholds:
                batch_metric = compute_metrics(logits, masks, threshold=t)
                all_metrics[t].append(batch_metric)

    best_f1, best_thresh = 0.0, 0.5

    for thresh in thresholds:
        m = aggregate_metrics(all_metrics[thresh])
        print(
            f"{thresh:>10.2f}  {m['f1']:>8.4f}  {m['precision']:>10.4f}  {m['recall']:>8.4f}  {m['iou']:>8.4f}"
        )
        if m["f1"] > best_f1:
            best_f1, best_thresh = m["f1"], thresh

    print(f"\nBest threshold: {best_thresh}  (Val F1: {best_f1:.4f})")
    return best_thresh


def oracle_threshold_sweep(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
    use_tta: bool = False,
) -> None:
    """Diagnostic-only sweep of thresholds directly on a target split.

    .. warning::
        This is a **diagnostic tool only** — not for selecting the operational
        threshold.  Using test-set labels to pick a threshold constitutes
        information leakage and would not be valid for reporting.  The purpose
        here is to quantify how much of the val→test F1 gap is attributable to
        calibration vs. representational shift.

    Args:
        model: Trained segmentation model.
        loader: Data loader for the split being diagnosed.
        device: Compute device.
        split_name: Split name for logging.
        use_tta: If ``True``, apply TTA during inference.
    """
    print(f"\n{'=' * 60}")
    print(f"DIAGNOSTIC: Oracle threshold sweep on {split_name.upper()} split")
    print("(For analysis only — not a valid operational threshold.)")
    print(f"{'=' * 60}")

    model.eval()
    all_logits, all_targets = [], []
    infer_fn = tta_inference if use_tta else sliding_window_inference

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = infer_fn(model, images)
            all_logits.append(logits.cpu())
            all_targets.append(masks.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    print(
        f"\n{'Threshold':>10}  {'F1':>8}  {'Precision':>10}  {'Recall':>8}  {'IoU':>8}"
    )
    print("-" * 55)
    best_f1, best_thresh = 0.0, 0.1
    for thresh in thresholds:
        m = compute_metrics(all_logits, all_targets, threshold=thresh)
        print(
            f"{thresh:>10.2f}  {m['f1']:>8.4f}  {m['precision']:>10.4f}  {m['recall']:>8.4f}  {m['iou']:>8.4f}"
        )
        if m["f1"] > best_f1:
            best_f1, best_thresh = m["f1"], thresh
    print(f"\nOracle best: threshold={best_thresh}  F1={best_f1:.4f}")


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate change detection model")
    parser.add_argument("--config", type=str, default="configs/config_exp3.yaml")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--tune_threshold", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable 8-fold test-time augmentation (4 rotations × flip).",
    )
    parser.add_argument(
        "--oracle_threshold",
        action="store_true",
        help="[DIAGNOSTIC ONLY] Sweep thresholds on the target split directly. "
        "Not a valid operational threshold — for calibration gap analysis only.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.data_path:
        cfg["data"]["root"] = args.data_path

    log_dir = Path(cfg["paths"]["log_dir"])
    suffix = ("_tuned" if args.tune_threshold else "") + (
        "_viz" if args.visualize else ""
    )
    log_path = log_dir / f"eval_{args.split}{suffix}_{run_timestamp()}.log"

    with Tee(log_path):
        print(f"Log: {log_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        print(f"\nLoading model from {args.weights}...")
        is_satlas = cfg["model"]["encoder_name"].startswith("satlas_")

        if is_satlas:
            satlas_id = cfg["model"].get(
                "satlas_identifier",
                "Sentinel2_SwinB_SI_RGB",
            )
            model = build_satlas_model(
                satlas_identifier=satlas_id,
                in_channels=cfg["model"]["in_channels"],
                classes=cfg["model"]["classes"],
                cache_dir=cfg["model"].get("weights_cache_dir", "weights_cache"),
            ).to(device)
        else:
            model = build_model(
                encoder_name=cfg["model"]["encoder_name"],
                encoder_weights=None,
                in_channels=cfg["model"]["in_channels"],
                classes=cfg["model"]["classes"],
            ).to(device)

        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")

        if args.tta:
            print("TTA enabled: 8-fold augmentation (4 rotations × horizontal-flip).")

        final_threshold = args.threshold
        if args.tune_threshold:
            print(f"\nThreshold tuning always runs on val split (not {args.split})")
            val_loader = get_dataloader(
                root=cfg["data"]["root"],
                split="val",
                batch_size=cfg["training"]["batch_size"],
                patch_size=cfg["data"]["patch_size"],
                num_workers=cfg["data"]["num_workers"],
                shuffle=False,
            )
            final_threshold = find_best_threshold(
                model, val_loader, device, use_tta=args.tta
            )
            print(f"\nUsing threshold {final_threshold} for evaluation on {args.split}")

        loader = get_dataloader(
            root=cfg["data"]["root"],
            split=args.split,
            batch_size=cfg["training"]["batch_size"],
            patch_size=cfg["data"]["patch_size"],
            num_workers=cfg["data"]["num_workers"],
            shuffle=False,
        )

        metrics = evaluate(
            model,
            loader,
            device,
            args.split,
            threshold=final_threshold,
            use_tta=args.tta,
        )
        print_metrics_table(metrics, args.split)

        if args.oracle_threshold:
            oracle_threshold_sweep(
                model,
                loader,
                device,
                args.split,
                use_tta=args.tta,
            )

        if args.visualize:
            visualize_predictions(
                model=model,
                data_root=cfg["data"]["root"],
                split=args.split,
                device=device,
                output_dir=Path(cfg["paths"]["eval_output_dir"]) / args.split,
                n_samples=8,
                threshold=final_threshold,
                sar_instance_norm=cfg["augmentation"].get("sar_instance_norm", False),
            )

        print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
