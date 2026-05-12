"""Evaluation metrics for binary change detection.

All metrics are computed for the positive (Change) class only, using globally
accumulated confusion-matrix counts to avoid the bias of per-batch averaging
under extreme class imbalance.
"""

import numpy as np
import torch


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary change-detection metrics for the Change class.

    Args:
        preds: Raw logits of shape ``(B, 1, H, W)``.
        targets: Ground-truth masks of shape ``(B, H, W)`` with values 0/1.
        threshold: Probability cutoff applied after sigmoid.

    Returns:
        Dictionary with keys ``iou``, ``precision``, ``recall``, ``f1``,
        ``tp``, ``fp``, ``fn``, ``tn``.
    """
    probs = torch.sigmoid(preds).squeeze(1)
    binary = (probs > threshold).long()

    pred_flat = binary.view(-1)
    target_flat = targets.view(-1)

    tp = ((pred_flat == 1) & (target_flat == 1)).sum().item()
    fp = ((pred_flat == 1) & (target_flat == 0)).sum().item()
    fn = ((pred_flat == 0) & (target_flat == 1)).sum().item()
    tn = ((pred_flat == 0) & (target_flat == 0)).sum().item()

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return a 2×2 confusion matrix as a NumPy array.

    Layout::

        [[TN, FP],
         [FN, TP]]

    Rows correspond to actual class; columns to predicted class.

    Args:
        preds: Raw logits of shape ``(B, 1, H, W)``.
        targets: Ground-truth masks of shape ``(B, H, W)`` with values 0/1.
        threshold: Probability cutoff applied after sigmoid.

    Returns:
        ``np.ndarray`` of shape ``(2, 2)``.
    """
    metrics = compute_metrics(preds, targets, threshold)
    return np.array(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ]
    )


def aggregate_metrics(metric_list: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-batch metrics via global TP/FP/FN/TN accumulation.

    Direct averaging of per-batch F1 scores is biased under class imbalance
    because batches with few positive pixels produce near-zero F1 regardless
    of model quality.  This function accumulates raw counts globally and
    computes metrics once.

    Args:
        metric_list: List of per-batch metric dictionaries produced by
            :func:`compute_metrics`.

    Returns:
        Dictionary with the same keys as :func:`compute_metrics`.
    """
    total_tp = sum(m["tp"] for m in metric_list)
    total_fp = sum(m["fp"] for m in metric_list)
    total_fn = sum(m["fn"] for m in metric_list)
    total_tn = sum(m["tn"] for m in metric_list)

    eps = 1e-7
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)

    return {
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }
