"""Generate combined training metric charts for all experiments.

Reads training logs from ``outputs/exp{N}/logs/`` and produces a single
high-quality PNG with overlaid loss/F1/IoU/recall curves.  Designed to be
included directly in the technical report.

Usage::

    python scripts/plot_all_experiments.py
    python scripts/plot_all_experiments.py --output report/figures/training_curves.png

The script automatically discovers available experiments by scanning the
``outputs/`` directory for training logs.
"""

import argparse
import re
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np


class EpochMetrics(NamedTuple):
    """Parsed per-epoch metrics from a training log."""

    train_loss: list[float]
    val_f1: list[float]
    val_iou: list[float]
    val_recall: list[float]
    val_precision: list[float]


_RE_TRAIN_LOSS = re.compile(r"Train Loss:\s+([\d.]+)")
_RE_VAL_METRICS = re.compile(
    r"Val IoU:\s+([\d.]+)\s+F1:\s+([\d.]+)\s+Precision:\s+([\d.]+)\s+Recall:\s+([\d.]+)"
)


def parse_training_log(log_path: Path) -> EpochMetrics:
    """Extract per-epoch training and validation metrics from a log file.

    Args:
        log_path: Path to the training log (plain text).

    Returns:
        Parsed metrics for all completed epochs.
    """
    losses, f1s, ious, recalls, precs = [], [], [], [], []

    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _RE_TRAIN_LOSS.search(line)
            if m:
                losses.append(float(m.group(1)))
                continue
            m = _RE_VAL_METRICS.search(line)
            if m:
                ious.append(float(m.group(1)))
                f1s.append(float(m.group(2)))
                precs.append(float(m.group(3)))
                recalls.append(float(m.group(4)))

    n = min(len(losses), len(f1s), len(ious), len(recalls), len(precs))
    return EpochMetrics(
        train_loss=losses[:n],
        val_f1=f1s[:n],
        val_iou=ious[:n],
        val_recall=recalls[:n],
        val_precision=precs[:n],
    )


_EXP_LABELS = {
    1: "Exp 1: ResNet-34 (α=0.25)",
    2: "Exp 2: MiT-B2 (α=0.75)",
    3: "Exp 3: ResNet-34 (α=0.75)",
    4: "Exp 4: MiT-B2 + Domain Aug",
    6: "Exp 6: ResNet-34 + SAR IN",
    7: "Exp 7: SatlasPretrain Swin-V2-B",
}

_EXP_COLORS = {
    1: "#636EFA",  # blue
    2: "#EF553B",  # red
    3: "#00CC96",  # green
    4: "#AB63FA",  # purple
    6: "#FF6692",  # pink
    7: "#FFA15A",  # orange
}

_EXP_STYLES = {
    1: "-",
    2: "-",
    3: "-",
    4: "--",
    6: "-",
    7: "-",
}


def discover_experiments(outputs_dir: Path) -> dict[int, Path]:
    """Find training logs for all available experiments.

    Args:
        outputs_dir: Root outputs directory (e.g. ``outputs/``).

    Returns:
        Mapping from experiment number to the training log path.
    """
    experiments = {}
    for exp_dir in sorted(outputs_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp"):
            continue
        try:
            exp_num = int(exp_dir.name[3:])
        except ValueError:
            continue
        log_dir = exp_dir / "logs"
        if not log_dir.exists():
            continue
        train_logs = sorted(log_dir.glob("train_*.log"))
        if train_logs:
            experiments[exp_num] = train_logs[-1]  # most recent
    return experiments


def plot_training_curves(
    experiments: dict[int, EpochMetrics],
    output_path: Path,
) -> None:
    """Generate a 2×2 comparison chart of training metrics.

    Panels: Training Loss | Validation F1 | Validation IoU | Validation Recall.

    Args:
        experiments: Mapping from experiment number to parsed metrics.
        output_path: Destination path for the saved PNG.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Training Dynamics Across All Experiments",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    panels = [
        (axes[0, 0], "train_loss", "Training Loss", "Loss"),
        (axes[0, 1], "val_f1", "Validation F1 Score", "F1"),
        (axes[1, 0], "val_iou", "Validation IoU", "IoU"),
        (axes[1, 1], "val_recall", "Validation Recall", "Recall"),
    ]

    for ax, attr, title, ylabel in panels:
        for exp_num in sorted(experiments):
            metrics = experiments[exp_num]
            values = getattr(metrics, attr)
            if not values:
                continue
            epochs = np.arange(1, len(values) + 1)
            label = _EXP_LABELS.get(exp_num, f"Exp {exp_num}")
            color = _EXP_COLORS.get(exp_num, None)
            style = _EXP_STYLES.get(exp_num, "-")
            ax.plot(
                epochs,
                values,
                label=label,
                color=color,
                linestyle=style,
                linewidth=1.5,
                alpha=0.85,
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Mark phase transition for Exp 7 (epoch 10→11)
        if 7 in experiments and len(getattr(experiments[7], attr)) > 10:
            ax.axvline(
                x=10.5, color=_EXP_COLORS[7], linestyle=":", alpha=0.5, linewidth=1
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {output_path}")


def plot_convergence_comparison(
    experiments: dict[int, EpochMetrics],
    output_path: Path,
) -> None:
    """Generate a focused F1 convergence plot with best-F1 markers.

    Args:
        experiments: Mapping from experiment number to parsed metrics.
        output_path: Destination path for the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for exp_num in sorted(experiments):
        metrics = experiments[exp_num]
        if not metrics.val_f1:
            continue
        epochs = np.arange(1, len(metrics.val_f1) + 1)
        label = _EXP_LABELS.get(exp_num, f"Exp {exp_num}")
        color = _EXP_COLORS.get(exp_num, None)
        style = _EXP_STYLES.get(exp_num, "-")

        ax.plot(
            epochs,
            metrics.val_f1,
            label=label,
            color=color,
            linestyle=style,
            linewidth=1.8,
            alpha=0.85,
        )

        # Mark best F1
        best_idx = int(np.argmax(metrics.val_f1))
        best_val = metrics.val_f1[best_idx]
        ax.scatter(
            best_idx + 1,
            best_val,
            color=color,
            s=60,
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
        )
        ax.annotate(
            f"{best_val:.3f}\n(ep {best_idx + 1})",
            xy=(best_idx + 1, best_val),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=7,
            color=color,
            fontweight="bold",
        )

    ax.set_title("Validation F1 Convergence", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Phase transition line for Exp 7
    if 7 in experiments:
        ax.axvline(
            x=10.5,
            color=_EXP_COLORS[7],
            linestyle=":",
            alpha=0.4,
            linewidth=1,
            label="Exp 7 Phase 2 start",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Convergence plot saved to {output_path}")


def main() -> None:
    """Discover experiments and generate comparison charts."""
    parser = argparse.ArgumentParser(
        description="Generate combined training metric charts for all experiments.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Root outputs directory (default: outputs/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="report/figures/training_curves.png",
        help="Output path for the main 2×2 chart.",
    )
    parser.add_argument(
        "--convergence",
        type=str,
        default="report/figures/f1_convergence.png",
        help="Output path for the F1 convergence plot.",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"Error: outputs directory not found: {outputs_dir}")
        return

    print("Discovering experiments...")
    log_paths = discover_experiments(outputs_dir)

    if not log_paths:
        print("No training logs found. Ensure logs exist in outputs/expN/logs/.")
        return

    print(f"Found {len(log_paths)} experiments: {sorted(log_paths.keys())}")

    # Parse all logs
    parsed = {}
    for exp_num, log_path in sorted(log_paths.items()):
        metrics = parse_training_log(log_path)
        if metrics.train_loss:
            parsed[exp_num] = metrics
            print(
                f"  Exp {exp_num}: {len(metrics.train_loss)} epochs "
                f"(best F1: {max(metrics.val_f1):.4f})"
            )
        else:
            print(f"  Exp {exp_num}: no epoch data found in {log_path.name}")

    if not parsed:
        print("No parseable training data found.")
        return

    # Generate charts
    plot_training_curves(parsed, Path(args.output))
    plot_convergence_comparison(parsed, Path(args.convergence))

    print("\nDone. Add these figures to the report:")
    print(f"  {args.output}")
    print(f"  {args.convergence}")


if __name__ == "__main__":
    main()
