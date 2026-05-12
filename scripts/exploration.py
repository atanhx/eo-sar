"""
Exploratory data analysis for the EO-SAR change detection dataset.

Prints per-split statistics (file counts, shapes, class distributions, scene
breakdown) and generates a side-by-side visualisation panel for a few training
samples.

Usage::

    python scripts/exploration.py
    python scripts/exploration.py --data_path /path/to/data
    python scripts/exploration.py --n_samples 8
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

SPLITS = ["train", "val", "test"]
CLASS_NAMES = {0: "Background", 1: "Intact", 2: "Damaged", 3: "Destroyed"}


# some util funcs
def remap_labels(mask: np.ndarray) -> np.ndarray:
    """Collapse 4-class annotations to binary: 0=No-Change, 1=Change.

    Mapping:
        0 (Background) → 0,  1 (Intact) → 0,
        2 (Damaged)    → 1,  3 (Destroyed) → 1.

    Args:
        mask: Integer array with values in {0, 1, 2, 3}.

    Returns:
        Binary ``uint8`` array with values in {0, 1}.
    """
    remapped = np.zeros_like(mask, dtype=np.uint8)
    remapped[mask == 2] = 1
    remapped[mask == 3] = 1
    return remapped


def load_tif(path: Path) -> np.ndarray:
    """Read a GeoTIFF file into a NumPy array.

    Args:
        path: Filesystem path to the ``.tif`` file.

    Returns:
        Array with the image contents.
    """
    return tiff.imread(str(path))


def _normalize_display(arr: np.ndarray) -> np.ndarray:
    """Percentile-based contrast stretch for visualisation.

    Args:
        arr: Input image array (any dtype).

    Returns:
        Float array in [0, 1] after 2nd–98th percentile stretch.
    """
    arr = arr.astype(float)
    low, high = np.percentile(arr, 2), np.percentile(arr, 98)
    if high > low:
        arr = np.clip((arr - low) / (high - low), 0, 1)
    return arr


def _extract_scene_id(filename: str) -> str:
    """Extract the scene identifier from a dataset filename.

    Expected format: ``scene_XX_YYYYYY_building_damage.tif``.

    Args:
        filename: Stem or full name of the file.

    Returns:
        Scene string like ``"scene_01"`` or ``"unknown"`` if not parseable.
    """
    parts = Path(filename).stem.split("_")
    if len(parts) >= 2 and parts[0] == "scene":
        return f"scene_{parts[1]}"
    return "unknown"


# exploring the splits
def explore_split(split_name: str, root: str) -> None:
    """Print dataset statistics for a single split.

    Reports file counts, sample shapes/dtypes, raw 4-class distribution,
    binary change distribution with imbalance ratio, and per-scene breakdown.

    Args:
        split_name: One of ``"train"``, ``"val"``, ``"test"``.
        root: Dataset root directory.
    """
    print(f"\n{'=' * 60}")
    print(f"SPLIT: {split_name.upper()}")
    print(f"{'=' * 60}")

    split_dir = Path(root) / split_name
    pre_dir = split_dir / "pre-event"
    post_dir = split_dir / "post-event"
    target_dir = split_dir / "target"

    pre_files = sorted(pre_dir.glob("*.tif"))
    post_files = sorted(post_dir.glob("*.tif"))
    mask_files = sorted(target_dir.glob("*.tif"))

    print(f"Pre-event files:  {len(pre_files)}")
    print(f"Post-event files: {len(post_files)}")
    print(f"Target files:     {len(mask_files)}")

    if not pre_files:
        print("  (empty split — skipping)")
        return

    # some samples analysis
    sample_pre = load_tif(pre_files[0])
    sample_post = load_tif(post_files[0])
    sample_mask = load_tif(mask_files[0])

    print(f"\nSample pre-event shape:  {sample_pre.shape}  dtype: {sample_pre.dtype}")
    print(f"Sample post-event shape: {sample_post.shape}  dtype: {sample_post.dtype}")
    print(f"Sample mask shape:       {sample_mask.shape}  dtype: {sample_mask.dtype}")
    print(f"Unique mask values:      {np.unique(sample_mask)}")
    print(f"Pre-event value range:   [{sample_pre.min()}, {sample_pre.max()}]")
    print(f"Post-event value range:  [{sample_post.min()}, {sample_post.max()}]")

    # total class distribution
    total_pixels, change_pixels = 0, 0
    raw_counts: Counter = Counter()
    scene_counts: Counter = Counter()
    scene_change_pixels: Counter = Counter()
    scene_total_pixels: Counter = Counter()

    for mask_path in mask_files:
        mask = load_tif(mask_path)
        scene_id = _extract_scene_id(mask_path.name)
        scene_counts[scene_id] += 1

        for v in np.unique(mask):
            raw_counts[int(v)] += int(np.sum(mask == v))

        remapped = remap_labels(mask)
        n_change = int(np.sum(remapped == 1))
        total_pixels += remapped.size
        change_pixels += n_change
        scene_change_pixels[scene_id] += n_change
        scene_total_pixels[scene_id] += remapped.size

    print("\nRaw class distribution:")
    total_raw = sum(raw_counts.values())
    for cls_val, count in sorted(raw_counts.items()):
        pct = 100 * count / total_raw if total_raw > 0 else 0
        print(f"  {CLASS_NAMES.get(cls_val, str(cls_val))}: {count:,} ({pct:.2f}%)")

    no_change = total_pixels - change_pixels
    print("\nAfter binary remapping:")
    print(f"  No-Change: {no_change:,} ({100 * no_change / total_pixels:.2f}%)")
    print(f"  Change:    {change_pixels:,} ({100 * change_pixels / total_pixels:.2f}%)")
    if change_pixels > 0:
        print(f"  Imbalance ratio: {no_change / change_pixels:.1f} : 1")
    else:
        print("  Imbalance ratio: ∞ (no change pixels)")

    print("\nPer-scene breakdown:")
    print(f"  {'Scene':<12} {'Samples':>8} {'Change %':>10} {'Pixels (change)':>18}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 10} {'-' * 18}")
    for scene_id in sorted(scene_counts):
        n = scene_counts[scene_id]
        total = scene_total_pixels[scene_id]
        change = scene_change_pixels[scene_id]
        pct = 100 * change / total if total > 0 else 0
        print(f"  {scene_id:<12} {n:>8} {pct:>9.2f}% {change:>12,} / {total:>12,}")


# N samples visualization for splits
def visualize_samples(root: str, n_samples: int = 4, output_dir: str = ".") -> None:
    """Generate a figure showing EO, SAR, raw mask, and binary mask side by side.

    Args:
        root: Dataset root directory.
        n_samples: Number of training samples to display.
        output_dir: Directory to save the output PNG.
    """
    pre_dir = Path(root) / "train" / "pre-event"
    post_dir = Path(root) / "train" / "post-event"
    target_dir = Path(root) / "train" / "target"

    pre_files = sorted(pre_dir.glob("*.tif"))[:n_samples]
    if not pre_files:
        print("No training files found — skipping visualisation.")
        return

    n = len(pre_files)
    fig, axes = plt.subplots(n, 4, figsize=(18, 4 * n), squeeze=False)

    for i, pre_path in enumerate(pre_files):
        pre_img = load_tif(pre_path)
        post_img = load_tif(post_dir / pre_path.name)
        mask_raw = load_tif(target_dir / pre_path.name)
        mask_bin = remap_labels(mask_raw)

        pre_display = _normalize_display(pre_img)
        post_display = _normalize_display(post_img)

        if pre_display.ndim == 3 and pre_display.shape[2] > 3:
            pre_display = pre_display[:, :, :3]

        axes[i, 0].imshow(pre_display, cmap=None if pre_display.ndim == 3 else "gray")
        axes[i, 0].set_title(f"Pre-event EO\n{pre_img.shape}", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(post_display, cmap="gray")
        axes[i, 1].set_title(f"Post-event SAR\n{post_img.shape}", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(mask_raw, cmap="tab10", vmin=0, vmax=3)
        axes[i, 2].set_title(f"Raw Mask\nValues: {np.unique(mask_raw)}", fontsize=10)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(mask_bin, cmap="gray")
        axes[i, 3].set_title(
            f"Binary Mask\nChange: {100 * mask_bin.mean():.1f}%",
            fontsize=10,
        )
        axes[i, 3].axis("off")

    plt.suptitle(
        "Data Exploration: EO Pre-Event vs SAR Post-Event", fontsize=14, y=1.01
    )
    plt.tight_layout()

    output_path = Path(output_dir) / "exploration.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nVisualisation saved to {output_path}")


def main() -> None:
    """Parse CLI arguments and run exploratory analysis."""
    parser = argparse.ArgumentParser(
        description="Exploratory data analysis for the EO-SAR change detection dataset.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/",
        help="Path to the dataset root directory (default: ./data/).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of training samples to visualise (default: 4).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the exploration.png figure (default: cwd).",
    )
    args = parser.parse_args()

    for split in SPLITS:
        split_path = Path(args.data_path) / split
        if split_path.exists():
            explore_split(split, args.data_path)
        else:
            print(f"\nSplit '{split}' not found at {split_path} — skipping.")

    visualize_samples(
        args.data_path, n_samples=args.n_samples, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
