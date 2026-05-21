"""Data loading, augmentation, and change-aware sampling for EO-SAR pairs."""

from pathlib import Path

import albumentations as A
import numpy as np
import tifffile as tiff
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def remap_labels(mask: np.ndarray) -> np.ndarray:
    """Collapse 4-class annotations to binary change labels.

    Mapping:
        0 (Background) → 0,  1 (Intact) → 0,
        2 (Damaged)    → 1,  3 (Destroyed) → 1.

    Args:
        mask: Integer array with values in {0, 1, 2, 3}.

    Returns:
        Binary ``uint8`` array with values in {0, 1}.
    """
    binary = np.zeros_like(mask, dtype=np.uint8)
    binary[mask == 2] = 1
    binary[mask == 3] = 1
    return binary


def load_tif(path: Path) -> np.ndarray:
    """Read a GeoTIFF file into a NumPy array.

    Args:
        path: Filesystem path to the ``.tif`` file.

    Returns:
        Array with the image contents.
    """
    return tiff.imread(str(path))


def instance_norm_sar(sar: np.ndarray) -> np.ndarray:
    """Normalize SAR patch to zero mean and unit variance."""
    return (sar - sar.mean()) / (sar.std() + 1e-6)


def get_transforms(split: str, use_rotate: bool = False) -> A.Compose:
    """Build an Albumentations pipeline for the given split.

    Cropping is handled separately in ``__getitem__`` to support change-aware
    patch selection during training.

    Args:
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        use_rotate: If ``True``, add continuous ±45° rotation (train only).

    Returns:
        Composed transform that expects ``image``, ``sar``, and ``mask`` keys.
    """
    additional_targets = {"sar": "image"}

    if split == "train":
        base: list[A.BasicTransform | A.BaseCompose] = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
        if use_rotate:
            base.append(A.Rotate(limit=45, border_mode=0, fill=0, fill_mask=0, p=0.5))
        base += [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ty:ignore[invalid-argument-type]
            ToTensorV2(),
        ]
        return A.Compose(base, additional_targets=additional_targets)

    return A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ty:ignore[invalid-argument-type]
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def change_aware_crop(
    eo: np.ndarray,
    sar: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a patch centred on a randomly selected change pixel.

    If the mask contains change pixels, one is chosen uniformly and used as
    the crop centre with small spatial jitter.  Otherwise a purely random
    crop is returned.

    Args:
        eo: EO image array of shape ``(H, W, 3)``.
        sar: SAR image array of shape ``(H, W, 1)``.
        mask: Binary mask of shape ``(H, W)``.
        patch_size: Side length of the square crop.
        rng: NumPy random generator for reproducibility.

    Returns:
        Tuple of cropped ``(eo, sar, mask)`` arrays.
    """
    H, W = mask.shape
    change_coords = np.argwhere(mask == 1)

    if len(change_coords) > 0:
        center_idx = rng.integers(0, len(change_coords))
        cy, cx = change_coords[center_idx]

        jitter = patch_size // 8
        cy = int(cy) + int(rng.integers(-jitter, jitter + 1))
        cx = int(cx) + int(rng.integers(-jitter, jitter + 1))

        y1 = int(np.clip(cy - patch_size // 2, 0, H - patch_size))
        x1 = int(np.clip(cx - patch_size // 2, 0, W - patch_size))
    else:
        y1 = int(rng.integers(0, H - patch_size + 1))
        x1 = int(rng.integers(0, W - patch_size + 1))

    y2 = y1 + patch_size
    x2 = x1 + patch_size
    return eo[y1:y2, x1:x2], sar[y1:y2, x1:x2], mask[y1:y2, x1:x2]


class ChangeDetectionDataset(Dataset):
    """PyTorch dataset for co-registered EO-SAR change detection pairs.

    Each sample returns a 4-channel tensor (3 EO + 1 SAR) and a binary mask.
    During training, patches are extracted via change-aware cropping.

    Args:
        root: Path to the dataset root containing split subdirectories.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        patch_size: Side length of the training crop.
        no_change_sample_weight: Sampling weight for images without change
            pixels (used by :class:`WeightedRandomSampler`).
        sar_noise_prob: Probability of applying multiplicative SAR noise.
        sar_noise_scale: Maximum deviation of the SAR noise scale factor
            from 1.0 (sampled from ``Uniform[1 - s, 1 + s]``).
        use_rotate: Enable continuous ±45° rotation augmentation.
        sar_instance_norm: If ``True``, normalise each SAR patch to zero mean
            and unit variance instead of simple ``/255`` scaling.  Removes
            scene-level brightness offsets that cause SAR domain shift.
        channel_dropout_prob: Probability of zeroing out one entire modality
            during training.  Each application randomly zeros EO channels
            (p/2) or the SAR channel (p/2), forcing single-modality robustness.
    """

    def __init__(
        self,
        root: str,
        split: str,
        patch_size: int = 512,
        no_change_sample_weight: float = 0.05,
        sar_noise_prob: float = 0.0,
        sar_noise_scale: float = 0.2,
        use_rotate: bool = False,
        sar_instance_norm: bool = False,
        channel_dropout_prob: float = 0.0,
    ):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self._no_change_weight = no_change_sample_weight
        self._sar_noise_prob = sar_noise_prob
        self._sar_noise_scale = sar_noise_scale
        self._sar_instance_norm = sar_instance_norm
        self._channel_dropout_prob = channel_dropout_prob
        self.transform = get_transforms(split, use_rotate=use_rotate)
        self.rng = None

        split_dir = self.root / split
        self.pre_dir = split_dir / "pre-event"
        self.post_dir = split_dir / "post-event"
        self.target_dir = split_dir / "target"

        all_stems = sorted([f.stem for f in self.pre_dir.glob("*.tif")])

        print(f"[{split.upper()}] Scanning {len(all_stems)} masks for change pixels...")
        self.file_stems: list[str] = []
        self.has_change: list[bool] = []

        for stem in all_stems:
            mask = remap_labels(load_tif(self.target_dir / f"{stem}.tif"))
            self.file_stems.append(stem)
            self.has_change.append(bool(mask.max() > 0))

        n_change = sum(self.has_change)
        n_no_change = len(self.has_change) - n_change
        print(
            f"[{split.upper()}] Found {len(self.file_stems)} samples: "
            f"{n_change} with change, {n_no_change} no-change"
        )

    def get_sample_weights(self) -> torch.Tensor:
        """Return per-sample weights for :class:`WeightedRandomSampler`.

        Change-containing images receive weight 1.0; purely no-change images
        receive a reduced weight to maintain some negative-scene exposure.

        Returns:
            Float tensor of length ``len(self)``.
        """
        weights = [
            1.0 if has_chg else self._no_change_weight for has_chg in self.has_change
        ]
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.file_stems)

    def _get_rng(self) -> np.random.Generator:
        """Create a per-worker random generator for reproducible sampling."""
        worker_info = torch.utils.data.get_worker_info()
        seed = 42 if worker_info is None else 42 + worker_info.id
        return np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # ty:ignore[invalid-method-override]
        if self.rng is None:
            self.rng = self._get_rng()

        stem = self.file_stems[idx]

        eo = load_tif(self.pre_dir / f"{stem}.tif")
        sar = load_tif(self.post_dir / f"{stem}.tif")
        if sar.ndim == 2:
            sar = sar[:, :, np.newaxis]
        sar = sar.astype(np.float32)  # keep uint8 range; normalisation applied below
        mask = remap_labels(load_tif(self.target_dir / f"{stem}.tif"))

        if self.split == "train":
            eo, sar, mask = change_aware_crop(eo, sar, mask, self.patch_size, self.rng)

        # SAR Normalisation and noise augmentation (train only)
        if self._sar_instance_norm:
            sar = instance_norm_sar(sar / 255.0)  # scale first, then IN
        else:
            sar = sar / 255.0

        if (
            self.split == "train"
            and self._sar_noise_prob > 0
            and not self._sar_instance_norm
        ):
            # Multiplicative noise is calibrated for [0, 1] scaled SAR.
            # Skip when instance normalisation is active (values are z-scored).
            if self.rng.random() < self._sar_noise_prob:
                scale = float(
                    self.rng.uniform(
                        1.0 - self._sar_noise_scale,
                        1.0 + self._sar_noise_scale,
                    )
                )
                sar = np.clip(sar * scale, 0.0, 1.0)

        augmented = self.transform(image=eo, sar=sar, mask=mask)
        eo_t = augmented["image"]
        sar_t = augmented["sar"]
        image_pair = torch.cat([eo_t, sar_t], dim=0)

        # Channel dropout (train only)
        if self.split == "train" and self._channel_dropout_prob > 0:
            r = self.rng.random()
            half_p = self._channel_dropout_prob / 2.0
            if r < half_p:
                image_pair[:3] = 0.0  # drop EO
            elif r < self._channel_dropout_prob:
                image_pair[3:] = 0.0  # drop SAR

        mask_t = augmented["mask"].long()
        return image_pair, mask_t


def get_dataloader(
    root: str,
    split: str,
    batch_size: int,
    patch_size: int,
    num_workers: int,
    shuffle: bool | None = None,
    use_weighted_sampler: bool = False,
    no_change_sample_weight: float = 0.05,
    sar_noise_prob: float = 0.0,
    sar_noise_scale: float = 0.2,
    use_rotate: bool = False,
    sar_instance_norm: bool = False,
    channel_dropout_prob: float = 0.0,
) -> DataLoader:
    """Construct a :class:`DataLoader` for a dataset split.

    Args:
        root: Dataset root directory.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        batch_size: Batch size.
        patch_size: Training crop size (ignored for val/test).
        num_workers: Number of data-loading workers.
        shuffle: Whether to shuffle; defaults to ``True`` for train.
        use_weighted_sampler: Enable :class:`WeightedRandomSampler` to
            oversample change-containing images.
        no_change_sample_weight: Weight for no-change images when using
            the weighted sampler.
        sar_noise_prob: Probability of SAR multiplicative noise augmentation.
        sar_noise_scale: Scale range for SAR noise.
        use_rotate: Enable continuous rotation augmentation.
        sar_instance_norm: Enable per-image instance normalisation on the SAR
            channel to remove scene-level brightness offsets.
        channel_dropout_prob: Probability of dropping one modality per sample
            during training to encourage single-modality robustness.

    Returns:
        Configured :class:`DataLoader`.
    """
    dataset = ChangeDetectionDataset(
        root=root,
        split=split,
        patch_size=patch_size,
        no_change_sample_weight=no_change_sample_weight,
        sar_noise_prob=sar_noise_prob,
        sar_noise_scale=sar_noise_scale,
        use_rotate=use_rotate,
        sar_instance_norm=sar_instance_norm,
        channel_dropout_prob=channel_dropout_prob,
    )

    if shuffle is None:
        shuffle = split == "train"

    sampler = None
    if use_weighted_sampler and split == "train":
        weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights,  # ty:ignore[invalid-argument-type]
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
