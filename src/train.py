"""Training loop for binary change detection."""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import get_dataloader
from metrics import aggregate_metrics, compute_metrics
from model import CombinedLoss, build_model, build_satlas_model
from utils import Tee, run_timestamp


def set_seed(seed: int) -> None:
    """Set random seeds across all libraries for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: AdamW,
    epoch: int,
    metric_val: float,
    path: Path,
) -> None:
    """Persist model and optimiser state to disk.

    Args:
        model: The model whose weights are saved.
        optimizer: The optimiser whose state is saved.
        epoch: Current epoch number.
        metric_val: Validation F1 at this checkpoint.
        path: Destination file path.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "best_f1": metric_val,
        },
        path,
    )
    print(f"  Checkpoint saved to {path}  (F1: {metric_val:.4f})")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: AdamW,
    device: torch.device,
    epoch: int,
    accum_steps: int = 1,
) -> tuple[float, float]:
    """Run a single training epoch.

    Args:
        model: The segmentation model.
        loader: Training data loader.
        loss_fn: Combined loss function.
        optimizer: AdamW optimiser.
        device: Compute device.
        epoch: Current epoch number (for logging).
        accum_steps: Gradient accumulation steps (effective batch =
            ``batch_size × accum_steps``).  Default 1 (no accumulation).

    Returns:
        Tuple of ``(average_loss, epoch_wall_time_seconds)``.
    """
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    epoch_start = time.time()
    optimizer.zero_grad()

    for batch_idx, (images, masks) in enumerate(loader):
        batch_start = time.time()
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = loss_fn(logits, masks) / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == n_batches:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps  # log un-scaled loss
        batch_time = time.time() - batch_start

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
            elapsed = time.time() - epoch_start
            print(
                f"  Epoch {epoch} [{batch_idx + 1}/{n_batches}]  "
                f"Loss: {loss.item() * accum_steps:.4f}  "
                f"Avg: {total_loss / (batch_idx + 1):.4f}  "
                f"Batch time: {batch_time:.2f}s  "
                f"Elapsed: {elapsed:.0f}s"
            )

    return total_loss / n_batches, time.time() - epoch_start


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on a validation split using global metric accumulation.

    Args:
        model: The segmentation model (set to eval mode internally).
        loader: Validation data loader.
        device: Compute device.

    Returns:
        Aggregated metrics dictionary.
    """
    model.eval()
    batch_metrics = []
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        batch_metrics.append(compute_metrics(logits, masks))
    return aggregate_metrics(batch_metrics)


def train(config_path: str) -> None:
    """Execute the full training loop from a YAML configuration file.

    Args:
        config_path: Path to the experiment configuration YAML.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(cfg["paths"]["log_dir"])
    log_path = log_dir / f"train_{run_timestamp()}.log"

    with Tee(log_path):
        print(f"Device:    {device}")
        print(f"Config:    {config_path}")
        print(f"Log:       {log_path}")
        print(f"Started:   {time.strftime('%Y-%m-%d %H:%M:%S')}")

        print("\nLoading datasets...")
        train_loader = get_dataloader(
            root=cfg["data"]["root"],
            split="train",
            batch_size=cfg["training"]["batch_size"],
            patch_size=cfg["data"]["patch_size"],
            num_workers=cfg["data"]["num_workers"],
            use_weighted_sampler=cfg["data"].get("use_weighted_sampler", False),
            no_change_sample_weight=cfg["data"].get("no_change_sample_weight", 0.05),
            sar_noise_prob=cfg["augmentation"].get("sar_noise_prob", 0.0),
            sar_noise_scale=cfg["augmentation"].get("sar_noise_scale", 0.2),
            use_rotate=cfg["augmentation"].get("use_rotate", False),
            sar_instance_norm=cfg["augmentation"].get("sar_instance_norm", False),
            channel_dropout_prob=cfg["augmentation"].get("channel_dropout_prob", 0.0),
        )
        val_loader = get_dataloader(
            root=cfg["data"]["root"],
            split="val",
            batch_size=cfg["training"]["batch_size"],
            patch_size=cfg["data"]["patch_size"],
            num_workers=cfg["data"]["num_workers"],
        )

        print("\nBuilding model...")
        is_satlas = cfg["model"]["encoder_name"].startswith("satlas_")

        if is_satlas:
            # Experiment 7: SatlasPretrain Swin-B
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
            freeze_epochs = cfg["training"].get("freeze_epochs", 10)
            unfreeze_lr = cfg["training"].get("unfreeze_lr", 1e-5)
            model.freeze_backbone(True)
            print(f"  Phase 1: backbone frozen for {freeze_epochs} epochs")
            print(f"  Phase 2: will unfreeze at LR={unfreeze_lr}")
        else:
            model = build_model(
                encoder_name=cfg["model"]["encoder_name"],
                encoder_weights=cfg["model"]["encoder_weights"],
                in_channels=cfg["model"]["in_channels"],
                classes=cfg["model"]["classes"],
            ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

        loss_fn = CombinedLoss(
            dice_weight=cfg["training"]["dice_weight"],
            focal_weight=cfg["training"]["focal_weight"],
            focal_alpha=cfg["training"]["focal_alpha"],
            focal_gamma=cfg["training"]["focal_gamma"],
        )
        optimizer = AdamW(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg["training"]["epochs"],
            eta_min=1e-6,
        )

        best_f1 = 0.0
        best_epoch = 0
        epochs = cfg["training"]["epochs"]
        accum_steps = cfg["training"].get("gradient_accumulation", 1)
        epoch_times: list[float] = []
        training_start = time.time()

        eff_batch = cfg["training"]["batch_size"] * accum_steps
        print(
            f"\nEffective batch size: {cfg['training']['batch_size']} × {accum_steps} = {eff_batch}"
        )

        print(f"\nStarting training for {epochs} epochs...")
        if is_satlas:
            print(
                f"  2-phase schedule: frozen[1-{freeze_epochs}] → unfrozen[{freeze_epochs + 1}-{epochs}]"
            )
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            # Phase 2 unfreezing for SatlasPretrain backbone (Experiment 7)
            if is_satlas and epoch == freeze_epochs + 1:
                print(f"\n{'=' * 60}")
                print(f"PHASE 2: Unfreezing backbone at epoch {epoch}")
                print(f"{'=' * 60}")
                model.freeze_backbone(False)
                # Reset optimizer with lower LR for backbone
                optimizer = AdamW(
                    [
                        {
                            "params": model.channel_adapter.parameters(),
                            "lr": unfreeze_lr,
                        },
                        {"params": model.backbone.parameters(), "lr": unfreeze_lr},
                        {"params": model.fpn.parameters(), "lr": unfreeze_lr},
                        {"params": model.decoder.parameters(), "lr": unfreeze_lr * 5},
                    ],
                    weight_decay=cfg["training"]["weight_decay"],
                )
                remaining = epochs - freeze_epochs
                scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-7)
                print(
                    f"  New optimizer: backbone LR={unfreeze_lr}, decoder LR={unfreeze_lr * 5}"
                )
                print(f"  Cosine schedule over {remaining} remaining epochs")

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch}/{epochs}  LR: {current_lr:.6f}")

            train_loss, epoch_time = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                epoch,
                accum_steps=accum_steps,
            )
            epoch_times.append(epoch_time)

            val_start = time.time()
            val_metrics = validate(model, val_loader, device)
            val_time = time.time() - val_start

            scheduler.step()

            print(f"\n  Train Loss:    {train_loss:.4f}")
            print(
                f"  Val IoU:       {val_metrics['iou']:.4f}  "
                f"F1: {val_metrics['f1']:.4f}  "
                f"Precision: {val_metrics['precision']:.4f}  "
                f"Recall: {val_metrics['recall']:.4f}"
            )
            print(
                f"  Epoch time:    {epoch_time:.1f}s  "
                f"Val time: {val_time:.1f}s  "
                f"Avg epoch: {np.mean(epoch_times):.1f}s"
            )

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_f1,
                    checkpoint_dir / "best_checkpoint.pth",
                )

            if epoch % 10 == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_metrics["f1"],
                    checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                )

        total_time = time.time() - training_start

        print(f"\n{'=' * 60}")
        print("TRAINING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Best validation F1:   {best_f1:.4f} at epoch {best_epoch}")
        print(
            f"Total training time:  {total_time / 3600:.2f} hours  ({total_time:.0f} seconds)"
        )
        print(f"Avg time per epoch:   {np.mean(epoch_times):.1f}s")
        print(f"Device:               {device}")
        print(f"Encoder:              {cfg['model']['encoder_name']}")
        print(f"Batch size:           {cfg['training']['batch_size']}")
        print(f"Best checkpoint:      {checkpoint_dir}/best_checkpoint.pth")
        print("\nPer-epoch times (seconds):")
        for i, t in enumerate(epoch_times, 1):
            print(f"  Epoch {i:3d}: {t:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train change detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_exp3.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    train(args.config)
