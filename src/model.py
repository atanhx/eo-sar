"""Model architectures and loss functions for binary change detection."""

import collections
import logging
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

logger = logging.getLogger(__name__)


def build_model(
    encoder_name: str = "mit_b2",
    encoder_weights: str = "imagenet",
    in_channels: int = 4,
    classes: int = 1,
) -> nn.Module:
    """Build a UNet segmentation model with the specified encoder backbone.

    The model uses early fusion: 3-channel EO (pre-event) and 1-channel SAR
    (post-event) are concatenated along the channel axis before encoding.

    Args:
        encoder_name: Encoder backbone name compatible with
            ``segmentation_models_pytorch`` (e.g. ``"resnet34"``, ``"mit_b2"``).
        encoder_weights: Pretrained weight identifier or ``None``.
        in_channels: Number of input channels (3 EO + 1 SAR = 4).
        classes: Number of output classes (1 for binary segmentation).

    Returns:
        A ``torch.nn.Module`` producing raw logits of shape ``(B, 1, H, W)``.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )


class CombinedLoss(nn.Module):
    """Weighted sum of Dice loss and Focal loss for imbalanced segmentation.

    Dice loss directly optimises mask overlap and is insensitive to class
    imbalance.  Focal loss down-weights confident predictions via the
    modulating factor ``(1 - p_t)^gamma``, concentrating gradient on hard
    examples.

    Args:
        dice_weight: Scalar multiplier for Dice loss.
        focal_weight: Scalar multiplier for Focal loss.
        focal_alpha: Class-weight for the positive class in Focal loss.
        focal_gamma: Focusing parameter for Focal loss.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.dice_loss = smp.losses.DiceLoss(
            mode="binary",
            from_logits=True,
        )
        self.focal_loss = smp.losses.FocalLoss(
            mode="binary",
            alpha=focal_alpha,
            gamma=focal_gamma,
            normalized=False,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss from raw logits and binary targets.

        Args:
            logits: Raw model output of shape ``(B, 1, H, W)``.
            targets: Ground-truth mask of shape ``(B, H, W)`` with values 0/1.

        Returns:
            Scalar loss tensor.
        """
        targets_float = targets.float()
        dice = self.dice_loss(logits, targets_float)
        focal = self.focal_loss(logits, targets_float)
        return self.dice_weight * dice + self.focal_weight * focal


# SatlasPretrain Swin-B. This is for Experiment7.
# Architecture:
# [1×1 Conv: 4ch > 3ch] > [Swin-V2-B backbone] > [FPN 128ch] > [Decoder] > logits


# SatlasPretrain checkpoints
_SATLAS_WEIGHTS = {
    "Sentinel2_SwinB_SI_RGB": (
        "https://huggingface.co/allenai/satlas-pretrain/resolve/main/"
        "sentinel2_swinb_si_rgb.pth"
    ),
    "Aerial_SwinB_SI": (
        "https://huggingface.co/allenai/satlas-pretrain/resolve/main/"
        "aerial_swinb_si.pth"
    ),
}


def _download_satlas_weights(
    identifier: str,
    cache_dir: str = "weights_cache",
) -> dict:
    """Download SatlasPretrain weights from HuggingFace and cache locally.

    Args:
        identifier: Key into :data:`_SATLAS_WEIGHTS`, e.g.
            ``"Sentinel2_SwinB_SI_RGB"``.
        cache_dir: Local directory for caching downloaded checkpoints.

    Returns:
        State-dict loaded from the checkpoint file.
    """
    if identifier not in _SATLAS_WEIGHTS:
        raise ValueError(
            f"Unknown SatlasPretrain identifier '{identifier}'. "
            f"Available: {list(_SATLAS_WEIGHTS)}"
        )

    url = _SATLAS_WEIGHTS[identifier]
    cache_path = Path(cache_dir) / f"{identifier.lower()}.pth"

    if cache_path.exists():
        logger.info("Loading cached SatlasPretrain weights from %s", cache_path)
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    logger.info("Downloading SatlasPretrain weights: %s", identifier)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = torch.hub.load_state_dict_from_url(
        url,
        model_dir=str(cache_path.parent),
        map_location="cpu",
        file_name=cache_path.name,
    )
    return state_dict


def _extract_backbone_state_dict(full_state: dict) -> dict:
    """Extract backbone-only keys from a SatlasPretrain checkpoint.

    The checkpoint stores keys like ``backbone.backbone.features.0.0.weight``.
    For a single-image Swin model the desired prefix depth is one
    ``backbone.`` — we strip the outer prefix so the result has keys like
    ``backbone.features.0.0.weight`` which matches
    :class:`SwinBackbone`'s ``self.backbone = swin_v2_b()``.
    """
    out = {}
    for key, val in full_state.items():
        if not key.startswith("backbone."):
            continue
        # Strip exactly one leading "backbone." prefix
        new_key = key[len("backbone.") :]
        out[new_key] = val
    return out


def _extract_fpn_state_dict(full_state: dict) -> dict:
    """Extract FPN keys from a SatlasPretrain checkpoint.

    Checkpoint prefix: ``intermediates.0.fpn.`` → strip to ``fpn.``.
    """
    out = {}
    prefix = "intermediates.0.fpn."
    for key, val in full_state.items():
        if not key.startswith(prefix):
            continue
        new_key = "fpn." + key[len(prefix) :]
        out[new_key] = val
    return out


class _SwinBackbone(nn.Module):
    """Swin-V2-B backbone producing 4-scale feature maps.

    Mirrors the SatlasPretrain ``SwinBackbone`` class exactly so that
    pretrained state dicts can be loaded without key surgery.
    """

    # Scale factors and channel counts for each output level
    out_channels = [
        [4, 128],  # 1/4  resolution
        [8, 256],  # 1/8  resolution
        [16, 512],  # 1/16 resolution
        [32, 1024],  # 1/32 resolution
    ]

    def __init__(self, num_channels: int = 3) -> None:
        super().__init__()
        self.backbone = torchvision.models.swin_v2_b()
        # Replace the stem conv to accept ``num_channels`` inputs
        stem_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            num_channels,
            stem_conv.out_channels,
            kernel_size=(4, 4),
            stride=(4, 4),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        # Return 4 scale levels matching the SatlasPretrain convention
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]


class _FPN(nn.Module):
    """Feature Pyramid Network producing 128-channel multi-scale features.

    Mirrors the SatlasPretrain ``FPN`` class so pretrained weights load
    directly.
    """

    def __init__(self, backbone_channels: list[list[int]]) -> None:
        super().__init__()
        out_ch = 128
        in_channels_list = [ch[1] for ch in backbone_channels]
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_ch,
        )
        self.out_channels = [[ch[0], out_ch] for ch in backbone_channels]

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        inp = collections.OrderedDict([(f"feat{i}", el) for i, el in enumerate(x)])
        output = self.fpn(inp)
        return list(output.values())


class _UNetDecoder(nn.Module):
    """Lightweight UNet-style decoder for 4-scale FPN features.

    Progressively upsamples and fuses feature maps from coarse to fine,
    producing a single-channel logit map at the FPN's highest resolution
    (1/4 of input).  A final bilinear upsample recovers full resolution.

    Args:
        fpn_channels: Number of channels at each FPN level (typically 128).
        classes: Number of output segmentation classes.
    """

    def __init__(self, fpn_channels: int = 128, classes: int = 1) -> None:
        super().__init__()
        # Lateral + upsample blocks (coarse → fine)
        self.up3 = self._up_block(fpn_channels, fpn_channels)
        self.up2 = self._up_block(fpn_channels * 2, fpn_channels)
        self.up1 = self._up_block(fpn_channels * 2, fpn_channels)
        self.up0 = self._up_block(fpn_channels * 2, fpn_channels)

        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels // 2, 3, padding=1),
            nn.BatchNorm2d(fpn_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels // 2, classes, 1),
        )

    @staticmethod
    def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        features: list[torch.Tensor],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Decode FPN features into a full-resolution logit map.

        Args:
            features: List of 4 FPN tensors ``[f0, f1, f2, f3]`` from
                finest (1/4) to coarsest (1/32).
            target_size: ``(H, W)`` of the original input for final upsample.

        Returns:
            Logit tensor of shape ``(B, classes, H, W)``.
        """
        f0, f1, f2, f3 = features  # 1/4, 1/8, 1/16, 1/32

        x = self.up3(f3)
        x = F.interpolate(x, size=f2.shape[2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, f2], dim=1))
        x = F.interpolate(x, size=f1.shape[2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, f1], dim=1))
        x = F.interpolate(x, size=f0.shape[2:], mode="bilinear", align_corners=False)
        x = self.up0(torch.cat([x, f0], dim=1))

        x = self.seg_head(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class SatlasUNet(nn.Module):
    """Swin-V2-B + FPN encoder with UNet decoder, initialised from SatlasPretrain.

    Designed for binary change detection with 4-channel early-fusion input
    (3 EO + 1 SAR).  A learned 1×1 convolution projects 4 channels down to
    the 3 channels expected by the pretrained Swin-V2-B stem.

    Supports 2-phase training:

    - **Phase 1:** ``freeze_backbone(True)`` — only the channel adapter and
      decoder are trainable.  Prevents the randomly-initialised decoder from
      corrupting the pretrained backbone.
    - **Phase 2:** ``freeze_backbone(False)`` — the full network is
      trainable at a lower learning rate.

    Args:
        satlas_identifier: SatlasPretrain checkpoint ID (e.g.
            ``"Sentinel2_SwinB_SI_RGB"``).
        in_channels: Number of input channels (4 for EO+SAR fusion).
        classes: Number of segmentation output classes.
        cache_dir: Directory for caching downloaded weights.
    """

    def __init__(
        self,
        satlas_identifier: str = "Sentinel2_SwinB_SI_RGB",
        in_channels: int = 4,
        classes: int = 1,
        cache_dir: str = "weights_cache",
    ) -> None:
        super().__init__()

        # Channel adapter: 4 > 3 via learned projection
        self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            nn.init.zeros_(self.channel_adapter.weight)
            for i in range(min(3, in_channels)):
                self.channel_adapter.weight[i, i] = 1.0

        # Swin-V2-B, 3-channel input
        self.backbone = _SwinBackbone(num_channels=3)

        self.fpn = _FPN(self.backbone.out_channels)

        self._load_pretrained(satlas_identifier, cache_dir)

        # Decoder (randomly initialised)
        fpn_ch = self.fpn.out_channels[0][1]  # 128
        self.decoder = _UNetDecoder(fpn_channels=fpn_ch, classes=classes)

    def _load_pretrained(self, identifier: str, cache_dir: str) -> None:
        """Load SatlasPretrain weights into backbone and FPN."""
        state_dict = _download_satlas_weights(identifier, cache_dir)

        # Backbone
        bb_state = _extract_backbone_state_dict(state_dict)
        missing, unexpected = self.backbone.load_state_dict(bb_state, strict=False)
        if missing:
            logger.warning("Backbone missing keys: %s", missing)
        if unexpected:
            logger.warning("Backbone unexpected keys: %s", unexpected)
        print(f"  Loaded SatlasPretrain backbone: {len(bb_state)} params")

        # FPN
        fpn_state = _extract_fpn_state_dict(state_dict)
        missing, unexpected = self.fpn.load_state_dict(fpn_state, strict=False)
        if missing:
            logger.warning("FPN missing keys: %s", missing)
        if unexpected:
            logger.warning("FPN unexpected keys: %s", unexpected)
        print(f"  Loaded SatlasPretrain FPN: {len(fpn_state)} params")

    def freeze_backbone(self, frozen: bool = True) -> None:
        """Freeze or unfreeze the backbone + FPN parameters.

        When frozen, only the channel adapter and decoder are trainable.
        Call with ``frozen=False`` to enter Phase 2 of training.

        Args:
            frozen: If ``True``, freeze backbone and FPN.
        """
        for param in self.backbone.parameters():
            param.requires_grad = not frozen
        for param in self.fpn.parameters():
            param.requires_grad = not frozen
        status = "FROZEN" if frozen else "UNFROZEN"
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Backbone + FPN {status}  (trainable params: {trainable:,})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: 4ch input → binary segmentation logits.

        Args:
            x: Input tensor of shape ``(B, 4, H, W)``.

        Returns:
            Logit tensor of shape ``(B, 1, H, W)``.
        """
        target_size = (x.shape[2], x.shape[3])
        x = self.channel_adapter(x)
        features = self.backbone(x)
        features = self.fpn(features)
        return self.decoder(features, target_size)


def build_satlas_model(
    satlas_identifier: str = "Sentinel2_SwinB_SI_RGB",
    in_channels: int = 4,
    classes: int = 1,
    cache_dir: str = "weights_cache",
) -> SatlasUNet:
    """Factory for :class:`SatlasUNet` with SatlasPretrain initialisation.

    This is the Experiment 7 equivalent of :func:`build_model`.

    Args:
        satlas_identifier: SatlasPretrain checkpoint ID.
        in_channels: Number of input channels.
        classes: Number of output classes.
        cache_dir: Weight download cache directory.

    Returns:
        :class:`SatlasUNet` instance with pretrained backbone + FPN and
        a randomly initialised decoder.
    """
    return SatlasUNet(
        satlas_identifier=satlas_identifier,
        in_channels=in_channels,
        classes=classes,
        cache_dir=cache_dir,
    )
