"""CenterNet-style gate detector CNN.

Lightweight architecture using MobileNetV3 backbone for real-time
gate detection. Outputs center heatmap, size regression, and corner offsets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class GateCenterNet(nn.Module):
    """CenterNet-style single-class gate detector.

    Architecture:
    - Backbone: MobileNetV3-Small features (576ch, H/32)
    - Neck: 2x upsample with convs → 64ch at H/8
    - Heads: heatmap (1ch), size (2ch), corners (8ch)
    """

    def __init__(self, input_size: tuple[int, int] = (320, 320)) -> None:
        super().__init__()
        self.input_size = input_size

        # Backbone
        backbone = torchvision.models.mobilenet_v3_small(weights=None)
        self.features = backbone.features  # output: (B, 576, H/32, W/32)

        # Neck: upsample 4x (H/32 → H/8)
        self.up1 = nn.Sequential(
            nn.Conv2d(576, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Heads
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )
        self.size_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),
        )
        self.corners_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, 3, H, W) normalized input images

        Returns:
            Dict with heatmap (B,1,H/8,W/8), size (B,2,H/8,W/8),
            corners (B,8,H/8,W/8).
        """
        feat = self.features(x)
        feat = self.up1(feat)
        feat = self.up2(feat)
        return {
            "heatmap": self.heatmap_head(feat),
            "size": self.size_head(feat),
            "corners": self.corners_head(feat),
        }
