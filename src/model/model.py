"""
src/model/model.py

Improved CNN architecture for facial expression recognition (FER2013).
"""

import torch
import torch.nn as nn
from src.config import get_config

NUM_CLASSES = get_config()["model"]["num_classes"]


class ConvBlock(nn.Module):
    """
    Conv -> BN -> ReLU
    Conv -> BN -> ReLU
    MaxPool
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FER2013CNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, input_channels: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(input_channels, 64),   # 48 -> 24
            ConvBlock(64, 128),              # 24 -> 12
            ConvBlock(128, 256),             # 12 -> 6
            ConvBlock(256, 512),             # 6 -> 3
        )

        # Remove dependência fixa de 3x3
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = FER2013CNN()
    print(model)
    dummy = torch.randn(1, 1, 48, 48)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    assert out.shape == (1, NUM_CLASSES)