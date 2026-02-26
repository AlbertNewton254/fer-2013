"""
src/model/model.py

CNN architecture for facial expression recognition (FER2013).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 7


class FER2013CNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 512 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


if __name__ == "__main__":
    model = FER2013CNN()
    print(model)
    dummy = torch.randn(1, 1, 48, 48)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    assert out.shape == (1, NUM_CLASSES)