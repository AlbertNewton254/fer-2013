"""
src/data/dataset.py

FER2013Dataset and utilities for data loading and preprocessing.
"""

from pathlib import Path
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import cv2
import torchvision.transforms as T
from src.config import get_config

CONFIG = get_config()

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / CONFIG["data"]["raw_data_dir"]

BATCH_SIZE = CONFIG["dataloader"]["batch_size"]
NUM_WORKERS = CONFIG["dataloader"]["num_workers"]

IMAGE_SIZE = CONFIG["data"]["image_size"]
NORM_MEAN = CONFIG["data"]["normalize"]["mean"]
NORM_STD = CONFIG["data"]["normalize"]["std"]
AUGMENTATION = CONFIG["data"]["augmentation"]
GABOR_CONFIG = CONFIG["data"].get("gabor", {})


def _build_gabor_kernels() -> list[np.ndarray]:
    if not GABOR_CONFIG.get("enabled", False):
        return []

    ksize = int(GABOR_CONFIG["kernel_size"])
    sigma = float(GABOR_CONFIG["sigma"])
    lambd = float(GABOR_CONFIG["lambda"])
    gamma = float(GABOR_CONFIG["gamma"])
    psi = float(GABOR_CONFIG["psi"])
    num_orientations = int(GABOR_CONFIG["num_orientations"])

    kernels: list[np.ndarray] = []
    for idx in range(num_orientations):
        theta = (math.pi * idx) / num_orientations
        kernel = cv2.getGaborKernel(
            ksize=(ksize, ksize),
            sigma=sigma,
            theta=theta,
            lambd=lambd,
            gamma=gamma,
            psi=psi,
            ktype=cv2.CV_32F,
        )
        kernels.append(kernel)
    return kernels


GABOR_KERNELS = _build_gabor_kernels()


train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(p=AUGMENTATION["horizontal_flip_p"]),
    T.RandomRotation(degrees=AUGMENTATION["rotation_degrees"]),
    T.RandomAffine(
        degrees=0,
        translate=tuple(AUGMENTATION["affine_translate"]),
        scale=tuple(AUGMENTATION["affine_scale"]),
    ),
    T.ColorJitter(
        brightness=AUGMENTATION["brightness"],
        contrast=AUGMENTATION["contrast"],
    ),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    T.RandomErasing(
        p=AUGMENTATION.get("random_erasing_p", 0.25),
        scale=tuple(AUGMENTATION.get("random_erasing_scale", [0.02, 0.2])),
        ratio=tuple(AUGMENTATION.get("random_erasing_ratio", [0.3, 3.3])),
        value=AUGMENTATION.get("random_erasing_value", "random"),
    ),
])

val_test_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


def get_weighted_sampler(dataframe: pd.DataFrame) -> WeightedRandomSampler:
    """Build a weighted sampler to compensate for class imbalance."""
    class_counts = dataframe["emotion"].value_counts()
    class_weights = 1.0 / class_counts
    sample_weights = torch.tensor(
        dataframe["emotion"].map(class_weights).values,
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


class FER2013Dataset(Dataset):
    """PyTorch Dataset for the FER2013 facial expression dataset."""

    def __init__(
        self,
        data_dir: Path = RAW_DATA_DIR,
        split: str = "train",
        transform=train_transform,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.data = self._load_data()
        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(sorted(self.data["emotion"].unique()))
        }
        self.gabor_enabled = bool(GABOR_CONFIG.get("enabled", False)) and bool(GABOR_KERNELS)
        self.include_original = bool(GABOR_CONFIG.get("include_original", True))
        gabor_channels = len(GABOR_KERNELS) if self.gabor_enabled else 0
        self.input_channels = gabor_channels + (1 if (self.include_original or not self.gabor_enabled) else 0)

    def _load_data(self) -> pd.DataFrame:
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        records = [
            {"path": img, "emotion": emotion_dir.name}
            for emotion_dir in split_dir.iterdir()
            if emotion_dir.is_dir()
            for img in emotion_dir.iterdir()
            if img.is_file()
        ]
        return pd.DataFrame(records)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = cv2.imread(str(row["path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {row['path']}")
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0

        if self.gabor_enabled:
            base_image = image[0]
            base_image = base_image * float(NORM_STD[0]) + float(NORM_MEAN[0])
            base_np = base_image.detach().cpu().numpy().astype(np.float32)

            response_tensors: list[torch.Tensor] = []
            for kernel in GABOR_KERNELS:
                response = cv2.filter2D(base_np, cv2.CV_32F, kernel)
                response = (response - response.mean()) / (response.std() + 1e-6)
                response_tensors.append(torch.from_numpy(response))

            channels: list[torch.Tensor] = []
            if self.include_original:
                channels.append(image[0])
            channels.extend(response_tensors)
            image = torch.stack(channels, dim=0)

        label = torch.tensor(self.class_to_idx[row["emotion"]], dtype=torch.long)
        return image, label


def get_dataloader(
    dataset: FER2013Dataset,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    sampler=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=(sampler is None),
    )


if __name__ == "__main__":
    train_dataset = FER2013Dataset(transform=train_transform)
    print(f"Train size: {len(train_dataset)}")
    image, label = train_dataset[0]
    print(f"Image shape: {image.shape}, label: {label}")

    test_dataset = FER2013Dataset(split="test", transform=val_test_transform)
    print(f"Test size: {len(test_dataset)}")
    image, label = test_dataset[0]
    print(f"Image shape: {image.shape}, label: {label}")