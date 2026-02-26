"""
src/data/dataset.py

FER2013Dataset and utilities for data loading and preprocessing.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import cv2
import torchvision.transforms as T

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

BATCH_SIZE = 32
NUM_WORKERS = 4


train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((48, 48)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

val_test_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((48, 48)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
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