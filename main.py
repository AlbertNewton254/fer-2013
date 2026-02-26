"""
main.py

Project entry point. Runs data fetching, training, and evaluation.
Run from the project root: python main.py
"""

import torch
import torch.optim as optim
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data import (
    FER2013Dataset,
    get_dataloader,
    get_weighted_sampler,
    train_transform,
    val_test_transform,
    fetch_data,
)
from src.focal_loss import FocalLoss
from src.model import FER2013CNN, train, evaluate
from src.model.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback
from src.config import get_config


CONFIG = get_config()


def get_effective_num_alpha(dataframe: pd.DataFrame, class_to_idx: dict[str, int], beta: float) -> torch.Tensor:
    class_counts = dataframe["emotion"].value_counts().to_dict()
    num_classes = len(class_to_idx)
    alpha = torch.ones(num_classes, dtype=torch.float32)

    for class_name, class_index in class_to_idx.items():
        n_i = float(class_counts.get(class_name, 0))
        if n_i <= 0:
            alpha[class_index] = 1.0
            continue
        effective_num = 1.0 - (beta ** n_i)
        alpha[class_index] = (1.0 - beta) / max(effective_num, 1e-12)

    alpha = alpha / alpha.sum() * num_classes
    return alpha


def main():
    # 1. Data
    fetch_data()

    train_dataset = FER2013Dataset(
        split=CONFIG["data"]["train_split"],
        transform=train_transform,
    )
    val_dataset = FER2013Dataset(
        split=CONFIG["data"]["val_split"],
        transform=val_test_transform,
    )

    sampler = get_weighted_sampler(train_dataset.data)
    train_loader = get_dataloader(train_dataset, sampler=sampler)
    val_loader = get_dataloader(val_dataset)

    # 2. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FER2013CNN().to(device)
    beta = CONFIG["loss"]["focal"]["class_balance_beta"]
    alpha = get_effective_num_alpha(train_dataset.data, train_dataset.class_to_idx, beta=beta)
    criterion = FocalLoss(alpha=alpha)
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["training"]["optimizer"]["lr"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=CONFIG["training"]["scheduler"]["mode"],
        patience=CONFIG["training"]["scheduler"]["patience"],
        factor=CONFIG["training"]["scheduler"]["factor"],
    )

    # 3. Callbacks
    callbacks = [
        EarlyStopping(
            monitor=CONFIG["callbacks"]["monitor"],
            patience=CONFIG["callbacks"]["early_stopping"]["patience"],
            min_delta=CONFIG["callbacks"]["early_stopping"]["min_delta"],
            mode=CONFIG["callbacks"]["early_stopping"]["mode"],
        ),
        ModelCheckpoint(
            filepath=CONFIG["callbacks"]["checkpoint"]["filepath"],
            monitor=CONFIG["callbacks"]["monitor"],
            mode=CONFIG["callbacks"]["checkpoint"]["mode"],
            verbose=CONFIG["callbacks"]["checkpoint"]["verbose"],
        ),
        LRSchedulerCallback(scheduler, monitor=CONFIG["callbacks"]["monitor"]),
    ]

    # 4. Training
    train(model, train_loader, criterion, optimizer, device, callbacks=callbacks)

    # 5. Evaluation
    val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
    print(f"\nVal Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")


if __name__ == "__main__":
    main()