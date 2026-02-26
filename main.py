"""
main.py

Project entry point. Runs data fetching, training, and evaluation.
Run from the project root: python main.py
"""

import torch
import torch.optim as optim
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
    criterion = FocalLoss()
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