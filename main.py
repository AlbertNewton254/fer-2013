"""
main.py

Project entry point. Runs data fetching, training, and evaluation.
Run from the project root: python main.py
"""

import torch
import torch.optim as optim
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
from src.model.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    PeriodicCheckpoint,
    LRSchedulerCallback,
)
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


def make_run_dir() -> Path:
    checkpoint_config = CONFIG["callbacks"]["checkpoint"]
    root_dir = Path(checkpoint_config["root_dir"])
    run_prefix = checkpoint_config["run_prefix"]
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = root_dir / f"{run_prefix}{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_training_plots(history: dict[str, list], run_dir: Path) -> None:
    epochs = history["epoch"]

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["loss"], label="train_loss")
    val_loss = [v for v in history["val_loss"] if v is not None]
    if val_loss:
        plt.plot(epochs[:len(val_loss)], val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["f1"], label="train_f1")
    val_f1 = [v for v in history["val_f1"] if v is not None]
    if val_f1:
        plt.plot(epochs[:len(val_f1)], val_f1, label="val_f1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Training and Validation F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "f1_curve.png")
    plt.close()


def main():
    run_dir = make_run_dir()

    # 1. Data
    fetch_data()

    full_train_dataset = FER2013Dataset(
        split=CONFIG["data"]["train_split"],
        transform=train_transform,
    )

    train_df, val_df = train_test_split(
        full_train_dataset.data,
        test_size=CONFIG["training"]["val_ratio"],
        random_state=CONFIG["training"]["random_state"],
        stratify=full_train_dataset.data["emotion"],
    )

    train_dataset = FER2013Dataset(
        split=CONFIG["data"]["train_split"],
        transform=train_transform,
    )
    val_dataset = FER2013Dataset(
        split=CONFIG["data"]["train_split"],
        transform=val_test_transform,
    )
    train_dataset.data = train_df.reset_index(drop=True)
    val_dataset.data = val_df.reset_index(drop=True)
    train_dataset.class_to_idx = full_train_dataset.class_to_idx
    val_dataset.class_to_idx = full_train_dataset.class_to_idx

    test_dataset = FER2013Dataset(
        split=CONFIG["data"]["test_split"],
        transform=val_test_transform,
    )

    sampler = get_weighted_sampler(train_dataset.data)
    train_loader = get_dataloader(train_dataset, sampler=sampler)
    val_loader = get_dataloader(val_dataset)
    test_loader = get_dataloader(test_dataset)

    # 2. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FER2013CNN(input_channels=train_dataset.input_channels).to(device)
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
            filepath=run_dir / CONFIG["callbacks"]["checkpoint"]["best_filename"],
            monitor=CONFIG["callbacks"]["monitor"],
            mode=CONFIG["callbacks"]["checkpoint"]["mode"],
            verbose=CONFIG["callbacks"]["checkpoint"]["verbose"],
        ),
        PeriodicCheckpoint(
            dirpath=run_dir,
            every_n_epochs=CONFIG["callbacks"]["checkpoint"]["periodic_every_epochs"],
            filename_pattern=CONFIG["callbacks"]["checkpoint"]["periodic_filename"],
            verbose=CONFIG["callbacks"]["checkpoint"]["verbose"],
        ),
        LRSchedulerCallback(scheduler, monitor=CONFIG["callbacks"]["lr_scheduler_monitor"]),
    ]

    # 4. Training
    history = train(model, train_loader, val_loader, criterion, optimizer, device, callbacks=callbacks)
    save_training_plots(history, run_dir)

    # 5. Load the best checkpoint selected by validation F1
    checkpoint_path = run_dir / CONFIG["callbacks"]["checkpoint"]["best_filename"]
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 6. Evaluation on test split
    test_loss, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")
    print(f"Run artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()