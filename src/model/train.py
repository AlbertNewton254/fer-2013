"""
src/model/train.py

Training loop for FER2013CNN.
"""

from sklearn.metrics import f1_score
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.model.callbacks import Callback
from src.model.eval import evaluate
from src.config import get_config

EPOCHS = get_config()["training"]["epochs"]


def train_one_epoch(model, dataloader: DataLoader, criterion, optimizer, device, scaler: GradScaler):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, emotions in dataloader:
        images, emotions = images.to(device), emotions.to(device)
        optimizer.zero_grad()

        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, emotions)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(emotions.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
    return epoch_loss, epoch_f1


def train(
    model,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    criterion,
    optimizer,
    device,
    epochs: int = EPOCHS,
    callbacks: list[Callback] | None = None,
):
    callbacks = callbacks or []
    history = {
        "epoch": [],
        "loss": [],
        "f1": [],
        "val_loss": [],
        "val_f1": [],
    }

    # Give ModelCheckpoint a reference to the model
    for cb in callbacks:
        if hasattr(cb, "set_model"):
            cb.set_model(model)

    # GradScaler is a no-op on CPU, so this is safe regardless of device
    scaler = GradScaler(device=device.type)

    for epoch in range(epochs):
        loss, f1 = train_one_epoch(model, train_dataloader, criterion, optimizer, device, scaler)

        logs = {"loss": loss, "f1": f1}
        if val_dataloader is not None:
            val_loss, val_f1 = evaluate(model, val_dataloader, criterion, device)
            logs["val_loss"] = val_loss
            logs["val_f1"] = val_f1
            print(
                f"Epoch {epoch + 1}/{epochs} — "
                f"Loss: {loss:.4f}, F1: {f1:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}"
            )
        else:
            print(f"Epoch {epoch + 1}/{epochs} — Loss: {loss:.4f}, F1: {f1:.4f}")

        history["epoch"].append(epoch + 1)
        history["loss"].append(loss)
        history["f1"].append(f1)
        history["val_loss"].append(logs.get("val_loss"))
        history["val_f1"].append(logs.get("val_f1"))

        for cb in callbacks:
            cb.on_epoch_end(epoch, logs)

        # EarlyStopping requested a stop
        if any(getattr(cb, "stop", False) for cb in callbacks):
            break

    return history