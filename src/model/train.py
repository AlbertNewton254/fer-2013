"""
src/model/train.py

Training loop for FER2013CNN.
"""

from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader

EPOCHS = 10


def train_one_epoch(model, dataloader: DataLoader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, emotions in dataloader:
        images, emotions = images.to(device), emotions.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, emotions)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(emotions.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
    return epoch_loss, epoch_f1


def train(model, train_dataloader: DataLoader, criterion, optimizer, device, epochs: int = EPOCHS):
    for epoch in range(epochs):
        loss, f1 = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{epochs} — Loss: {loss:.4f}, F1: {f1:.4f}")