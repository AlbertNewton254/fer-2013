"""
src/model/eval.py

Evaluation loop for FER2013CNN.
"""

from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader


def evaluate(model, dataloader: DataLoader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, emotions in dataloader:
            images, emotions = images.to(device), emotions.to(device)
            outputs = model(images)
            loss = criterion(outputs, emotions)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(emotions.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
    return epoch_loss, epoch_f1