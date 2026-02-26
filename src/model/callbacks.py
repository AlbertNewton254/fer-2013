"""
src/model/callbacks.py

Callbacks for the training loop.
"""

import torch
from pathlib import Path


class Callback:
    """Base class. Override the methods you need."""

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        pass


class EarlyStopping(Callback):
    """
    Stops training when a monitored metric stops improving.

    Args:
        monitor (str): Metric key to watch in logs. Default: 'loss'.
        patience (int): Epochs to wait without improvement. Default: 5.
        min_delta (float): Minimum change to qualify as improvement. Default: 0.
        mode (str): 'min' for loss, 'max' for f1/accuracy. Default: 'min'.
    """

    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.stop = False

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        value = logs.get(self.monitor)
        if value is None:
            return

        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(
                    f"  EarlyStopping: no improvement in '{self.monitor}' "
                    f"for {self.patience} epochs. Stopping."
                )
                self.stop = True


class ModelCheckpoint(Callback):
    """
    Saves the model whenever a monitored metric improves.

    Args:
        filepath (str | Path): Where to save the checkpoint.
        monitor (str): Metric key to watch in logs. Default: 'loss'.
        mode (str): 'min' for loss, 'max' for f1/accuracy. Default: 'min'.
        verbose (bool): Print a message on each save. Default: True.
    """

    def __init__(
        self,
        filepath: str | Path = "best_model.pt",
        monitor: str = "loss",
        mode: str = "min",
        verbose: bool = True,
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

        self.best = float("inf") if mode == "min" else float("-inf")
        self._model = None

    def set_model(self, model: torch.nn.Module) -> None:
        self._model = model

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        value = logs.get(self.monitor)
        if value is None or self._model is None:
            return

        improved = (
            value < self.best if self.mode == "min" else value > self.best
        )

        if improved:
            self.best = value
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), self.filepath)
            if self.verbose:
                print(
                    f"  ModelCheckpoint: '{self.monitor}' improved to {value:.4f}. "
                    f"Saved → {self.filepath}"
                )


class LRSchedulerCallback(Callback):
    """
    Steps a PyTorch lr_scheduler at the end of each epoch.

    Args:
        scheduler: Any torch.optim.lr_scheduler instance.
        monitor (str): Required only for ReduceLROnPlateau. Default: 'loss'.
    """

    def __init__(self, scheduler, monitor: str = "loss"):
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        if isinstance(self.scheduler, ReduceLROnPlateau):
            value = logs.get(self.monitor)
            if value is not None:
                self.scheduler.step(value)
        else:
            self.scheduler.step()