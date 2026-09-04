"""Supervised few-shot train / infer — no Qt."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from tensorspec.core.ml.models import SupervisedCNN

ProgressFn = Optional[Callable[[int, str], None]]


def _emit(on_progress: ProgressFn, value: int, message: str) -> None:
    if on_progress is not None:
        on_progress(value, message)


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def run_sup_train(
    X: np.ndarray,
    Y: np.ndarray,
    num_classes: int,
    on_progress: ProgressFn = None,
):
    """Train SupervisedCNN; return model on CPU with data_min/data_max."""
    device = _device()
    model = SupervisedCNN(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    X_ten = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    X_ten = F.interpolate(X_ten, size=(128, 128), mode="bilinear", align_corners=False)

    orig_min, orig_max = X_ten.min(), X_ten.max()
    X_ten = (X_ten - orig_min) / (orig_max - orig_min + 1e-8)
    Y_ten = torch.tensor(Y, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(X_ten, Y_ten), batch_size=min(16, len(X)), shuffle=True
    )

    epochs = 60
    model.train()
    for ep in range(epochs):
        for b_x, b_y in loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = model(b_x)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
        if ep % 5 == 0:
            _emit(on_progress, int((ep / epochs) * 100), f"Training Few-Shot... Epoch {ep}")

    model.eval()
    model.data_min = orig_min.item()
    model.data_max = orig_max.item()
    _emit(on_progress, 100, "Training Done")
    return model.cpu()


def run_sup_test(
    model,
    val_array: np.ndarray,
    on_progress: ProgressFn = None,
) -> np.ndarray:
    """Run inference; return (nY, nX, n_classes) probability map."""
    device = _device()
    model.to(device)
    model.eval()

    dim_E, dim_A, nY, nX = val_array.shape
    flat_val = val_array.transpose(2, 3, 0, 1).reshape(nY * nX, 1, dim_E, dim_A)

    t_orig = torch.tensor(flat_val, dtype=torch.float32)
    t_orig = F.interpolate(t_orig, size=(128, 128), mode="bilinear", align_corners=False)
    t_orig = (t_orig - model.data_min) / (model.data_max - model.data_min + 1e-8)

    loader = DataLoader(TensorDataset(t_orig), batch_size=256, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for i, (b_x,) in enumerate(loader):
            b_x = b_x.to(device)
            logits = model(b_x)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            if i % 10 == 0:
                _emit(
                    on_progress,
                    int((i / len(loader)) * 100),
                    "Running Inference on Full Map...",
                )

    prob_array = np.concatenate(all_probs, axis=0)
    prob_map = prob_array.reshape((nY, nX, -1))

    model.cpu()
    _emit(on_progress, 100, "Done!")
    return prob_map
