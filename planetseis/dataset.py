"""Torch dataset over cached windows, with on-the-fly augmentation (FR-5)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import normalize_window


class WindowDataset(Dataset):
    """Windows stored as arrays: X [N, n_samples] raw (unnormalized), y [N], offset [N]."""

    def __init__(self, X, y, offset, augment=False, seed=0):
        self.X, self.y, self.offset = X, y.astype(np.float32), offset.astype(np.float32)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def _augment(self, w: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.5:
            w = -w                                     # polarity flip
        w = w * self.rng.uniform(0.5, 2.0)             # amplitude scale
        if self.rng.random() < 0.5:                    # noise injection
            snr_scale = self.rng.uniform(0.05, 0.4)
            w = w + self.rng.normal(0, snr_scale * (w.std() + 1e-9), size=w.shape)
        return w

    def __getitem__(self, i):
        w = self.X[i].astype(np.float32)
        if self.augment:
            w = self._augment(w)
        w = normalize_window(w)
        return (
            torch.from_numpy(w).unsqueeze(0),
            torch.tensor(self.y[i]),
            torch.tensor(self.offset[i]),
        )
