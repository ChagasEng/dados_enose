"""Dataset PyTorch para Fingerprint Maps."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FingerprintDataset(Dataset):
    def __init__(self, fingerprints: np.ndarray, labels: np.ndarray) -> None:
        x = np.asarray(fingerprints, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("fingerprints deve ter dimensao (N, 3, H, W).")
        if len(x) != len(y):
            raise ValueError("Numero de mapas e rotulos incompativel.")
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

