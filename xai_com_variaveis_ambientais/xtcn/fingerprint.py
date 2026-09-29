"""Padronização e criação do Fingerprint Map discreto."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureStandardizer:
    """Padroniza cada descritor usando apenas estatísticas do treino."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "FeatureStandardizer":
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (6, 8):
            raise ValueError("features deve ter dimensao (amostras, 6, 8).")
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < 1e-7] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Chame fit antes de transform.")
        return ((np.asarray(features, dtype=np.float32) - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def state_dict(self) -> dict[str, np.ndarray]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("O normalizador ainda nao foi ajustado.")
        return {"mean": self.mean_, "scale": self.scale_}


def _scalar_to_rgb(unit_values: np.ndarray) -> np.ndarray:
    """Mapa de cores azul-ciano-amarelo contínuo aplicado só às 48 células."""
    x = np.clip(unit_values, 0.0, 1.0)
    red = np.clip(1.8 * x - 0.45, 0.0, 1.0)
    green = np.clip(1.8 - 3.6 * np.abs(x - 0.5), 0.0, 1.0)
    blue = np.clip(1.35 - 1.8 * x, 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1).astype(np.float32)


def build_fingerprint_map(
    standardized_features: np.ndarray,
    size: int = 224,
    clip_z: float = 3.0,
) -> np.ndarray:
    """Converte ``(..., 6, 8)`` em ``(..., 3, size, size)``.

    Cada uma das 48 células recebe uma cor única, copiada para um bloco inteiro.
    Os limites dos blocos usam partições inteiras; não há interpolação espacial.
    """
    matrix = np.asarray(standardized_features, dtype=np.float32)
    if matrix.shape[-2:] != (6, 8):
        raise ValueError("A matriz de caracteristicas deve terminar em (6, 8).")
    if size < 8:
        raise ValueError("size deve ser pelo menos 8.")
    unit = (np.clip(matrix, -clip_z, clip_z) + clip_z) / (2.0 * clip_z)
    cell_rgb = _scalar_to_rgb(unit)
    output = np.empty((*matrix.shape[:-2], size, size, 3), dtype=np.float32)
    row_edges = np.linspace(0, size, 7, dtype=int)
    col_edges = np.linspace(0, size, 9, dtype=int)
    for row in range(6):
        for col in range(8):
            output[..., row_edges[row] : row_edges[row + 1], col_edges[col] : col_edges[col + 1], :] = cell_rgb[..., row, col, None, None, :]
    return np.moveaxis(output, -1, -3)


def aggregate_heatmap_by_cell(heatmap: np.ndarray) -> np.ndarray:
    """Resume um Grad-CAM espacial nas 48 células sensor × descritor."""
    values = np.asarray(heatmap, dtype=np.float32).squeeze()
    if values.ndim != 2:
        raise ValueError("heatmap deve ser bidimensional.")
    height, width = values.shape
    row_edges = np.linspace(0, height, 7, dtype=int)
    col_edges = np.linspace(0, width, 9, dtype=int)
    cells = np.empty((6, 8), dtype=np.float32)
    for row in range(6):
        for col in range(8):
            block = values[row_edges[row] : row_edges[row + 1], col_edges[col] : col_edges[col + 1]]
            cells[row, col] = float(block.mean())
    return cells

