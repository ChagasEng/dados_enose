"""Pré-processamento das séries dos sensores MQ."""

from __future__ import annotations

import numpy as np


def relative_conductance(
    conductance: np.ndarray,
    baseline_samples: int = 20,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula S = G/G0 para cada sensor.

    Aceita ``(..., sensores, tempo)``. G0 é a mediana das primeiras amostras,
    uma escolha robusta a impulsos espúrios. Retorna a resposta relativa e G0.
    """
    values = np.asarray(conductance, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError("A entrada deve ter, no minimo, dimensoes (sensor, tempo).")
    if values.shape[-1] < 2:
        raise ValueError("A serie temporal deve conter pelo menos duas amostras.")
    if not np.all(np.isfinite(values)):
        raise ValueError("A condutancia contem NaN ou infinito.")
    n_base = min(max(1, baseline_samples), values.shape[-1])
    g0 = np.median(values[..., :n_base], axis=-1, keepdims=True)
    if np.any(np.abs(g0) <= eps):
        raise ValueError("G0 nulo ou demasiado pequeno; verifique o baseline.")
    return (values / g0).astype(np.float32), g0.squeeze(-1).astype(np.float32)

