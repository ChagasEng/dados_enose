"""Descritores e fingerprint conforme o suplemento do artigo X-TCN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import seaborn as sns


def _peak_width_seconds(signal: np.ndarray, peak: int, dt: float) -> float:
    amplitude = float(signal[peak])
    if amplitude <= 0:
        return 0.0
    half = amplitude / 2.0
    left = peak
    while left > 0 and signal[left] >= half:
        left -= 1
    right = peak
    while right < len(signal) - 1 and signal[right] >= half:
        right += 1
    return float(right - left) * dt


def extract_article_features(
    relative_signals: np.ndarray,
    sampling_rate: float = 4.0,
    dominant_count: int = 3,
    fourier_bins: int = 10,
) -> np.ndarray:
    """Retorna ``(N, sensores, 8)`` seguindo a Tabela S2.

    ES, PSD, DFC e FTF são vetores no suplemento, mas o fingerprint reserva uma
    célula para cada descritor. Como o código público não contém essa redução,
    usamos respectivamente energia total, pico de PSD, centro ponderado dos M
    componentes dominantes e magnitude média dos K primeiros bins não-DC.
    """
    values = np.asarray(relative_signals, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("Esperado (amostras, sensores, tempo).")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate deve ser positiva.")
    dt = 1.0 / sampling_rate
    output = np.empty((values.shape[0], values.shape[1], 8), dtype=np.float32)
    for sample_index, sample in enumerate(values):
        for sensor_index, response in enumerate(sample.astype(np.float64)):
            spectrum = np.fft.rfft(response)
            magnitude = np.abs(spectrum)
            power = magnitude**2
            frequencies = np.fft.rfftfreq(len(response), d=dt)
            non_dc_power = power[1:]
            non_dc_magnitude = magnitude[1:]
            non_dc_frequencies = frequencies[1:]
            peak = int(np.argmax(response))
            count = min(dominant_count, len(non_dc_power))
            if count:
                top = np.argpartition(non_dc_power, -count)[-count:]
                weights = non_dc_power[top]
                dominant = float(
                    np.average(non_dc_frequencies[top], weights=weights)
                    if weights.sum() > 0
                    else 0.0
                )
            else:
                dominant = 0.0
            k = min(fourier_bins, len(non_dc_magnitude))
            output[sample_index, sensor_index] = (
                float(np.sum(response) * dt),
                float(np.sum(power)),
                float(np.max(power / len(response))),
                _peak_width_seconds(response, peak, dt),
                float(np.var(response)),
                float(peak),
                dominant,
                float(np.mean(non_dc_magnitude[:k])) if k else 0.0,
            )
    return output


@dataclass
class ArticleMinMaxScaler:
    minimum_: np.ndarray | None = None
    span_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "ArticleMinMaxScaler":
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (6, 8):
            raise ValueError("Esperado (amostras, 6, 8).")
        self.minimum_ = values.min(axis=0)
        maximum = values.max(axis=0)
        self.span_ = maximum - self.minimum_
        self.span_[self.span_ < 1e-8] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.minimum_ is None or self.span_ is None:
            raise RuntimeError("O Min-Max ainda nao foi ajustado.")
        values = (np.asarray(features, dtype=np.float32) - self.minimum_) / self.span_
        return np.clip(values, 0.0, 1.0).astype(np.float32)


def build_article_fingerprints(features: np.ndarray, size: int = 224) -> np.ndarray:
    """Cria RGB discreto ``(N,3,224,224)`` com descritores nas linhas."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (6, 8):
        raise ValueError("Esperado (amostras, 6, 8).")
    # Artigo: eixo X=sensores e eixo Y=descritores, portanto 8 x 6.
    matrices = np.transpose(values, (0, 2, 1))
    cmap = sns.color_palette("crest", as_cmap=True)
    cell_rgb = cmap(np.clip(matrices, 0.0, 1.0))[..., :3].astype(np.float32)
    output = np.empty((len(values), size, size, 3), dtype=np.float32)
    row_edges = np.linspace(0, size, 9, dtype=int)
    column_edges = np.linspace(0, size, 7, dtype=int)
    for row in range(8):
        for column in range(6):
            output[
                :,
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
                :,
            ] = cell_rgb[:, row, column, None, None, :]
    return np.moveaxis(output, -1, 1)


def heatmap_to_cells(heatmap: np.ndarray) -> np.ndarray:
    """Agrega um mapa 224×224 em ``(8 descritores, 6 sensores)``."""
    values = np.asarray(heatmap, dtype=np.float32).squeeze()
    row_edges = np.linspace(0, values.shape[0], 9, dtype=int)
    column_edges = np.linspace(0, values.shape[1], 7, dtype=int)
    cells = np.empty((8, 6), dtype=np.float32)
    for row in range(8):
        for column in range(6):
            cells[row, column] = values[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ].mean()
    return cells

