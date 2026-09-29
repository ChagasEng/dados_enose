"""Extração dos oito descritores temporais/espectrais por sensor."""

from __future__ import annotations

import numpy as np


def _half_height_peak_width(signal: np.ndarray, peak_index: int) -> float:
    """Largura normalizada do pico na metade da sua amplitude."""
    amplitude = float(signal[peak_index])
    if amplitude <= 0.0:
        return 0.0
    half = amplitude / 2.0
    left = peak_index
    while left > 0 and signal[left] >= half:
        left -= 1
    right = peak_index
    while right < signal.size - 1 and signal[right] >= half:
        right += 1
    return float(right - left) / max(1, signal.size - 1)


def extract_features(relative_response: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Extrai uma matriz ``(..., 6, 8)`` de características.

    As características são calculadas sobre ``abs(G/G0 - 1)``:
    AUC, energia espectral, pico de PSD, largura do pico, variância,
    posição normalizada do pico, frequência dominante normalizada por Nyquist
    e entropia espectral normalizada (FTF).
    """
    response = np.asarray(relative_response, dtype=np.float32)
    if response.ndim < 2:
        raise ValueError("Esperado (..., sensores, tempo).")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate deve ser positivo.")

    original_shape = response.shape[:-2]
    n_sensors, n_time = response.shape[-2:]
    flat = response.reshape(-1, n_sensors, n_time)
    result = np.empty((flat.shape[0], n_sensors, 8), dtype=np.float32)
    dt = 1.0 / sampling_rate

    for sample_idx, sample in enumerate(flat):
        for sensor_idx, sensor_signal in enumerate(sample):
            dynamic = np.abs(sensor_signal.astype(np.float64) - 1.0)
            centered = dynamic - dynamic.mean()
            spectrum = np.fft.rfft(centered)
            power = np.abs(spectrum) ** 2
            freqs = np.fft.rfftfreq(n_time, d=dt)

            # O termo DC é removido para os descritores de frequência.
            non_dc_power = power.copy()
            if non_dc_power.size:
                non_dc_power[0] = 0.0
            power_sum = float(non_dc_power.sum())
            psd = non_dc_power / (sampling_rate * n_time)
            peak_idx = int(np.argmax(dynamic))
            dominant_idx = int(np.argmax(non_dc_power)) if power_sum > 0 else 0
            probabilities = non_dc_power / (power_sum + 1e-12)
            nonzero = probabilities[probabilities > 0]
            entropy_denominator = np.log(max(2, non_dc_power.size - 1))
            spectral_entropy = float(-(nonzero * np.log(nonzero)).sum() / entropy_denominator)
            nyquist = sampling_rate / 2.0

            result[sample_idx, sensor_idx] = (
                np.trapezoid(dynamic, dx=dt),
                power_sum / max(1, n_time),
                float(psd.max(initial=0.0)),
                _half_height_peak_width(dynamic, peak_idx),
                float(np.var(dynamic)),
                peak_idx / max(1, n_time - 1),
                float(freqs[dominant_idx] / nyquist) if nyquist > 0 else 0.0,
                spectral_entropy,
            )

    return result.reshape(*original_shape, n_sensors, 8)

