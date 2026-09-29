"""Gerador reprodutível de séries sintéticas para validar o pipeline."""

from __future__ import annotations

import numpy as np


def generate_synthetic_enose(
    samples_per_class: int = 24,
    time_steps: int = 256,
    sampling_rate: float = 10.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera condutâncias ``(N, 6, T)`` balanceadas para seis tratamentos.

    Os padrões simulam efeitos parcialmente sobrepostos de nemátodes, humidade
    e pressão, sem pretender substituir dados físicos reais.
    """
    if samples_per_class < 2 or time_steps < 32:
        raise ValueError("Use samples_per_class >= 2 e time_steps >= 32.")
    rng = np.random.default_rng(seed)
    t = np.arange(time_steps, dtype=np.float32) / sampling_rate
    signals: list[np.ndarray] = []
    labels: list[int] = []

    # Assinaturas relativas: sensores MQ respondem de forma distinta aos fatores.
    nematode_signature = np.array([0.20, 0.34, 0.27, 0.10, 0.13, 0.42])
    moisture_signature = np.array([0.15, 0.08, 0.12, 0.20, 0.07, 0.30])
    pressure_signature = np.array([0.13, 0.16, 0.14, 0.09, 0.11, 0.18])
    control_signature = np.array([0.08, 0.06, 0.07, 0.04, 0.05, 0.09])

    for class_idx in range(6):
        for _ in range(samples_per_class):
            baseline = rng.uniform(0.75, 1.35, size=(6, 1))
            baseline_drift = rng.normal(0.0, 0.025, size=(6, 1)) * (t / t[-1])

            has_nematodes = class_idx in (1, 3)
            is_moist = class_idx in (3, 4)
            forced_pressure = class_idx in (1, 2, 3, 4)
            amplitude = control_signature.copy()
            amplitude += nematode_signature * has_nematodes
            amplitude += moisture_signature * is_moist
            amplitude += pressure_signature * forced_pressure
            if class_idx == 5:  # protocolo de referência: dinâmica mais lenta
                amplitude += 0.55 * nematode_signature + 0.35 * moisture_signature

            amplitude *= rng.normal(1.0, 0.07, size=6)
            onset = rng.uniform(1.8, 2.8)
            rise_tau = 1.0 + 0.35 * is_moist + 0.20 * (class_idx == 5)
            decay_tau = 8.0 + 2.5 * is_moist + 2.0 * (class_idx == 5)
            shifted = np.maximum(t - onset, 0.0)
            pulse = (1.0 - np.exp(-shifted / rise_tau)) * np.exp(-shifted / decay_tau)

            # Oscilações de baixa amplitude dão conteúdo espectral específico.
            class_frequency = 0.12 + 0.035 * class_idx
            phase = rng.uniform(0, 2 * np.pi, size=(6, 1))
            oscillation = 0.018 * np.sin(2 * np.pi * class_frequency * t + phase)
            relative = 1.0 + amplitude[:, None] * pulse + oscillation
            relative += baseline_drift + rng.normal(0.0, 0.008, size=(6, time_steps))
            conductance = baseline * relative
            signals.append(conductance.astype(np.float32))
            labels.append(class_idx)

    order = rng.permutation(len(labels))
    return np.stack(signals)[order], np.asarray(labels, dtype=np.int64)[order]


def stratified_split(
    labels: np.ndarray,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cria índices estratificados de treino, validação e teste."""
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("As fracoes devem deixar uma particao de teste positiva.")
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    parts = [[], [], []]
    for class_idx in np.unique(labels):
        indices = np.flatnonzero(labels == class_idx)
        if len(indices) < 3:
            raise ValueError("Cada classe precisa de pelo menos 3 amostras para treino/validacao/teste.")
        rng.shuffle(indices)
        n_train = max(1, int(len(indices) * train_fraction))
        n_val = max(1, int(len(indices) * val_fraction))
        if n_train + n_val >= len(indices):
            n_train = len(indices) - 2
            n_val = 1
        parts[0].extend(indices[:n_train])
        parts[1].extend(indices[n_train : n_train + n_val])
        parts[2].extend(indices[n_train + n_val :])
    return tuple(rng.permutation(np.asarray(part, dtype=np.int64)) for part in parts)  # type: ignore[return-value]

