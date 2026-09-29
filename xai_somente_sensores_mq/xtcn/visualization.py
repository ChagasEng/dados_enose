"""Visualizações do treino, matriz de confusão e explicações."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import CLASS_NAMES, FEATURE_NAMES, SENSOR_NAMES
from .fingerprint import aggregate_heatmap_by_cell
from .training import History


def save_training_history(history: History, path: Path) -> None:
    epochs = np.arange(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history.train_loss, label="Treino")
    axes[0].plot(epochs, history.val_loss, label="Validação")
    axes[0].set(xlabel="Época", ylabel="Loss", title="Cross-entropy")
    axes[1].plot(epochs, history.train_accuracy, label="Treino")
    axes[1].plot(epochs, history.val_accuracy, label="Validação")
    axes[1].set(xlabel="Época", ylabel="Acurácia", title="Acurácia")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion_matrix(confusion: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(confusion, cmap="Blues")
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            axis.text(col, row, str(confusion[row, col]), ha="center", va="center")
    axis.set_xticks(range(6), labels=range(1, 7))
    axis.set_yticks(range(6), labels=range(1, 7))
    axis.set(xlabel="Classe predita", ylabel="Classe real", title="Matriz de confusão")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_gradcam_explanation(
    fingerprint_chw: np.ndarray,
    heatmap: np.ndarray,
    true_class: int,
    predicted_class: int,
    path: Path,
) -> np.ndarray:
    """Salva fingerprint, sobreposição e relevância nas 48 células."""
    fingerprint = np.moveaxis(fingerprint_chw, 0, -1)
    heatmap = np.asarray(heatmap).squeeze()
    cell_importance = aggregate_heatmap_by_cell(heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(fingerprint)
    axes[0].set_title("Fingerprint 6 × 8")
    axes[1].imshow(fingerprint)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.48, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM sobreposto")
    image = axes[2].imshow(cell_importance, cmap="magma", vmin=0, vmax=max(1e-8, float(cell_importance.max())))
    axes[2].set_xticks(range(8), FEATURE_NAMES, rotation=45, ha="right")
    axes[2].set_yticks(range(6), SENSOR_NAMES)
    axes[2].set_title("Importância sensor × descritor")
    fig.colorbar(image, ax=axes[2], fraction=0.046)
    for axis in axes[:2]:
        axis.set_xticks(np.linspace(14, 210, 8), FEATURE_NAMES, rotation=45, ha="right")
        axis.set_yticks(np.linspace(18, 205, 6), SENSOR_NAMES)
    fig.suptitle(f"Real: {CLASS_NAMES[true_class]}\nPredita: {CLASS_NAMES[predicted_class]}")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return cell_importance

