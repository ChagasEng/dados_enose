"""Renderiza as matrizes no formato visual do exemplo enviado pelo usuario."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gerar_matrizes import MODELS, OUTPUT, historical_matrices, loso_matrix


LABELS = ["WV", "NO"]  # Display order: class 0; class 1.


def draw(ax: plt.Axes, matrix: np.ndarray, title: str) -> None:
    values = matrix / matrix.sum(axis=1, keepdims=True) * 100
    ax.imshow(values, cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
    ax.set_title(title, fontsize=14, pad=5, weight="normal")
    ax.set_xticks((0, 1), LABELS, fontsize=11)
    ax.set_yticks((0, 1), LABELS, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12, labelpad=2)
    ax.set_ylabel("Actual", fontsize=12, labelpad=2)
    ax.tick_params(axis="both", length=3, width=0.8)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in (0, 1):
        for column in (0, 1):
            value = values[row, column]
            ax.text(column, row, f"{value:.1f}", ha="center", va="center",
                    fontsize=12, color="white" if value >= 70 else "#303030")


def sheet(path: Path, panels: list[tuple[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.6), facecolor="white")
    for ax, panel in zip(axes.flat, panels):
        draw(ax, panel[1], panel[0])
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.09, top=0.95,
                        wspace=0.42, hspace=0.38)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    main_names = ["Extra Trees", "Random Forest", "MLP (ANN)",
                  "k-NN", "SVM", "LDA"]
    matrix_by_scenario = {
        scenario: {model[0]: loso_matrix(scenario, model) for model in MODELS
                   if model[0] != "ANN + k-NN + SVM + LDA"}
        for scenario in ("mq", "mq_ambiente")
    }
    for scenario, name in (("mq", "five_mq"), ("mq_ambiente", "five_mq_environment")):
        sheet(OUTPUT / f"example_style_{name}.png", [
            (model, matrix_by_scenario[scenario][model]) for model in main_names
        ])
        sheet(OUTPUT / f"example_style_{name}_tcn.png", [
            ("TCN", matrix_by_scenario[scenario]["TCN"])
        ])
    historical = historical_matrices()
    sheet(OUTPUT / "example_style_historical.png", historical)
    print("Saved example-style images")


if __name__ == "__main__":
    main()
