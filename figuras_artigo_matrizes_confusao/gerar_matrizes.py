"""Gera as matrizes de confusao por leitura usadas nas Tabelas 2 e 3 do artigo."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
REF = ROOT / "06_07_melhor_modelo" / "analise_seco_molhado"
XTCN = (
    ROOT
    / "08_10_2026_xtcn_solo_medio"
    / "resultados_tabela22_xtcn_mesmas_features"
    / "predicoes_por_leitura.csv"
)
LABELS = ["Com nematoide\n(classe 0)", "Sem nematoide\n(classe 1)"]


def read_reference(condition: str) -> np.ndarray:
    name = "seco" if condition == "seco" else "molhado"
    frame = pd.read_csv(REF / f"matriz_confusao_{name}.csv", index_col=0)
    return frame.to_numpy(dtype=int)


def read_xtcn(condition: str, predictions: pd.DataFrame) -> np.ndarray:
    subset = predictions.loc[predictions["faixa"] == condition]
    table = pd.crosstab(subset["Classe"], subset["classe_prevista_xtcn"])
    return table.reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy(dtype=int)


def draw_matrix(ax, values: np.ndarray, title: str):
    percentages = values / values.sum(axis=1, keepdims=True) * 100
    image = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks([0, 1], LABELS, fontsize=10)
    ax.set_yticks([0, 1], LABELS, fontsize=10)
    ax.set_xlabel("Classe prevista", fontsize=11, labelpad=9)
    ax.set_ylabel("Classe real", fontsize=11, labelpad=9)
    ax.tick_params(length=0)
    for row in range(2):
        for column in range(2):
            pct = percentages[row, column]
            ax.text(
                column,
                row,
                f"{values[row, column]:,}".replace(",", ".") + f"\n({pct:.1f}%)".replace(".", ","),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if pct > 55 else "#17263a",
            )
    return image


def main():
    predictions = pd.read_csv(XTCN)
    expected = {"seco": (6507, 10185), "umido": (4152, 1271)}
    for condition, name in [("seco", "solo_seco"), ("umido", "solo_umido")]:
        matrices = [read_xtcn(condition, predictions), read_reference(condition)]
        for matrix in matrices:
            assert matrix.shape == (2, 2)
            assert tuple(matrix.sum(axis=1)) == expected[condition]

        fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.6), layout="constrained")
        for ax, matrix, title in zip(
            axes, matrices, ["(a) X-TCN", "(b) Random Forest"]
        ):
            image = draw_matrix(ax, matrix, title)
        fig.colorbar(
            image,
            ax=axes,
            label="Percentual da classe real (%)",
            shrink=0.72,
            pad=0.025,
        )
        for extension in ("png", "pdf"):
            fig.savefig(OUT / f"matrizes_{name}.{extension}", dpi=350)
        plt.close(fig)
        print(name, "X-TCN", matrices[0].tolist(), "Extra Trees", matrices[1].tolist())


if __name__ == "__main__":
    main()
