"""Gera o painel de coletas colorido por presenca de nematoide."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
DATASET = BASE / "dados" / "dataset_dia_20_mais_com_ambiente.csv"
OUT = BASE / "graficos"
MAPA_OUT = OUT / "mapa_coletas_nematoide_dia_20_mais.csv"
FIG_OUT = OUT / "coletas_por_nematoide_dia_20_mais_estilo_06_07.png"

FEATURES = ["Soil", "Temp.", "Pres.", "MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
CLASS_LABEL = {0: "Com nematoide", 1: "Sem nematoide"}
CLASS_COLOR = {0: "#f7d8d5", 1: "#dcefe3"}
CLASS_LINE = {0: "#e57373", 1: "#6ab187"}


def build_segments(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cursor = 0
    for number, (coleta, block) in enumerate(df.groupby("Coleta", sort=False), start=1):
        classe = int(block["Classe"].mode().iloc[0])
        start = cursor
        end = cursor + len(block) - 1
        rows.append(
            {
                "Coleta_plot": f"C{number}",
                "Coleta": coleta,
                "Classe": classe,
                "Nematoide": CLASS_LABEL[classe],
                "inicio": start,
                "fim": end,
                "linhas": len(block),
            }
        )
        cursor = end + 1
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, segments: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(18, 20), sharex=True)
    x = np.arange(len(df))

    for ax, feature in zip(axes, FEATURES):
        for _, segment in segments.iterrows():
            classe = int(segment["Classe"])
            ax.axvspan(segment["inicio"], segment["fim"], color=CLASS_COLOR[classe], alpha=0.62)
            ax.axvline(segment["inicio"], color=CLASS_LINE[classe], linewidth=0.8, alpha=0.72)

        values = pd.to_numeric(df[feature], errors="coerce").rolling(35, center=True, min_periods=1).median()
        ax.plot(x, values, color="#34495e", linewidth=0.85)
        ax.set_ylabel(feature, fontsize=9)
        ax.grid(axis="y", alpha=0.16)
        ax.set_xlim(0, len(df))

    for _, segment in segments.iterrows():
        classe = int(segment["Classe"])
        axes[0].text(
            (segment["inicio"] + segment["fim"]) / 2,
            1.012,
            segment["Coleta_plot"],
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=CLASS_LINE[classe],
            fontweight="bold",
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[0], alpha=0.8, label="Com nematoide"),
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[1], alpha=0.8, label="Sem nematoide"),
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Indice da linha no dataset")
    fig.suptitle("Dia 20+ - coletas demarcadas por nematoide com MQ + ambiente", y=0.996)
    fig.tight_layout(rect=[0, 0, 1, 0.982])
    fig.savefig(FIG_OUT, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET)
    segments = build_segments(df)
    segments.to_csv(MAPA_OUT, index=False, encoding="utf-8-sig")
    plot(df, segments)
    print(f"Grafico salvo em: {FIG_OUT}")


if __name__ == "__main__":
    main()
