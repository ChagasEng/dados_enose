from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "extra_trees_pressao_filtrada"
OUT = BASE_DIR / "graficos"
OUT.mkdir(parents=True, exist_ok=True)

DATASET = ROOT / "comparacao" / "pressao_filtrada" / "antes_dia_20_pressao_filtrada_estrito.csv"
MAPA_OUT = OUT / "mapa_coletas_nematoide_pressao_filtrada.csv"
FIG_OUT = OUT / "antes_dia_20_pressao_filtrada_curvas_coletas_por_nematoide.png"

FEATURES = ["Soil", "Temp.", "Pres.", "MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
CLASS_LABEL = {0: "Com nematoide", 1: "Sem nematoide"}
CLASS_COLOR = {0: "#f4b6b0", 1: "#bde5c8"}
CLASS_LINE = {0: "#b33838", 1: "#2f7d56"}


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Classe"].notna()].copy()
    df["Classe"] = df["Classe"].astype(int)
    return df.reset_index(drop=True)


def build_segments(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cursor = 0
    for index, (coleta, block) in enumerate(df.groupby("Coleta", sort=False), 1):
        start = cursor
        end = cursor + len(block) - 1
        classe = int(block["Classe"].mode().iloc[0])
        rows.append(
            {
                "dataset": "antes_dia_20_pressao_filtrada",
                "numero_coleta": index,
                "condicao": CLASS_LABEL[classe],
                "Coleta": coleta,
                "Dia": block["Dia"].dropna().iloc[0] if block["Dia"].notna().any() else None,
                "Vaso": block["Vaso"].dropna().iloc[0] if block["Vaso"].notna().any() else None,
                "Classe": classe,
                "linha_inicio": start,
                "linha_fim": end,
                "linhas": len(block),
            }
        )
        cursor = end + 1
    return pd.DataFrame(rows)


def plot_dataset(df: pd.DataFrame, segments: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(22, 19), sharex=True)
    x = np.arange(len(df))

    for ax, feature in zip(axes, FEATURES):
        for _, row in segments.iterrows():
            classe = int(row["Classe"])
            ax.axvspan(
                row["linha_inicio"],
                row["linha_fim"],
                color=CLASS_COLOR[classe],
                alpha=0.28,
                linewidth=0,
            )
            ax.axvline(
                row["linha_inicio"],
                color=CLASS_LINE[classe],
                linewidth=0.85,
                alpha=0.75,
            )

        y = pd.to_numeric(df[feature], errors="coerce").rolling(21, center=True, min_periods=1).mean()
        ax.plot(x, y, color="#26364f", linewidth=0.9)
        ax.set_ylabel(feature)
        ax.grid(axis="y", alpha=0.2)

    top_ax = axes[0]
    y_min, y_max = top_ax.get_ylim()
    y_range = y_max - y_min
    for _, row in segments.iterrows():
        classe = int(row["Classe"])
        mid = (row["linha_inicio"] + row["linha_fim"]) / 2
        top_ax.text(
            mid,
            y_max + y_range * 0.03,
            f"C{int(row['numero_coleta'])}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=CLASS_LINE[classe],
            fontweight="bold",
        )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[0], alpha=0.7, label="Com nematoide"),
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[1], alpha=0.7, label="Sem nematoide"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", frameon=True)
    axes[0].set_title(
        "Antes do dia 20 com pressao filtrada - coletas demarcadas por presenca de nematoide",
        fontsize=16,
        pad=28,
    )
    axes[-1].set_xlabel("Indice da linha no dataset")
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_dataset(DATASET)
    segments = build_segments(df)
    segments.to_csv(MAPA_OUT, index=False)
    plot_dataset(df, segments)

    print(f"Dataset usado: {DATASET}")
    print(f"Linhas no grafico: {len(df)}")
    print(f"Coletas: {len(segments)}")
    print(f"Mapa salvo em: {MAPA_OUT}")
    print(f"Grafico salvo em: {FIG_OUT}")


if __name__ == "__main__":
    main()
