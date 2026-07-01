from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "comparacao" / "coletas_numeradas"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ["Soil", "Temp.", "Pres.", "MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Classe"].notna()].copy()
    df["Classe"] = df["Classe"].astype(int)
    return df.reset_index(drop=True)


def build_segments(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    cursor = 0
    for idx, (coleta, block) in enumerate(df.groupby("Coleta", sort=False), 1):
        start = cursor
        end = cursor + len(block) - 1
        rows.append(
            {
                "dataset": dataset_name,
                "numero_coleta": idx,
                "Coleta": coleta,
                "Dia": block["Dia"].dropna().iloc[0] if block["Dia"].notna().any() else None,
                "Vaso": block["Vaso"].dropna().iloc[0] if block["Vaso"].notna().any() else None,
                "Classe": block["Classe"].dropna().iloc[0] if block["Classe"].notna().any() else None,
                "linha_inicio": start,
                "linha_fim": end,
                "linhas": len(block),
            }
        )
        cursor = end + 1
    return pd.DataFrame(rows)


def plot_dataset(df: pd.DataFrame, segments: pd.DataFrame, dataset_name: str, title: str) -> None:
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(22, 19), sharex=True)
    x = np.arange(len(df))

    for ax, feature in zip(axes, FEATURES):
        y = pd.to_numeric(df[feature], errors="coerce").rolling(21, center=True, min_periods=1).mean()
        ax.plot(x, y, color="#26364f", linewidth=0.85)
        ax.set_ylabel(feature)
        ax.grid(axis="y", alpha=0.2)

        for _, row in segments.iterrows():
            ax.axvline(row["linha_inicio"], color="#c94b4b", linewidth=0.85, alpha=0.55)

    top_ax = axes[0]
    y_top = top_ax.get_ylim()[1]
    y_range = top_ax.get_ylim()[1] - top_ax.get_ylim()[0]
    for _, row in segments.iterrows():
        mid = (row["linha_inicio"] + row["linha_fim"]) / 2
        top_ax.text(
            mid,
            y_top + y_range * 0.03,
            f"C{int(row['numero_coleta'])}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#b33838",
            rotation=0,
        )

    axes[0].set_title(title, fontsize=16, pad=28)
    axes[-1].set_xlabel("Indice da linha no dataset")
    fig.tight_layout()
    fig.savefig(OUT / f"{dataset_name}_curvas_com_coletas_numeradas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    datasets = [
        (
            "antes_dia_20",
            ROOT / "comparacao" / "datasets_com_ambiente" / "antes_dia_20_com_ambiente.csv",
            "Antes do dia 20 - curvas cruas com coletas numeradas",
        ),
        (
            "dia_20_mais",
            ROOT / "comparacao" / "datasets_com_ambiente" / "dia_20_mais_com_ambiente.csv",
            "Dia 20 em diante - curvas cruas com coletas numeradas",
        ),
    ]
    all_segments = []
    for dataset_name, path, title in datasets:
        df = load_dataset(path)
        segments = build_segments(df, dataset_name)
        segments.to_csv(OUT / f"mapa_coletas_{dataset_name}.csv", index=False)
        plot_dataset(df, segments, dataset_name, title)
        all_segments.append(segments)

    pd.concat(all_segments, ignore_index=True).to_csv(OUT / "mapa_coletas_todos.csv", index=False)
    print("Arquivos gerados em:", OUT)
    for path in sorted(OUT.iterdir()):
        print(path.name)


if __name__ == "__main__":
    main()
