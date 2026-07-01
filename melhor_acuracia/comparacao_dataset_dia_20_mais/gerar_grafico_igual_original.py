from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "melhor_acuracia" / "comparacao_dataset_dia_20_mais" / "graficos_igual_original"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
CLASS_LABELS = {0: "0 - doente", 1: "1 - saudavel"}
COLORS = {"Treino": "#2d6cdf", "Teste": "#ff7f0e"}
TRAIN_RATIO = 0.70
RANDOM_STATE = 42


def split_by_coleta(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    summary_rows = []
    cursor = 0

    for classe, class_df in df.groupby("Classe", sort=True):
        class_df = class_df.copy()
        groups = (
            pd.Series(class_df["Coleta"].dropna().unique())
            .sample(frac=1, random_state=RANDOM_STATE)
            .tolist()
        )
        train_count = int(len(groups) * TRAIN_RATIO)
        train_groups = set(groups[:train_count])

        for coleta, coleta_df in class_df.groupby("Coleta", sort=False):
            subset = coleta_df.copy()
            conjunto = "Treino" if coleta in train_groups else "Teste"
            subset["Conjunto"] = conjunto
            start = cursor
            end = cursor + len(subset) - 1
            summary_rows.append(
                {
                    "Classe": int(classe),
                    "Coleta": coleta,
                    "Conjunto": conjunto,
                    "linhas": int(len(subset)),
                    "linha_inicial": int(start),
                    "linha_final": int(end),
                }
            )
            cursor = end + 1
            parts.append(subset)

    ordered = pd.concat(parts, ignore_index=True)
    split = pd.DataFrame(summary_rows)
    return ordered, split


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df = df[df["Classe"].notna()].copy()
    df["Classe"] = df["Classe"].astype(int)
    df = df.sort_values(["Classe"], kind="stable").reset_index(drop=True)
    return split_by_coleta(df)


def normalize_series(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return series * 0
    return (series - min_value) / (max_value - min_value)


def normalize_series_with_bounds(
    series: pd.Series, min_value: float, max_value: float
) -> pd.Series:
    if max_value == min_value:
        return series * 0
    return (series - min_value) / (max_value - min_value)


def draw_dataset_panel(fig, axes, df: pd.DataFrame, split: pd.DataFrame, title: str) -> None:
    ax_top, ax_mid, ax_bottom = axes

    for _, row in split.iterrows():
        color = COLORS[row["Conjunto"]]
        y = 1 if int(row["Classe"]) == 0 else 0
        ax_top.barh(
            y,
            row["linhas"],
            left=row["linha_inicial"],
            height=0.62,
            color=color,
            edgecolor="white",
            linewidth=0.45,
        )

    ax_top.set_yticks([1, 0], [CLASS_LABELS[0], CLASS_LABELS[1]])
    ax_top.set_xlim(0, len(df))
    ax_top.set_title("Recortes usados no split 70/30 por Coleta", fontsize=11)
    ax_top.grid(axis="x", alpha=0.22)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["Treino"], label="Treino"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["Teste"], label="Teste"),
    ]
    ax_top.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)

    for _, row in split.iterrows():
        ax_mid.axvspan(
            row["linha_inicial"],
            row["linha_final"],
            color=COLORS[row["Conjunto"]],
            alpha=0.11,
            linewidth=0,
        )
        ax_mid.axvline(row["linha_inicial"], color="#d9d9d9", linewidth=0.45, alpha=0.8)

    offsets = np.arange(len(FEATURES))
    for offset, feature in zip(offsets, FEATURES):
        ax_mid.plot(
            np.arange(len(df)),
            normalize_series(df[feature]) + offset,
            linewidth=0.8,
            label=feature,
        )

    ax_mid.set_yticks(offsets + 0.5, FEATURES)
    ax_mid.set_xlim(0, len(df))
    ax_mid.set_ylabel("Sensor normalizado + deslocamento")
    ax_mid.set_xlabel("Indice da linha no dataset")
    ax_mid.set_title("Leituras MQ normalizadas ao longo do dataset", fontsize=11)
    ax_mid.grid(axis="x", alpha=0.18)
    ax_mid.legend(loc="upper right", ncol=3, fontsize=8, frameon=True)

    grouped = (
        split.groupby(["Classe", "Conjunto"])["linhas"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=[0, 1], columns=["Treino", "Teste"], fill_value=0)
    )
    x = np.arange(len(grouped))
    treino = grouped["Treino"].values
    teste = grouped["Teste"].values
    bars_train = ax_bottom.bar(x, treino, color=COLORS["Treino"], label="Treino")
    bars_test = ax_bottom.bar(x, teste, bottom=treino, color=COLORS["Teste"], label="Teste")
    ax_bottom.set_xticks(x, [CLASS_LABELS[int(i)] for i in grouped.index])
    ax_bottom.set_ylabel("Linhas")
    ax_bottom.set_title("Quantidade de linhas por classe e conjunto", fontsize=11)
    ax_bottom.grid(axis="y", alpha=0.22)

    for bar, value in zip(bars_train, treino):
        ax_bottom.text(
            bar.get_x() + bar.get_width() / 2,
            value / 2,
            f"Treino\n{int(value)}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )
    for bar, base, value in zip(bars_test, treino, teste):
        ax_bottom.text(
            bar.get_x() + bar.get_width() / 2,
            base + value / 2,
            f"Teste\n{int(value)}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )

    ax_top.text(
        0,
        1.32,
        title,
        transform=ax_top.transAxes,
        fontsize=15,
        ha="left",
        va="bottom",
        fontweight="bold",
    )


def draw_dataset_panel_global_norm(
    fig,
    axes,
    df: pd.DataFrame,
    split: pd.DataFrame,
    title: str,
    bounds: dict[str, tuple[float, float]],
) -> None:
    ax_top, ax_mid, ax_bottom = axes

    for _, row in split.iterrows():
        color = COLORS[row["Conjunto"]]
        y = 1 if int(row["Classe"]) == 0 else 0
        ax_top.barh(
            y,
            row["linhas"],
            left=row["linha_inicial"],
            height=0.62,
            color=color,
            edgecolor="white",
            linewidth=0.45,
        )

    ax_top.set_yticks([1, 0], [CLASS_LABELS[0], CLASS_LABELS[1]])
    ax_top.set_xlim(0, len(df))
    ax_top.set_title("Recortes usados no split 70/30 por Coleta", fontsize=11)
    ax_top.grid(axis="x", alpha=0.22)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["Treino"], label="Treino"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["Teste"], label="Teste"),
    ]
    ax_top.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)

    for _, row in split.iterrows():
        ax_mid.axvspan(
            row["linha_inicial"],
            row["linha_final"],
            color=COLORS[row["Conjunto"]],
            alpha=0.11,
            linewidth=0,
        )
        ax_mid.axvline(row["linha_inicial"], color="#d9d9d9", linewidth=0.45, alpha=0.8)

    offsets = np.arange(len(FEATURES)) * 1.35
    for offset, feature in zip(offsets, FEATURES):
        low, high = bounds[feature]
        y = normalize_series_with_bounds(df[feature], low, high) + offset
        ax_mid.plot(
            np.arange(len(df)),
            y,
            linewidth=0.95,
            label=feature,
        )
        ax_mid.axhline(offset, color="#e0e0e0", linewidth=0.45, alpha=0.75)
        ax_mid.axhline(offset + 1, color="#eeeeee", linewidth=0.45, alpha=0.75)

    ax_mid.set_yticks(offsets + 0.5, FEATURES)
    ax_mid.set_ylim(-0.2, offsets[-1] + 1.25)
    ax_mid.set_xlim(0, len(df))
    ax_mid.set_ylabel("Sensor normalizado globalmente + deslocamento")
    ax_mid.set_xlabel("Indice da linha no dataset")
    ax_mid.set_title("Leituras MQ normalizadas na mesma escala dos dois datasets", fontsize=11)
    ax_mid.grid(axis="x", alpha=0.18)
    ax_mid.legend(loc="upper right", ncol=3, fontsize=8, frameon=True)

    grouped = (
        split.groupby(["Classe", "Conjunto"])["linhas"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=[0, 1], columns=["Treino", "Teste"], fill_value=0)
    )
    x = np.arange(len(grouped))
    treino = grouped["Treino"].values
    teste = grouped["Teste"].values
    bars_train = ax_bottom.bar(x, treino, color=COLORS["Treino"], label="Treino")
    bars_test = ax_bottom.bar(x, teste, bottom=treino, color=COLORS["Teste"], label="Teste")
    ax_bottom.set_xticks(x, [CLASS_LABELS[int(i)] for i in grouped.index])
    ax_bottom.set_ylabel("Linhas")
    ax_bottom.set_title("Quantidade de linhas por classe e conjunto", fontsize=11)
    ax_bottom.grid(axis="y", alpha=0.22)

    for bar, value in zip(bars_train, treino):
        ax_bottom.text(
            bar.get_x() + bar.get_width() / 2,
            value / 2,
            f"Treino\n{int(value)}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )
    for bar, base, value in zip(bars_test, treino, teste):
        ax_bottom.text(
            bar.get_x() + bar.get_width() / 2,
            base + value / 2,
            f"Teste\n{int(value)}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )

    ax_top.text(
        0,
        1.32,
        title,
        transform=ax_top.transAxes,
        fontsize=15,
        ha="left",
        va="bottom",
        fontweight="bold",
    )


def save_single(name: str, df: pd.DataFrame, split: pd.DataFrame, title: str) -> None:
    fig = plt.figure(figsize=(20, 12), constrained_layout=False)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 2.8, 1.2], hspace=0.95)
    axes = [fig.add_subplot(grid[i, 0]) for i in range(3)]
    draw_dataset_panel(fig, axes, df, split, title)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sem_df, sem_split = load_dataset(ROOT / "sem pressao" / "dataset_sem_pressao.csv")
    dia_df, dia_split = load_dataset(ROOT / "dia_20_mais" / "dia_20_mais" / "dataset_dia_20_mais.csv")
    combined_for_bounds = pd.concat([sem_df[FEATURES], dia_df[FEATURES]], ignore_index=True)
    global_bounds = {
        feature: (
            float(combined_for_bounds[feature].quantile(0.01)),
            float(combined_for_bounds[feature].quantile(0.99)),
        )
        for feature in FEATURES
    }

    sem_split.to_csv(OUT / "coletas_split_treino_teste_sem_pressao.csv", index=False)
    dia_split.to_csv(OUT / "coletas_split_treino_teste_dia_20_mais.csv", index=False)

    save_single(
        "grafico_dataset_treino_teste_sem_pressao_recriado.png",
        sem_df,
        sem_split,
        "Dataset sem pressao: sensores MQ e recortes de treino/teste",
    )
    save_single(
        "grafico_dataset_treino_teste_dia_20_mais_igual_original.png",
        dia_df,
        dia_split,
        "Dataset dia_20_mais: sensores MQ e recortes de treino/teste",
    )

    fig = plt.figure(figsize=(20, 22), constrained_layout=False)
    grid = fig.add_gridspec(
        6,
        1,
        height_ratios=[1.0, 2.8, 1.2, 1.0, 2.8, 1.2],
        hspace=0.95,
    )
    axes_sem = [fig.add_subplot(grid[i, 0]) for i in range(3)]
    axes_dia = [fig.add_subplot(grid[i, 0]) for i in range(3, 6)]
    draw_dataset_panel(
        fig,
        axes_sem,
        sem_df,
        sem_split,
        "Dataset sem pressao: sensores MQ e recortes de treino/teste",
    )
    draw_dataset_panel(
        fig,
        axes_dia,
        dia_df,
        dia_split,
        "Dataset dia_20_mais: sensores MQ e recortes de treino/teste",
    )
    fig.savefig(
        OUT / "comparacao_datasets_igual_grafico_original.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = plt.figure(figsize=(20, 22), constrained_layout=False)
    grid = fig.add_gridspec(
        6,
        1,
        height_ratios=[1.0, 3.2, 1.2, 1.0, 3.2, 1.2],
        hspace=0.95,
    )
    axes_sem = [fig.add_subplot(grid[i, 0]) for i in range(3)]
    axes_dia = [fig.add_subplot(grid[i, 0]) for i in range(3, 6)]
    draw_dataset_panel_global_norm(
        fig,
        axes_sem,
        sem_df,
        sem_split,
        "Dataset sem pressao: sensores MQ e recortes de treino/teste",
        global_bounds,
    )
    draw_dataset_panel_global_norm(
        fig,
        axes_dia,
        dia_df,
        dia_split,
        "Dataset dia_20_mais: sensores MQ e recortes de treino/teste",
        global_bounds,
    )
    fig.savefig(
        OUT / "comparacao_datasets_normalizacao_global_melhor_curvas.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Graficos gerados em:", OUT)
    for path in sorted(OUT.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
