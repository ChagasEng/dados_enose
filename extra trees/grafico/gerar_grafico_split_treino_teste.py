from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
OUTPUT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "sem pressao" / "dataset_sem_pressao.csv"

GROUP_COLUMN = "Coleta"
TRAIN_RATIO = 0.70
RANDOM_STATE = 42

TRAIN_COLOR = "#1f66e5"
TEST_COLOR = "#ff7f0e"
SIGNAL_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]


def find_target_column(columns: list[str]) -> str:
    for column in columns:
        if column.lower() == "classe":
            return column
    raise ValueError("Coluna alvo 'classe' nao encontrada.")


def load_dataset() -> tuple[pd.DataFrame, list[str], str]:
    raw_df = pd.read_csv(DATASET_PATH)
    raw_df.insert(0, "linha_dataset", raw_df.index)

    target_column = find_target_column(raw_df.columns.tolist())
    mq_columns = [column for column in raw_df.columns if column.upper().startswith("MQ")]

    df = raw_df[["linha_dataset", GROUP_COLUMN] + mq_columns + [target_column]].copy()
    for column in mq_columns + [target_column]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[GROUP_COLUMN] + mq_columns + [target_column])
    df[target_column] = df[target_column].astype(int)
    return df.sort_values("linha_dataset").reset_index(drop=True), mq_columns, target_column


def apply_split(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_parts = []
    group_rows = []

    for class_value in sorted(df[target_column].unique()):
        class_block = df[df[target_column] == class_value].copy()
        groups = (
            pd.Series(class_block[GROUP_COLUMN].unique())
            .sample(frac=1, random_state=RANDOM_STATE)
            .tolist()
        )
        train_group_count = int(len(groups) * TRAIN_RATIO)
        train_groups = set(groups[:train_group_count])

        class_block["Conjunto"] = np.where(
            class_block[GROUP_COLUMN].isin(train_groups), "Treino", "Teste"
        )
        split_parts.append(class_block)

        for group in groups:
            group_block = class_block[class_block[GROUP_COLUMN] == group]
            group_rows.append(
                {
                    "Classe": int(class_value),
                    "Coleta": group,
                    "Conjunto": "Treino" if group in train_groups else "Teste",
                    "linhas": int(len(group_block)),
                    "linha_inicial": int(group_block["linha_dataset"].min()),
                    "linha_final": int(group_block["linha_dataset"].max()),
                }
            )

    split_df = pd.concat(split_parts).sort_values("linha_dataset").reset_index(drop=True)
    groups_df = pd.DataFrame(group_rows).sort_values(["Classe", "linha_inicial"])
    return split_df, groups_df


def contiguous_ranges(df: pd.DataFrame, value_column: str) -> list[dict[str, object]]:
    rows = df[["linha_dataset", value_column]].sort_values("linha_dataset").to_dict("records")
    ranges = []
    start = int(rows[0]["linha_dataset"])
    previous_x = start
    current_value = rows[0][value_column]

    for row in rows[1:]:
        x = int(row["linha_dataset"])
        value = row[value_column]
        if value != current_value or x != previous_x + 1:
            ranges.append({"inicio": start, "fim": previous_x, "valor": current_value})
            start = x
            current_value = value
        previous_x = x

    ranges.append({"inicio": start, "fim": previous_x, "valor": current_value})
    return ranges


def save_csv_outputs(split_df: pd.DataFrame, groups_df: pd.DataFrame, target_column: str) -> None:
    split_df.to_csv(OUTPUT_DIR / "dataset_com_split_treino_teste.csv", index=False)
    groups_df.to_csv(OUTPUT_DIR / "coletas_split_treino_teste.csv", index=False)

    line_summary = (
        split_df.groupby(["Conjunto", target_column])
        .size()
        .reset_index(name="linhas")
        .rename(columns={target_column: "Classe"})
    )
    group_summary = (
        groups_df.groupby(["Conjunto", "Classe"])["Coleta"]
        .nunique()
        .reset_index(name="coletas")
    )
    line_summary.merge(group_summary, on=["Conjunto", "Classe"]).sort_values(
        ["Classe", "Conjunto"]
    ).to_csv(OUTPUT_DIR / "resumo_split_treino_teste.csv", index=False)


def plot_split_ranges(ax: plt.Axes, groups_df: pd.DataFrame) -> None:
    labels = {0: "0 - doente", 1: "1 - saudavel"}
    y_positions = {0: 10, 1: 0}

    for _, row in groups_df.iterrows():
        color = TRAIN_COLOR if row["Conjunto"] == "Treino" else TEST_COLOR
        ax.broken_barh(
            [(row["linha_inicial"], row["linha_final"] - row["linha_inicial"] + 1)],
            (y_positions[int(row["Classe"])], 7),
            facecolors=color,
            edgecolors="white",
            linewidth=0.3,
        )

    ax.set_yticks([13.5, 3.5])
    ax.set_yticklabels([labels[0], labels[1]])
    ax.set_title("Recortes usados no split 70/30 por Coleta", fontsize=10)
    ax.set_xlim(0, int(groups_df["linha_final"].max()))
    ax.grid(axis="x", alpha=0.18)
    ax.legend(
        handles=[
            mpatches.Patch(color=TRAIN_COLOR, label="Treino"),
            mpatches.Patch(color=TEST_COLOR, label="Teste"),
        ],
        loc="upper right",
        fontsize=8,
    )


def plot_sensor_lines(ax: plt.Axes, split_df: pd.DataFrame, mq_columns: list[str]) -> None:
    for split_range in contiguous_ranges(split_df, "Conjunto"):
        color = TRAIN_COLOR if split_range["valor"] == "Treino" else TEST_COLOR
        ax.axvspan(split_range["inicio"], split_range["fim"], color=color, alpha=0.09)

    for index, column in enumerate(mq_columns):
        series = split_df[column].astype(float)
        min_value = series.min()
        max_value = series.max()
        denominator = max(max_value - min_value, 1e-9)
        normalized = ((series - min_value) / denominator) * 0.70 + index
        ax.plot(
            split_df["linha_dataset"],
            normalized,
            label=column,
            linewidth=0.85,
            color=SIGNAL_COLORS[index % len(SIGNAL_COLORS)],
        )

    ax.set_yticks(range(len(mq_columns)))
    ax.set_yticklabels(mq_columns)
    ax.set_ylabel("Sensor normalizado + deslocamento")
    ax.set_title("Leituras MQ normalizadas ao longo do dataset", fontsize=10)
    ax.grid(axis="x", alpha=0.18)
    ax.legend(loc="upper right", ncol=3, fontsize=7)


def plot_quantity_bars(ax: plt.Axes, split_df: pd.DataFrame, target_column: str) -> None:
    labels = {0: "0 - doente", 1: "1 - saudavel"}
    pivot = (
        split_df.groupby([target_column, "Conjunto"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Treino", "Teste"], fill_value=0)
    )

    x = np.arange(len(pivot.index))
    treino = pivot["Treino"].to_numpy()
    teste = pivot["Teste"].to_numpy()
    ax.bar(x, treino, color=TRAIN_COLOR, label="Treino")
    ax.bar(x, teste, bottom=treino, color=TEST_COLOR, label="Teste")

    for i, class_value in enumerate(pivot.index):
        ax.text(i, treino[i] / 2, f"Treino\n{treino[i]}", ha="center", va="center", color="white", fontsize=8)
        ax.text(
            i,
            treino[i] + teste[i] / 2,
            f"Teste\n{teste[i]}",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(int(value), str(value)) for value in pivot.index])
    ax.set_ylabel("Linhas")
    ax.set_title("Quantidade de linhas por classe e conjunto", fontsize=10)
    ax.grid(axis="y", alpha=0.18)


def plot_dashboard(split_df: pd.DataFrame, groups_df: pd.DataFrame, mq_columns: list[str], target_column: str) -> None:
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.15, 3.2, 1.35], hspace=0.42)
    ax_ranges = fig.add_subplot(grid[0])
    ax_lines = fig.add_subplot(grid[1], sharex=ax_ranges)
    ax_bars = fig.add_subplot(grid[2])

    fig.suptitle("Dataset sem pressao: sensores MQ e recortes de treino/teste", fontsize=13)
    plot_split_ranges(ax_ranges, groups_df)
    plot_sensor_lines(ax_lines, split_df, mq_columns)
    plot_quantity_bars(ax_bars, split_df, target_column)
    ax_lines.set_xlabel("Indice da linha no dataset")

    fig.savefig(OUTPUT_DIR / "grafico_dataset_treino_teste.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_df, mq_columns, target_column = load_dataset()
    split_df, groups_df = apply_split(split_df, target_column)

    save_csv_outputs(split_df, groups_df, target_column)
    plot_dashboard(split_df, groups_df, mq_columns, target_column)

    print(f"Grafico: {OUTPUT_DIR / 'grafico_dataset_treino_teste.png'}")
    print(f"Dataset marcado: {OUTPUT_DIR / 'dataset_com_split_treino_teste.csv'}")
    print(f"Coletas marcadas: {OUTPUT_DIR / 'coletas_split_treino_teste.csv'}")


if __name__ == "__main__":
    main()
