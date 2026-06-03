from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
DATASET_PATH = ROOT_DIR / "sem pressao" / "dataset_sem_pressao.csv"
OUTPUT_IMAGE = PROJECT_DIR / "grafico_dataset_split_treino_teste.png"
OUTPUT_SPLIT = PROJECT_DIR / "split_treino_teste_por_coleta.csv"

GROUP_COLUMN = "Coleta"
TRAIN_RATIO = 0.70
RANDOM_STATE = 42

COLORS = {
    "Treino": "#2563eb",
    "Teste": "#f97316",
}


def find_target_column(columns: list[str]) -> str:
    for column in columns:
        if column.lower() == "classe":
            return column
    raise ValueError("Coluna alvo 'classe' nao encontrada.")


def build_split(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_parts = []
    split_summary = []

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

        grouped = (
            class_block.groupby([GROUP_COLUMN, target_column, "Conjunto"], sort=False)
            .agg(
                indice_inicio=("indice_original", "min"),
                indice_fim=("indice_original", "max"),
                linhas=("indice_original", "count"),
            )
            .reset_index()
        )
        split_summary.append(grouped)

    split_df = pd.concat(split_parts).sort_values("indice_original")
    summary_df = pd.concat(split_summary).sort_values("indice_inicio")
    return split_df, summary_df


def normalize_sensor_data(df: pd.DataFrame, mq_columns: list[str]) -> pd.DataFrame:
    sensor_df = df[["indice_original"] + mq_columns].copy()
    for column in mq_columns:
        values = sensor_df[column].astype(float)
        min_value = values.min()
        max_value = values.max()
        sensor_df[column] = (values - min_value) / (max_value - min_value)
    return sensor_df


def plot_split(split_df: pd.DataFrame, summary_df: pd.DataFrame, mq_columns: list[str]) -> None:
    class_labels = {
        0: "0 - doente",
        1: "1 - saudavel",
    }

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.2, 3.0, 1.1], hspace=0.34)
    ax_timeline = fig.add_subplot(grid[0])
    ax_sensors = fig.add_subplot(grid[1], sharex=ax_timeline)
    ax_counts = fig.add_subplot(grid[2])

    for _, row in summary_df.iterrows():
        class_value = int(row["Classe"])
        set_name = row["Conjunto"]
        start = int(row["indice_inicio"])
        end = int(row["indice_fim"])
        width = end - start + 1
        y_center = 1 - class_value
        ax_timeline.broken_barh(
            [(start, width)],
            (y_center - 0.28, 0.56),
            facecolors=COLORS[set_name],
            edgecolors="white",
            linewidth=0.7,
        )
        ax_sensors.axvspan(start, end, color=COLORS[set_name], alpha=0.08, linewidth=0)

    ax_timeline.set_title("Recortes usados no split 70/30 por Coleta")
    ax_timeline.set_yticks([1, 0], [class_labels[0], class_labels[1]])
    ax_timeline.set_ylabel("Classe")
    ax_timeline.grid(axis="x", alpha=0.18)

    sensor_df = normalize_sensor_data(split_df, mq_columns)
    sensor_df = sensor_df.rolling(window=120, min_periods=1).mean()
    sensor_df = sensor_df.iloc[::25]

    for offset, column in enumerate(mq_columns):
        ax_sensors.plot(
            sensor_df["indice_original"],
            sensor_df[column] + offset,
            linewidth=1.0,
            label=column,
        )

    ax_sensors.set_title("Leituras MQ normalizadas ao longo do dataset")
    ax_sensors.set_ylabel("Sensor normalizado + deslocamento")
    ax_sensors.set_xlabel("Indice da linha no dataset")
    ax_sensors.set_yticks(range(len(mq_columns)), mq_columns)
    ax_sensors.grid(alpha=0.18)
    ax_sensors.legend(loc="upper right", ncol=3, frameon=True)

    counts = (
        split_df.groupby(["Classe", "Conjunto"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=[0, 1], columns=["Treino", "Teste"])
    )
    x = np.arange(len(counts.index))
    train_counts = counts["Treino"].to_numpy()
    test_counts = counts["Teste"].to_numpy()
    ax_counts.bar(x, train_counts, color=COLORS["Treino"], label="Treino")
    ax_counts.bar(x, test_counts, bottom=train_counts, color=COLORS["Teste"], label="Teste")
    ax_counts.set_xticks(x, [class_labels[int(value)] for value in counts.index])
    ax_counts.set_ylabel("Linhas")
    ax_counts.set_title("Quantidade de linhas por classe e conjunto")
    ax_counts.grid(axis="y", alpha=0.18)

    for index, (train_count, test_count) in enumerate(zip(train_counts, test_counts)):
        ax_counts.text(index, train_count / 2, f"Treino\n{train_count}", ha="center", va="center", color="white")
        ax_counts.text(index, train_count + test_count / 2, f"Teste\n{test_count}", ha="center", va="center", color="white")

    legend_handles = [
        mpatches.Patch(color=COLORS["Treino"], label="Treino"),
        mpatches.Patch(color=COLORS["Teste"], label="Teste"),
    ]
    ax_timeline.legend(handles=legend_handles, loc="upper right", frameon=True)

    fig.suptitle(
        "Dataset sem pressao: sensores MQ e recortes de treino/teste",
        fontsize=16,
        y=0.98,
    )
    fig.savefig(OUTPUT_IMAGE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_PATH)
    df = df.reset_index().rename(columns={"index": "indice_original"})
    target_column = find_target_column(df.columns.tolist())
    mq_columns = [column for column in df.columns if column.upper().startswith("MQ")]

    for column in mq_columns + [target_column]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[GROUP_COLUMN] + mq_columns + [target_column]).copy()
    df[target_column] = df[target_column].astype(int)

    split_df, summary_df = build_split(df, target_column)
    summary_df.to_csv(OUTPUT_SPLIT, index=False)
    plot_split(split_df, summary_df, mq_columns)

    print(f"Grafico salvo em: {OUTPUT_IMAGE}")
    print(f"Resumo do split salvo em: {OUTPUT_SPLIT}")
    print(split_df.groupby([target_column, "Conjunto"]).size().to_string())


if __name__ == "__main__":
    main()
