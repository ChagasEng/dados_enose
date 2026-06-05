from itertools import combinations
from pathlib import Path
import re

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import kruskal, pearsonr


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ANALYSIS_DIR = Path(__file__).resolve().parent
DATASET_PATH = ANALYSIS_DIR.parent / "dataset_sem_pressao.csv"
MQ_COLUMNS = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]


def extract_vase_from_collection(collection: str) -> float:
    match = re.search(r"Vaso\s+(\d+)\s*$", str(collection))
    if not match:
        return np.nan
    return int(match.group(1))


def group_from_vase(vase: int) -> str:
    if 1 <= vase <= 5:
        return "saudavel_vasos_1_a_5"
    if 6 <= vase <= 10:
        return "doente_vasos_6_a_10"
    return "fora_do_mapeamento"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df.columns = [column.replace("\ufeff", "") for column in df.columns]

    for column in MQ_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Vaso_numero"] = df["Coleta"].apply(extract_vase_from_collection)
    df = df.dropna(subset=["Coleta", "Vaso_numero"] + MQ_COLUMNS).copy()
    df["Vaso_numero"] = df["Vaso_numero"].astype(int)
    df["Grupo_vaso"] = df["Vaso_numero"].apply(group_from_vase)
    df = df[df["Grupo_vaso"] != "fora_do_mapeamento"].copy()

    # Padroniza os sensores para nenhum MQ dominar a comparacao geral so por escala.
    for column in MQ_COLUMNS:
        std = df[column].std(ddof=0)
        df[f"{column}_z"] = (df[column] - df[column].mean()) / std

    z_columns = [f"{column}_z" for column in MQ_COLUMNS]
    df["Indice_MQ_geral"] = df[z_columns].mean(axis=1)
    df["Magnitude_MQ_geral"] = np.sqrt((df[z_columns] ** 2).sum(axis=1))
    df["Amostra_no_vaso"] = df.groupby(["Grupo_vaso", "Vaso_numero"]).cumcount()
    return df


def build_general_pair_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    z_columns = [f"{column}_z" for column in MQ_COLUMNS]

    for group_name, group_df in df.groupby("Grupo_vaso"):
        vessels = sorted(group_df["Vaso_numero"].unique())
        for vaso_a, vaso_b in combinations(vessels, 2):
            block_a = group_df[group_df["Vaso_numero"] == vaso_a]
            block_b = group_df[group_df["Vaso_numero"] == vaso_b]
            merged = block_a[["Amostra_no_vaso"] + z_columns].merge(
                block_b[["Amostra_no_vaso"] + z_columns],
                on="Amostra_no_vaso",
                suffixes=("_a", "_b"),
            )

            profile_a = merged[[f"{column}_a" for column in z_columns]].to_numpy()
            profile_b = merged[[f"{column}_b" for column in z_columns]].to_numpy()
            flat_a = profile_a.reshape(-1)
            flat_b = profile_b.reshape(-1)

            if len(flat_a) < 3 or flat_a.std() == 0 or flat_b.std() == 0:
                correlation = np.nan
                p_value = np.nan
            else:
                correlation, p_value = pearsonr(flat_a, flat_b)

            centroid_a = profile_a.mean(axis=0)
            centroid_b = profile_b.mean(axis=0)
            centroid_distance = float(np.linalg.norm(centroid_a - centroid_b))
            aligned_rmse = float(np.sqrt(np.mean((profile_a - profile_b) ** 2)))

            rows.append(
                {
                    "Grupo_vaso": group_name,
                    "Vaso_A": int(vaso_a),
                    "Vaso_B": int(vaso_b),
                    "n_amostras_alinhadas": int(len(merged)),
                    "correlacao_geral_todos_mq": float(correlation),
                    "p_valor_correlacao_geral": float(p_value),
                    "distancia_media_perfil_mq": centroid_distance,
                    "erro_medio_alinhado_rmse": aligned_rmse,
                }
            )

    return pd.DataFrame(rows)


def build_general_difference_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for group_name, group_df in df.groupby("Grupo_vaso"):
        vessels = sorted(group_df["Vaso_numero"].unique())
        magnitude_arrays = [
            group_df.loc[group_df["Vaso_numero"] == vessel, "Magnitude_MQ_geral"].to_numpy()
            for vessel in vessels
        ]
        index_arrays = [
            group_df.loc[group_df["Vaso_numero"] == vessel, "Indice_MQ_geral"].to_numpy()
            for vessel in vessels
        ]

        mag_h, mag_p = kruskal(*magnitude_arrays)
        idx_h, idx_p = kruskal(*index_arrays)

        for metric_name, arrays, h_stat, p_value in [
            ("Magnitude_MQ_geral", magnitude_arrays, mag_h, mag_p),
            ("Indice_MQ_geral", index_arrays, idx_h, idx_p),
        ]:
            total_n = int(sum(len(array) for array in arrays))
            k = len(arrays)
            epsilon_squared = max((h_stat - k + 1) / (total_n - k), 0) if total_n > k else np.nan
            means = [float(np.mean(array)) for array in arrays]
            rows.append(
                {
                    "Grupo_vaso": group_name,
                    "Metrica_geral": metric_name,
                    "vasos_comparados": ",".join(str(vessel) for vessel in vessels),
                    "kruskal_h": float(h_stat),
                    "p_valor_diferenca_vasos": float(p_value),
                    "epsilon_squared": float(epsilon_squared),
                    "diferenca_media_max_menos_min": float(max(means) - min(means)),
                    "houve_diferenca_estatistica_p_0_05": bool(p_value < 0.05),
                }
            )

    return pd.DataFrame(rows)


def build_vase_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Grupo_vaso", "Vaso_numero"])
        .agg(
            n=("Coleta", "size"),
            indice_mq_geral_media=("Indice_MQ_geral", "mean"),
            indice_mq_geral_mediana=("Indice_MQ_geral", "median"),
            magnitude_mq_geral_media=("Magnitude_MQ_geral", "mean"),
            magnitude_mq_geral_mediana=("Magnitude_MQ_geral", "median"),
        )
        .reset_index()
    )


def save_heatmap(pair_df: pd.DataFrame, group_name: str, output_name: str) -> None:
    group_df = pair_df[pair_df["Grupo_vaso"] == group_name]
    vessels = sorted(set(group_df["Vaso_A"]).union(set(group_df["Vaso_B"])))
    matrix = pd.DataFrame(np.eye(len(vessels)), index=vessels, columns=vessels)

    for _, row in group_df.iterrows():
        matrix.loc[row["Vaso_A"], row["Vaso_B"]] = row["correlacao_geral_todos_mq"]
        matrix.loc[row["Vaso_B"], row["Vaso_A"]] = row["correlacao_geral_todos_mq"]

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    image = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_title(output_name.replace("_", " ").replace(".png", ""))
    ax.set_xticks(np.arange(len(vessels)))
    ax.set_xticklabels(vessels)
    ax.set_yticks(np.arange(len(vessels)))
    ax.set_yticklabels(vessels)
    ax.set_xlabel("Vaso")
    ax.set_ylabel("Vaso")

    for row_index in range(len(vessels)):
        for column_index in range(len(vessels)):
            value = matrix.iloc[row_index, column_index]
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(image, ax=ax, label="correlacao geral")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / output_name, dpi=180)
    plt.close(fig)


def save_bar_plot(pair_df: pd.DataFrame) -> None:
    plot_df = pair_df.copy()
    plot_df["Par"] = plot_df["Vaso_A"].astype(str) + " x " + plot_df["Vaso_B"].astype(str)
    plot_df["Grupo_label"] = plot_df["Grupo_vaso"].map(
        {
            "saudavel_vasos_1_a_5": "saudavel: vasos 1 a 5",
            "doente_vasos_6_a_10": "doente: vasos 6 a 10",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, group_name in zip(
        axes, ["saudavel_vasos_1_a_5", "doente_vasos_6_a_10"]
    ):
        group_plot = plot_df[plot_df["Grupo_vaso"] == group_name].sort_values(
            "correlacao_geral_todos_mq"
        )
        colors = ["#2e7d57" if value >= 0.8 else "#d88725" for value in group_plot["correlacao_geral_todos_mq"]]
        ax.barh(group_plot["Par"], group_plot["correlacao_geral_todos_mq"], color=colors)
        ax.axvline(0.8, color="#2e7d57", linestyle="--", linewidth=1)
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_title(group_plot["Grupo_label"].iloc[0])
        ax.set_xlabel("Correlacao geral usando todos os MQ")
        ax.set_xlim(-1, 1)
        ax.grid(axis="x", alpha=0.2)

    fig.suptitle("Correlacao geral entre pares de vasos", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ANALYSIS_DIR / "grafico_correlacao_geral_pares_vasos.png", dpi=180)
    plt.close(fig)


def save_report(
    pair_df: pd.DataFrame,
    test_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    group_summary = (
        pair_df.groupby("Grupo_vaso")
        .agg(
            pares=("correlacao_geral_todos_mq", "size"),
            correlacao_media=("correlacao_geral_todos_mq", "mean"),
            correlacao_mediana=("correlacao_geral_todos_mq", "median"),
            correlacao_minima=("correlacao_geral_todos_mq", "min"),
            correlacao_maxima=("correlacao_geral_todos_mq", "max"),
            distancia_media=("distancia_media_perfil_mq", "mean"),
            rmse_medio=("erro_medio_alinhado_rmse", "mean"),
        )
        .reset_index()
    )

    high_pairs = int((pair_df["correlacao_geral_todos_mq"] >= 0.8).sum())
    lines = [
        "Analise geral de correlacao entre vasos",
        "",
        "Criterio usado:",
        "- Vaso 1 a 5 = saudavel.",
        "- Vaso 6 a 10 = doente.",
        "- Nao separa por dia.",
        "- Nao separa por sensor.",
        "",
        "Como foi calculado:",
        "1. Todos os sensores MQ foram padronizados para a mesma escala.",
        "2. Cada vaso virou um perfil geral com MQ2, MQ3, MQ7, MQ8, MQ135 e MQ138 juntos.",
        "3. A correlacao compara o perfil completo de um vaso contra outro vaso.",
        "",
        f"Linhas validas usadas: {len(df)}",
        f"Pares de vasos comparados: {len(pair_df)}",
        f"Pares com correlacao geral alta >= 0.80: {high_pairs}/{len(pair_df)}",
        "",
        "Resumo por grupo:",
        group_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "Teste geral de diferenca entre vasos:",
        test_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "Resumo geral por vaso:",
        summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "Interpretacao curta:",
        "Correlacao alta significa que os vasos tiveram perfil MQ parecido.",
        "Correlacao baixa ou negativa significa que os vasos do mesmo grupo nao seguiram o mesmo padrao geral.",
    ]
    (ANALYSIS_DIR / "resumo_correlacao_geral_vasos.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    pair_df = build_general_pair_analysis(df)
    test_df = build_general_difference_tests(df)
    summary_df = build_vase_summary(df)

    pair_df.to_csv(ANALYSIS_DIR / "correlacao_geral_pares_vasos.csv", index=False)
    test_df.to_csv(ANALYSIS_DIR / "teste_diferenca_geral_vasos.csv", index=False)
    summary_df.to_csv(ANALYSIS_DIR / "resumo_geral_por_vaso.csv", index=False)

    save_bar_plot(pair_df)
    save_heatmap(pair_df, "saudavel_vasos_1_a_5", "mapa_correlacao_geral_saudavel_vasos_1_a_5.png")
    save_heatmap(pair_df, "doente_vasos_6_a_10", "mapa_correlacao_geral_doente_vasos_6_a_10.png")
    save_report(pair_df, test_df, summary_df, df)

    print(f"Analise geral salva em: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
