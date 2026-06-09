from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
OUTPUT_DIR = Path(__file__).resolve().parent
DATASET_PATH = (
    ROOT_DIR
    / "dataset_processado_por_dia_vaso_sem_vref0"
    / "dataset_unico_por_dia_vaso_sem_vref0.csv"
)
FEATURE_COLUMNS = [
    "MQ2",
    "MQ3",
    "MQ7",
    "MQ8",
    "MQ135",
    "MQ138",
]
ENVIRONMENT_COLUMNS = ["Soil", "Temp.", "Pres."]
METADATA_COLUMNS = ["Dia", "Vaso", "Classe"]
FEATURE_NOTE = (
    "PCA/DBSCAN: apenas MQ2, MQ3, MQ7, MQ8, MQ135, MQ138. "
    "Soil, Temp. e Pres. ficam so para diagnostico."
)
RANDOM_STATE = 42
EPS = 1.15
MIN_SAMPLES = 80


def load_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(DATASET_PATH)
    df.columns = [column.replace("\ufeff", "") for column in df.columns]

    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes no dataset: {missing_columns}")

    for column in FEATURE_COLUMNS + ENVIRONMENT_COLUMNS + METADATA_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS).copy()
    if "Classe" in df.columns:
        df["Classe"] = pd.to_numeric(df["Classe"], errors="coerce").astype("Int64")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[FEATURE_COLUMNS])

    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled_features)
    pca_3d_model = PCA(n_components=3, random_state=RANDOM_STATE)
    pca_3d = pca_3d_model.fit_transform(scaled_features)
    return df, scaled_features, pca_2d, pca_3d_model.explained_variance_ratio_


def run_dbscan(df: pd.DataFrame, scaled_features: np.ndarray) -> pd.DataFrame:
    model = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, n_jobs=-1)
    df = df.copy()
    df["cluster_teste"] = model.fit_predict(scaled_features)
    return df


def save_knn_distance_plot(scaled_features: np.ndarray) -> None:
    neighbors = NearestNeighbors(n_neighbors=MIN_SAMPLES, n_jobs=-1)
    distances, _ = neighbors.fit(scaled_features).kneighbors(scaled_features)
    kth_distances = np.sort(distances[:, -1])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(kth_distances, color="#2f6fba", linewidth=1)
    ax.axhline(EPS, color="#d88725", linestyle="--", label=f"eps usado = {EPS}")
    ax.set_title("Curva k-distancia para apoiar escolha do eps")
    ax.set_xlabel("Pontos ordenados")
    ax.set_ylabel(f"Distancia ate o {MIN_SAMPLES}o vizinho")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "grafico_teste_kdistancia.png", dpi=180)
    plt.close(fig)


def save_scatter_2d(df: pd.DataFrame, pca_coordinates: np.ndarray) -> None:
    labels = sorted(df["cluster_teste"].unique())
    cmap = plt.get_cmap("tab10")

    if "Classe" in df.columns and df["Classe"].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    else:
        fig, ax = plt.subplots(figsize=(7, 5.8))
        axes = [ax]

    for index, label in enumerate(labels):
        mask = df["cluster_teste"] == label
        color = "#333333" if label == -1 else cmap(index % 10)
        cluster_label = "ruido (-1)" if label == -1 else f"Cluster {label}"
        axes[0].scatter(
            pca_coordinates[mask, 0],
            pca_coordinates[mask, 1],
            s=8,
            alpha=0.45,
            color=color,
            label=cluster_label,
        )

    axes[0].set_title(f"DBSCAN teste eps={EPS}, min_samples={MIN_SAMPLES}")

    if len(axes) > 1:
        class_colors = {0: "#d88725", 1: "#2f6fba"}
        class_labels = {0: "Classe 0 - doente", 1: "Classe 1 - saudavel"}
        for class_value in sorted(df["Classe"].dropna().unique()):
            mask = df["Classe"] == class_value
            label = class_labels.get(int(class_value), f"Classe {int(class_value)}")
            axes[1].scatter(
                pca_coordinates[mask, 0],
                pca_coordinates[mask, 1],
                s=8,
                alpha=0.45,
                color=class_colors.get(int(class_value), "#555555"),
                label=label,
            )
        axes[1].set_title("Classe real apenas para comparacao")

    for ax in axes:
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(alpha=0.18)
        ax.legend(markerscale=2)

    fig.suptitle("Agrupamento nao supervisionado - teste com sensores MQ", fontsize=14)
    fig.text(0.5, 0.015, FEATURE_NOTE, ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(OUTPUT_DIR / "grafico_teste_pca.png", dpi=180)
    plt.close(fig)


def save_scatter_3d(df: pd.DataFrame, pca_3d: np.ndarray, explained: np.ndarray) -> None:
    labels = sorted(df["cluster_teste"].unique())
    cmap = plt.get_cmap("tab10")

    has_class = "Classe" in df.columns and df["Classe"].notna().any()
    fig = plt.figure(figsize=(14, 7) if has_class else (8, 7))
    ax = fig.add_subplot(1, 2, 1, projection="3d") if has_class else fig.add_subplot(111, projection="3d")
    for index, label in enumerate(labels):
        mask = df["cluster_teste"] == label
        color = "#333333" if label == -1 else cmap(index % 10)
        cluster_label = "ruido (-1)" if label == -1 else f"Cluster {label}"
        ax.scatter(
            pca_3d[mask, 0],
            pca_3d[mask, 1],
            pca_3d[mask, 2],
            s=3,
            alpha=0.3,
            color=color,
            label=cluster_label,
        )

    axes = [ax]
    if has_class:
        class_ax = fig.add_subplot(1, 2, 2, projection="3d")
        class_colors = {0: "#d88725", 1: "#2f6fba"}
        class_labels = {0: "Classe 0 - doente", 1: "Classe 1 - saudavel"}
        for class_value in sorted(df["Classe"].dropna().unique()):
            mask = df["Classe"] == class_value
            label = class_labels.get(int(class_value), f"Classe {int(class_value)}")
            class_ax.scatter(
                pca_3d[mask, 0],
                pca_3d[mask, 1],
                pca_3d[mask, 2],
                s=3,
                alpha=0.3,
                color=class_colors.get(int(class_value), "#555555"),
                label=label,
            )
        class_ax.set_title("Classe real em PCA 3D, apenas comparacao")
        axes.append(class_ax)

    for axis in axes:
        axis.set_xlabel(f"PC1 ({explained[0] * 100:.2f}%)", labelpad=8)
        axis.set_ylabel(f"PC2 ({explained[1] * 100:.2f}%)", labelpad=8)
        axis.set_zlabel(f"PC3 ({explained[2] * 100:.2f}%)", labelpad=8)
        axis.view_init(elev=22, azim=-58)
        axis.legend(markerscale=3, loc="best")

    ax.set_title("DBSCAN teste em PCA 3D")
    fig.suptitle("Agrupamento nao supervisionado - teste com sensores MQ em 3D", fontsize=14)
    fig.text(0.5, 0.015, FEATURE_NOTE, ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(OUTPUT_DIR / "grafico_teste_pca_3d.png", dpi=180)
    plt.close(fig)


def save_environment_plot(df: pd.DataFrame, env_columns: list[str]) -> None:
    labels = sorted(df["cluster_teste"].unique())
    x = np.arange(len(labels))
    x_labels = ["ruido (-1)" if label == -1 else f"C{label}" for label in labels]

    fig, axes = plt.subplots(1, len(env_columns), figsize=(14, 4.8))
    if len(env_columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, env_columns):
        summary = df.groupby("cluster_teste")[column].agg(["mean", "std"]).reindex(labels)
        ax.bar(
            x,
            summary["mean"],
            yerr=summary["std"].fillna(0),
            color="#6f8fb8",
            alpha=0.85,
            capsize=3,
        )
        ax.set_title(column)
        ax.set_xlabel("Cluster")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Media +/- desvio padrao")
    fig.suptitle("Diagnostico ambiental por cluster", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "Soil, Temp. e Pres. NAO foram usadas no PCA/DBSCAN; grafico apenas para verificar influencia ambiental.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.9])
    fig.savefig(OUTPUT_DIR / "grafico_teste_ambiente_por_cluster.png", dpi=180)
    plt.close(fig)


def save_outputs(
    df: pd.DataFrame,
    scaled_features: np.ndarray,
    pca_2d: np.ndarray,
    explained_3d: np.ndarray,
) -> None:
    df.to_csv(OUTPUT_DIR / "dataset_com_cluster_teste.csv", index=False)

    cluster_sizes = df.groupby("cluster_teste").size().rename("n_cluster").reset_index()

    env_columns = [column for column in ENVIRONMENT_COLUMNS if column in df.columns]
    if env_columns:
        env_summary = (
            df.groupby("cluster_teste")[env_columns]
            .agg(["mean", "std", "min", "max"])
            .round(4)
        )
        env_summary.columns = [
            f"{column}_{metric}" for column, metric in env_summary.columns.to_flat_index()
        ]
        env_summary.reset_index().to_csv(
            OUTPUT_DIR / "resumo_ambiente_por_cluster_teste.csv", index=False
        )
        save_environment_plot(df, env_columns)
    if "Coleta" in df.columns:
        coleta_columns = [
            column
            for column in ["cluster_teste", "Coleta", "Dia", "Vaso", "Classe"]
            if column in df.columns
        ]
        coleta_summary = (
            df.groupby(coleta_columns, dropna=False)
            .size()
            .rename("n_coleta_no_cluster")
            .reset_index()
            .merge(cluster_sizes, on="cluster_teste", how="left")
        )
        coleta_summary["pct_no_cluster"] = (
            coleta_summary["n_coleta_no_cluster"] / coleta_summary["n_cluster"] * 100
        ).round(4)
        coleta_summary = coleta_summary.sort_values(
            ["cluster_teste", "n_coleta_no_cluster"], ascending=[True, False]
        )
        coleta_summary.to_csv(OUTPUT_DIR / "resumo_clusters_por_coleta_teste.csv", index=False)
    else:
        coleta_summary = pd.DataFrame()

    if "Classe" in df.columns and df["Classe"].notna().any():
        comparison = pd.crosstab(
            df["cluster_teste"],
            df["Classe"],
            rownames=["cluster_teste"],
            colnames=["classe_real"],
        )
        comparison.to_csv(OUTPUT_DIR / "comparacao_cluster_classe_teste.csv")
        adjusted_rand = adjusted_rand_score(df["Classe"].astype(int), df["cluster_teste"])
    else:
        comparison = pd.DataFrame()
        adjusted_rand = np.nan

    non_noise_mask = df["cluster_teste"] != -1
    cluster_count = len(set(df["cluster_teste"]) - {-1})
    if cluster_count >= 2 and non_noise_mask.sum() > 1:
        silhouette = silhouette_score(
            scaled_features[non_noise_mask], df.loc[non_noise_mask, "cluster_teste"]
        )
    else:
        silhouette = np.nan

    report = [
        "Teste DBSCAN nao supervisionado",
        "",
        "Dataset: dataset_processado_por_dia_vaso_sem_vref0/dataset_unico_por_dia_vaso_sem_vref0.csv",
        "As colunas Coleta, Dia, Vaso, Soil, Temp., Pres. e Classe NAO foram usadas como features.",
        "Soil, Temp. e Pres. foram mantidas apenas para diagnostico ambiental depois dos clusters.",
        "Features usadas: " + ", ".join(FEATURE_COLUMNS),
        "Dados padronizados com StandardScaler antes do DBSCAN.",
        f"eps: {EPS}",
        f"min_samples: {MIN_SAMPLES}",
        "",
        f"Linhas usadas: {len(df)}",
        f"Quantidade de clusters sem contar ruido: {cluster_count}",
        f"Pontos marcados como ruido (-1): {int((df['cluster_teste'] == -1).sum())}",
        (
            f"Silhouette sem ruido: {silhouette:.6f}"
            if not np.isnan(silhouette)
            else "Silhouette sem ruido: nao aplicavel"
        ),
        (
            f"Adjusted Rand Index contra a classe real: {adjusted_rand:.6f}"
            if not np.isnan(adjusted_rand)
            else "Adjusted Rand Index contra a classe real: nao calculado"
        ),
        "",
        f"PC1: {explained_3d[0] * 100:.4f}%",
        f"PC2: {explained_3d[1] * 100:.4f}%",
        f"PC3: {explained_3d[2] * 100:.4f}%",
        f"PC1 + PC2: {explained_3d[:2].sum() * 100:.4f}%",
        f"PC1 + PC2 + PC3: {explained_3d[:3].sum() * 100:.4f}%",
        "",
    ]

    if not comparison.empty:
        report.extend(
            [
                "Comparacao cluster x classe real, apenas para interpretacao:",
                comparison.to_string(),
                "",
            ]
        )

    if not coleta_summary.empty:
        report.append("Principais coletas por cluster:")
        for cluster in sorted(df["cluster_teste"].unique()):
            subset = coleta_summary[coleta_summary["cluster_teste"] == cluster].head(5)
            cluster_label = "ruido (-1)" if cluster == -1 else f"Cluster {cluster}"
            report.append(cluster_label)
            for row in subset.itertuples(index=False):
                details = []
                if hasattr(row, "Dia") and not pd.isna(row.Dia):
                    details.append(f"dia {int(row.Dia)}")
                if hasattr(row, "Vaso") and not pd.isna(row.Vaso):
                    details.append(f"vaso {int(row.Vaso)}")
                if hasattr(row, "Classe") and not pd.isna(row.Classe):
                    details.append(f"classe {int(row.Classe)}")
                suffix = f" | {', '.join(details)}" if details else ""
                report.append(
                    f"  {row.n_coleta_no_cluster} ({row.pct_no_cluster:.2f}%) | {row.Coleta}{suffix}"
                )
        report.append("")

    report.extend(
        [
            "Observacao:",
            "O DBSCAN nao usa a classe real para treinar. Neste teste, Classe, Dia, Vaso, Soil, Temp. e Pres. foram mantidos apenas para comparacao posterior.",
        ]
    )
    (OUTPUT_DIR / "resumo_teste.txt").write_text("\n".join(report), encoding="utf-8")

    save_scatter_2d(df, pca_2d)
    save_scatter_3d(df, PCA(n_components=3, random_state=RANDOM_STATE).fit_transform(scaled_features), explained_3d)
    save_knn_distance_plot(scaled_features)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, scaled_features, pca_2d, explained_3d = load_dataset()
    clustered_df = run_dbscan(df, scaled_features)
    save_outputs(clustered_df, scaled_features, pca_2d, explained_3d)
    print(f"Teste salvo em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
