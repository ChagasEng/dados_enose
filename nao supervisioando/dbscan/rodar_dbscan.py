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
DATASET_PATH = ROOT_DIR / "sem pressao" / "dataset_sem_pressao.csv"
MQ_COLUMNS = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
RANDOM_STATE = 42
EPS = 1.15
MIN_SAMPLES = 80


def load_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(DATASET_PATH)
    df.columns = [column.replace("\ufeff", "") for column in df.columns]

    for column in MQ_COLUMNS + ["Classe"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=MQ_COLUMNS + ["Classe"]).copy()
    df["Classe"] = df["Classe"].astype(int)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[MQ_COLUMNS])
    pca_coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(
        scaled_features
    )
    return df, scaled_features, pca_coordinates


def run_dbscan(df: pd.DataFrame, scaled_features: np.ndarray) -> pd.DataFrame:
    model = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, n_jobs=-1)
    df = df.copy()
    df["cluster_dbscan"] = model.fit_predict(scaled_features)
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
    fig.savefig(OUTPUT_DIR / "grafico_dbscan_kdistancia.png", dpi=180)
    plt.close(fig)


def save_scatter(df: pd.DataFrame, pca_coordinates: np.ndarray) -> None:
    labels = sorted(df["cluster_dbscan"].unique())
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    for index, label in enumerate(labels):
        mask = df["cluster_dbscan"] == label
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

    class_colors = {0: "#d88725", 1: "#2f6fba"}
    for class_value, label in [(0, "Classe 0 - doente"), (1, "Classe 1 - saudavel")]:
        mask = df["Classe"] == class_value
        axes[1].scatter(
            pca_coordinates[mask, 0],
            pca_coordinates[mask, 1],
            s=8,
            alpha=0.45,
            color=class_colors[class_value],
            label=label,
        )

    axes[0].set_title(f"DBSCAN eps={EPS}, min_samples={MIN_SAMPLES}")
    axes[1].set_title("Classe real para comparacao")
    for ax in axes:
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(alpha=0.18)
        ax.legend(markerscale=2)

    fig.suptitle("Agrupamento nao supervisionado com DBSCAN", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_DIR / "grafico_dbscan_pca.png", dpi=180)
    plt.close(fig)


def save_outputs(df: pd.DataFrame, scaled_features: np.ndarray, pca_coordinates: np.ndarray) -> None:
    df.to_csv(OUTPUT_DIR / "dataset_com_cluster_dbscan.csv", index=False)

    comparison = pd.crosstab(
        df["cluster_dbscan"],
        df["Classe"],
        rownames=["cluster_dbscan"],
        colnames=["classe_real"],
    )
    comparison.to_csv(OUTPUT_DIR / "comparacao_cluster_classe_dbscan.csv")

    non_noise_mask = df["cluster_dbscan"] != -1
    cluster_count = len(set(df["cluster_dbscan"]) - {-1})
    if cluster_count >= 2 and non_noise_mask.sum() > 1:
        silhouette = silhouette_score(
            scaled_features[non_noise_mask], df.loc[non_noise_mask, "cluster_dbscan"]
        )
    else:
        silhouette = np.nan

    adjusted_rand = adjusted_rand_score(df["Classe"], df["cluster_dbscan"])
    report = [
        "DBSCAN nao supervisionado",
        "",
        "Features usadas: " + ", ".join(MQ_COLUMNS),
        "Dados padronizados com StandardScaler antes do DBSCAN.",
        f"eps: {EPS}",
        f"min_samples: {MIN_SAMPLES}",
        "",
        f"Linhas usadas: {len(df)}",
        f"Quantidade de clusters sem contar ruido: {cluster_count}",
        f"Pontos marcados como ruido (-1): {int((df['cluster_dbscan'] == -1).sum())}",
        f"Silhouette sem ruido: {silhouette:.6f}" if not np.isnan(silhouette) else "Silhouette sem ruido: nao aplicavel",
        f"Adjusted Rand Index contra a classe real: {adjusted_rand:.6f}",
        "",
        "Comparacao cluster x classe real:",
        comparison.to_string(),
        "",
        "Observacao:",
        "DBSCAN nao usa a classe real para treinar. A classe real aparece apenas para interpretacao depois.",
    ]
    (OUTPUT_DIR / "resumo_dbscan.txt").write_text("\n".join(report), encoding="utf-8")
    save_scatter(df, pca_coordinates)
    save_knn_distance_plot(scaled_features)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, scaled_features, pca_coordinates = load_dataset()
    clustered_df = run_dbscan(df, scaled_features)
    save_outputs(clustered_df, scaled_features, pca_coordinates)
    print(f"DBSCAN salvo em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
