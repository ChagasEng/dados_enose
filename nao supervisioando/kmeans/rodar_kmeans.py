from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
OUTPUT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "sem pressao" / "dataset_sem_pressao.csv"
MQ_COLUMNS = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
RANDOM_STATE = 42
SILHOUETTE_SAMPLE_SIZE = 10000


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


def run_kmeans(df: pd.DataFrame, scaled_features: np.ndarray) -> pd.DataFrame:
    model = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=50)
    df = df.copy()
    df["cluster_kmeans"] = model.fit_predict(scaled_features)
    return df


def save_scatter(df: pd.DataFrame, pca_coordinates: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    colors = {0: "#d88725", 1: "#2f6fba"}

    for cluster in sorted(df["cluster_kmeans"].unique()):
        mask = df["cluster_kmeans"] == cluster
        axes[0].scatter(
            pca_coordinates[mask, 0],
            pca_coordinates[mask, 1],
            s=8,
            alpha=0.45,
            color=colors.get(cluster, "#555555"),
            label=f"Cluster {cluster}",
        )

    for class_value, label in [(0, "Classe 0 - doente"), (1, "Classe 1 - saudavel")]:
        mask = df["Classe"] == class_value
        axes[1].scatter(
            pca_coordinates[mask, 0],
            pca_coordinates[mask, 1],
            s=8,
            alpha=0.45,
            color=colors.get(class_value, "#555555"),
            label=label,
        )

    axes[0].set_title("KMeans com k=2")
    axes[1].set_title("Classe real para comparacao")
    for ax in axes:
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(alpha=0.18)
        ax.legend(markerscale=2)

    fig.suptitle("Agrupamento nao supervisionado com KMeans", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_DIR / "grafico_kmeans_pca.png", dpi=180)
    plt.close(fig)


def save_elbow_plot(scaled_features: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30)
        labels = model.fit_predict(scaled_features)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette_amostra": float(
                    silhouette_score(
                        scaled_features,
                        labels,
                        sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(labels)),
                        random_state=RANDOM_STATE,
                    )
                ),
            }
        )

    metrics_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(metrics_df["k"], metrics_df["inertia"], marker="o", color="#2f6fba")
    axes[0].set_title("Metodo do cotovelo")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(alpha=0.2)

    axes[1].plot(metrics_df["k"], metrics_df["silhouette_amostra"], marker="o", color="#d88725")
    axes[1].set_title("Silhouette por k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "grafico_kmeans_cotovelo_silhouette.png", dpi=180)
    plt.close(fig)
    return metrics_df


def save_outputs(df: pd.DataFrame, scaled_features: np.ndarray, pca_coordinates: np.ndarray) -> None:
    df.to_csv(OUTPUT_DIR / "dataset_com_cluster_kmeans.csv", index=False)

    comparison = pd.crosstab(
        df["cluster_kmeans"],
        df["Classe"],
        rownames=["cluster_kmeans"],
        colnames=["classe_real"],
    )
    comparison.to_csv(OUTPUT_DIR / "comparacao_cluster_classe_kmeans.csv")

    metrics_df = save_elbow_plot(scaled_features)
    metrics_df.to_csv(OUTPUT_DIR / "metricas_kmeans_por_k.csv", index=False)

    silhouette = silhouette_score(
        scaled_features,
        df["cluster_kmeans"],
        sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE,
    )
    adjusted_rand = adjusted_rand_score(df["Classe"], df["cluster_kmeans"])
    report = [
        "KMeans nao supervisionado",
        "",
        "Features usadas: " + ", ".join(MQ_COLUMNS),
        "Dados padronizados com StandardScaler antes do KMeans.",
        "k usado no grafico principal: 2 clusters.",
        "",
        f"Linhas usadas: {len(df)}",
        f"Silhouette k=2 em amostra de ate {SILHOUETTE_SAMPLE_SIZE}: {silhouette:.6f}",
        f"Adjusted Rand Index contra a classe real: {adjusted_rand:.6f}",
        "",
        "Comparacao cluster x classe real:",
        comparison.to_string(),
        "",
        "Observacao:",
        "KMeans nao usa a classe real para treinar. A classe real aparece apenas para interpretacao depois.",
    ]
    (OUTPUT_DIR / "resumo_kmeans.txt").write_text("\n".join(report), encoding="utf-8")
    save_scatter(df, pca_coordinates)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, scaled_features, pca_coordinates = load_dataset()
    clustered_df = run_kmeans(df, scaled_features)
    save_outputs(clustered_df, scaled_features, pca_coordinates)
    print(f"KMeans salvo em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
