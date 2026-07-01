from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "comparacao" / "curvas_visiveis"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
COLORS = {"sem_pressao": "#2f6f73", "dia_20_mais": "#c7503d"}
DISPLAY_NAMES = {
    "sem_pressao": "Antes do dia 20",
    "dia_20_mais": "Dia 20 em diante",
}
CLASS_LABELS = {0: "0 - doente", 1: "1 - saudavel"}
CLASS_NAMES = {0: "Doente", 1: "Saudavel"}
CLASS_LINESTYLES = {0: "-", 1: "--"}


def load_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Classe"].notna()].copy()
    df["Classe"] = df["Classe"].astype(int)
    df["dataset"] = dataset_name
    df["idx_original"] = np.arange(len(df))
    return df


def robust_normalize(series: pd.Series, low: float, high: float) -> pd.Series:
    if high == low:
        return series * 0
    normalized = (series - low) / (high - low)
    return normalized.clip(0, 1)


def resample_curve(df: pd.DataFrame, feature: str, n_bins: int = 450) -> pd.DataFrame:
    tmp = df[["idx_original", feature]].copy()
    tmp["x_pct"] = tmp["idx_original"] / max(tmp["idx_original"].max(), 1) * 100
    tmp["bin"] = pd.cut(tmp["x_pct"], bins=n_bins, labels=False, include_lowest=True)
    out = tmp.groupby("bin", as_index=False).agg(
        x_pct=("x_pct", "mean"),
        value=(feature, "mean"),
    )
    out["value_smooth"] = out["value"].rolling(7, center=True, min_periods=1).mean()
    return out


def plot_sensors_by_dataset(sem: pd.DataFrame, dia: pd.DataFrame) -> None:
    combined = pd.concat([sem, dia], ignore_index=True)
    bounds = {
        feature: (
            float(combined[feature].quantile(0.01)),
            float(combined[feature].quantile(0.99)),
        )
        for feature in FEATURES
    }

    fig, axes = plt.subplots(3, 2, figsize=(18, 13), sharex=True)
    axes = axes.ravel()

    for ax, feature in zip(axes, FEATURES):
        low, high = bounds[feature]
        for name, df in [("sem_pressao", sem), ("dia_20_mais", dia)]:
            for classe in [0, 1]:
                class_df = df[df["Classe"] == classe].reset_index(drop=True).copy()
                class_df["idx_original"] = np.arange(len(class_df))
                curve = resample_curve(class_df, feature)
                curve["value_norm"] = robust_normalize(curve["value_smooth"], low, high)
                ax.plot(
                    curve["x_pct"],
                    curve["value_norm"],
                    color=COLORS[name],
                    linestyle=CLASS_LINESTYLES[classe],
                    linewidth=2.0,
                    label=f"{DISPLAY_NAMES[name]} - {CLASS_NAMES[classe]}",
                )
        ax.set_title(feature)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.22)
        ax.set_ylabel("Normalizado")

    axes[-2].set_xlabel("Posicao no dataset (%)")
    axes[-1].set_xlabel("Posicao no dataset (%)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle(
        "Curvas MQ normalizadas e reamostradas por classe",
        fontsize=18,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "01_curvas_mq_normalizadas_reamostradas.png", dpi=180)
    plt.close(fig)


def plot_sensors_by_class(sem: pd.DataFrame, dia: pd.DataFrame) -> None:
    combined = pd.concat([sem, dia], ignore_index=True)
    bounds = {
        feature: (
            float(combined[feature].quantile(0.01)),
            float(combined[feature].quantile(0.99)),
        )
        for feature in FEATURES
    }

    for classe in [0, 1]:
        fig, axes = plt.subplots(3, 2, figsize=(18, 13), sharex=True)
        axes = axes.ravel()
        sem_c = sem[sem["Classe"] == classe].reset_index(drop=True)
        dia_c = dia[dia["Classe"] == classe].reset_index(drop=True)
        sem_c["idx_original"] = np.arange(len(sem_c))
        dia_c["idx_original"] = np.arange(len(dia_c))

        for ax, feature in zip(axes, FEATURES):
            low, high = bounds[feature]
            for name, df in [("sem_pressao", sem_c), ("dia_20_mais", dia_c)]:
                curve = resample_curve(df, feature)
                curve["value_norm"] = robust_normalize(curve["value_smooth"], low, high)
                ax.plot(
                    curve["x_pct"],
                    curve["value_norm"],
                    color=COLORS[name],
                    linewidth=1.9,
                    label=DISPLAY_NAMES[name],
                )
            ax.set_title(feature)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.22)
            ax.set_ylabel("Normalizado")

        axes[-2].set_xlabel("Posicao dentro da classe (%)")
        axes[-1].set_xlabel("Posicao dentro da classe (%)")
        axes[0].legend(frameon=False, loc="upper right")
        fig.suptitle(
            f"Curvas MQ normalizadas e reamostradas - {CLASS_LABELS[classe]}",
            fontsize=18,
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(OUT / f"02_curvas_mq_classe_{classe}_normalizadas_reamostradas.png", dpi=180)
        plt.close(fig)


def plot_zoom_first_points(sem: pd.DataFrame, dia: pd.DataFrame, n_points: int = 3500) -> None:
    combined = pd.concat([sem, dia], ignore_index=True)
    bounds = {
        feature: (
            float(combined[feature].quantile(0.01)),
            float(combined[feature].quantile(0.99)),
        )
        for feature in FEATURES
    }

    fig, axes = plt.subplots(2, 1, figsize=(20, 11), sharex=False)
    for ax, name, df in [
        (axes[0], "sem_pressao", sem.head(n_points).copy()),
        (axes[1], "dia_20_mais", dia.head(n_points).copy()),
    ]:
        offsets = np.arange(len(FEATURES)) * 1.25
        for offset, feature in zip(offsets, FEATURES):
            low, high = bounds[feature]
            y = robust_normalize(df[feature], low, high).rolling(15, center=True, min_periods=1).mean()
            ax.plot(np.arange(len(df)), y + offset, linewidth=1.2, label=feature)
        ax.set_title(f"{name} - zoom nas primeiras {len(df)} linhas")
        ax.set_yticks(offsets + 0.5, FEATURES)
        ax.grid(axis="x", alpha=0.16)
        ax.legend(ncol=6, loc="upper right", fontsize=8)
    axes[1].set_xlabel("Indice da linha no trecho ampliado")
    fig.suptitle("Zoom das curvas MQ para enxergar o formato local", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "03_zoom_primeiras_linhas_curvas_mq.png", dpi=180)
    plt.close(fig)


def main() -> None:
    sem = load_dataset(ROOT / "sem pressao" / "dataset_sem_pressao.csv", "sem_pressao")
    dia = load_dataset(ROOT / "dia_20_mais" / "dia_20_mais" / "dataset_dia_20_mais.csv", "dia_20_mais")

    plot_sensors_by_dataset(sem, dia)
    plot_sensors_by_class(sem, dia)
    plot_zoom_first_points(sem, dia)

    readme = OUT / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Graficos de curvas mais visiveis",
                "",
                "Estes graficos reduzem a dispersao do eixo X reamostrando o dataset em blocos.",
                "Tambem aplicam media movel leve para facilitar a leitura visual das curvas.",
                "",
                "Arquivos:",
                "01_curvas_mq_normalizadas_reamostradas.png - comparacao geral por sensor.",
                "02_curvas_mq_classe_0_normalizadas_reamostradas.png - comparacao apenas classe 0.",
                "02_curvas_mq_classe_1_normalizadas_reamostradas.png - comparacao apenas classe 1.",
                "03_zoom_primeiras_linhas_curvas_mq.png - zoom local das primeiras linhas.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Graficos gerados em:", OUT)
    for path in sorted(OUT.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
