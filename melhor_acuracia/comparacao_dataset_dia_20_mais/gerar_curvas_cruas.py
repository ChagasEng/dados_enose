from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "comparacao" / "curvas_cruas"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
COLORS = {"antes": "#2f6f73", "dia20": "#c7503d"}
CLASS_NAMES = {0: "Doente", 1: "Saudavel"}
CLASS_LINESTYLES = {0: "-", 1: "--"}


def load_dataset(path: Path, periodo: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Classe"].notna()].copy()
    df["Classe"] = df["Classe"].astype(int)
    df["periodo"] = periodo
    df["idx_original"] = np.arange(len(df))
    return df


def resample_raw_curve(df: pd.DataFrame, feature: str, n_bins: int = 450) -> pd.DataFrame:
    tmp = df[["idx_original", feature]].copy()
    tmp["x_pct"] = tmp["idx_original"] / max(tmp["idx_original"].max(), 1) * 100
    tmp["bin"] = pd.cut(tmp["x_pct"], bins=n_bins, labels=False, include_lowest=True)
    out = tmp.groupby("bin", as_index=False).agg(
        x_pct=("x_pct", "mean"),
        valor_cru=(feature, "mean"),
    )
    out["valor_suavizado"] = out["valor_cru"].rolling(7, center=True, min_periods=1).mean()
    return out


def plot_raw_by_sensor(antes: pd.DataFrame, dia20: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(18, 13), sharex=True)
    axes = axes.ravel()

    for ax, feature in zip(axes, FEATURES):
        for periodo, df in [("antes", antes), ("dia20", dia20)]:
            for classe in [0, 1]:
                class_df = df[df["Classe"] == classe].reset_index(drop=True).copy()
                class_df["idx_original"] = np.arange(len(class_df))
                curve = resample_raw_curve(class_df, feature)
                ax.plot(
                    curve["x_pct"],
                    curve["valor_suavizado"],
                    color=COLORS[periodo],
                    linestyle=CLASS_LINESTYLES[classe],
                    linewidth=2.0,
                    label=(
                        ("Antes do dia 20" if periodo == "antes" else "Dia 20 em diante")
                        + f" - {CLASS_NAMES[classe]}"
                    ),
                )
        ax.set_title(feature)
        ax.set_ylabel("Valor cru")
        ax.grid(alpha=0.22)

    axes[-2].set_xlabel("Posicao dentro da classe (%)")
    axes[-1].set_xlabel("Posicao dentro da classe (%)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle(
        "Curvas MQ com dados crus - antes do dia 20 vs dia 20 em diante",
        fontsize=18,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "01_curvas_mq_dados_crus_por_sensor.png", dpi=180)
    plt.close(fig)


def plot_raw_each_sensor_large(antes: pd.DataFrame, dia20: pd.DataFrame) -> None:
    for feature in FEATURES:
        fig, ax = plt.subplots(figsize=(16, 7))
        for periodo, df in [("antes", antes), ("dia20", dia20)]:
            for classe in [0, 1]:
                class_df = df[df["Classe"] == classe].reset_index(drop=True).copy()
                class_df["idx_original"] = np.arange(len(class_df))
                curve = resample_raw_curve(class_df, feature)
                ax.plot(
                    curve["x_pct"],
                    curve["valor_suavizado"],
                    color=COLORS[periodo],
                    linestyle=CLASS_LINESTYLES[classe],
                    linewidth=2.2,
                    label=(
                        ("Antes do dia 20" if periodo == "antes" else "Dia 20 em diante")
                        + f" - {CLASS_NAMES[classe]}"
                    ),
                )
        ax.set_title(f"{feature} - dados crus")
        ax.set_xlabel("Posicao dentro da classe (%)")
        ax.set_ylabel("Valor cru")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(OUT / f"sensor_{feature}_dados_crus.png", dpi=180)
        plt.close(fig)


def save_raw_statistics(antes: pd.DataFrame, dia20: pd.DataFrame) -> None:
    rows = []
    for periodo, df in [("Antes do dia 20", antes), ("Dia 20 em diante", dia20)]:
        for classe in [0, 1]:
            part = df[df["Classe"] == classe]
            for feature in FEATURES:
                rows.append(
                    {
                        "periodo": periodo,
                        "classe": classe,
                        "sensor": feature,
                        "min": part[feature].min(),
                        "q25": part[feature].quantile(0.25),
                        "media": part[feature].mean(),
                        "mediana": part[feature].median(),
                        "q75": part[feature].quantile(0.75),
                        "max": part[feature].max(),
                    }
                )
    pd.DataFrame(rows).to_csv(OUT / "estatisticas_dados_crus_mq.csv", index=False)


def main() -> None:
    antes = load_dataset(ROOT / "sem pressao" / "dataset_sem_pressao.csv", "antes")
    dia20 = load_dataset(ROOT / "dia_20_mais" / "dia_20_mais" / "dataset_dia_20_mais.csv", "dia20")

    plot_raw_by_sensor(antes, dia20)
    plot_raw_each_sensor_large(antes, dia20)
    save_raw_statistics(antes, dia20)

    (OUT / "README.txt").write_text(
        "\n".join(
            [
                "Graficos com dados crus",
                "",
                "Estes graficos nao usam normalizacao.",
                "Os valores dos sensores MQ aparecem na escala original do CSV.",
                "A unica transformacao visual aplicada foi reamostragem em blocos e media movel leve para reduzir a compressao do eixo X.",
                "",
                "Linhas:",
                "Verde continuo = Antes do dia 20 - Doente",
                "Verde tracejado = Antes do dia 20 - Saudavel",
                "Vermelho continuo = Dia 20 em diante - Doente",
                "Vermelho tracejado = Dia 20 em diante - Saudavel",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Graficos crus gerados em:", OUT)
    for path in sorted(OUT.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
