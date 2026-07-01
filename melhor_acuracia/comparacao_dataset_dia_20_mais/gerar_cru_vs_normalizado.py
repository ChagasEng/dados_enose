from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "comparacao" / "cru_vs_normalizado"
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


def normalize(series: pd.Series, low: float, high: float) -> pd.Series:
    if high == low:
        return series * 0
    return ((series - low) / (high - low)).clip(0, 1)


def plot_feature(feature: str, antes: pd.DataFrame, dia20: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
    low, high = bounds[feature]

    for ax_index, mode in enumerate(["cru", "normalizado"]):
        ax = axes[ax_index]
        for periodo, df in [("antes", antes), ("dia20", dia20)]:
            for classe in [0, 1]:
                part = df[df["Classe"] == classe].reset_index(drop=True).copy()
                part["idx_original"] = np.arange(len(part))
                curve = resample_curve(part, feature)
                y = curve["value_smooth"]
                if mode == "normalizado":
                    y = normalize(y, low, high)
                ax.plot(
                    curve["x_pct"],
                    y,
                    color=COLORS[periodo],
                    linestyle=CLASS_LINESTYLES[classe],
                    linewidth=2.0,
                    label=(
                        ("Antes do dia 20" if periodo == "antes" else "Dia 20 em diante")
                        + f" - {CLASS_NAMES[classe]}"
                    ),
                )
        ax.grid(alpha=0.22)
        ax.set_xlabel("Posicao dentro da classe (%)")
        ax.set_title(f"{feature} - {'valor cru' if mode == 'cru' else 'normalizado 0 a 1'}")
        ax.set_ylabel("Valor cru" if mode == "cru" else "Valor normalizado")
        if mode == "normalizado":
            ax.set_ylim(-0.05, 1.05)

    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Comparacao cru vs normalizado - {feature}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / f"{feature}_cru_vs_normalizado.png", dpi=180)
    plt.close(fig)


def main() -> None:
    antes = load_dataset(ROOT / "sem pressao" / "dataset_sem_pressao.csv", "antes")
    dia20 = load_dataset(ROOT / "dia_20_mais" / "dia_20_mais" / "dataset_dia_20_mais.csv", "dia20")
    combined = pd.concat([antes[FEATURES], dia20[FEATURES]], ignore_index=True)
    bounds = {
        feature: (
            float(combined[feature].quantile(0.01)),
            float(combined[feature].quantile(0.99)),
        )
        for feature in FEATURES
    }

    for feature in FEATURES:
        plot_feature(feature, antes, dia20, bounds)

    (OUT / "README.txt").write_text(
        "A curva normalizada deve manter praticamente o mesmo formato da curva crua, porque a normalizacao usada e linear. A diferenca principal esta no eixo Y: cru usa a escala original do sensor, normalizado usa 0 a 1.\n",
        encoding="utf-8",
    )
    print("Graficos gerados em:", OUT)
    for path in sorted(OUT.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
