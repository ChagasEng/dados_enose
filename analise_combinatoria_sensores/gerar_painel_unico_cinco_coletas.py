"""Uma imagem com seis MQ, solo, temperatura, pressao e erros nas cinco coletas."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "loso_balanceado_52486_solo_binario"
OUT = ROOT / "diagnostico_erros_et_mq_52486" / "graficos_todos_sensores"
COLLECTIONS = [
    ("dia 19 - Soja Heterodera Vaso 8", "Dia 19, inoc. 8"),
    ("dia 11 - Soja Heterodera Vaso 7", "Dia 11, inoc. 7"),
    ("dia 18 - Soja Heterodera Vaso 9", "Dia 18, inoc. 9"),
    ("dia 18 - Soja Heterodera Vaso 6", "Dia 18, inoc. 6"),
    ("dia 19 - Soja Heterodera Vaso 7", "Dia 19, inoc. 7"),
]
MQ_COLORS = {
    "MQ2": "#2860aa", "MQ3": "#e08b22", "MQ7": "#7b8188",
    "MQ8": "#2b9a65", "MQ135": "#8352a5", "MQ138": "#ca4655",
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = pd.read_csv(SRC / "base_balanceada.csv")
    mq = pd.read_csv(SRC / "predicoes_mq_et.csv")
    full = pd.read_csv(SRC / "predicoes_mq_ambiente_et.csv")
    for key in ("linha_balanceada", "Coleta", "Vaso_id", "Classe", "Tempo"):
        assert base[key].equals(mq[key]) and base[key].equals(full[key])
    base["erro_mq"] = base.Classe.ne(mq.predicao)
    base["p1_mq"] = mq.probabilidade_classe_1
    base["p1_full"] = full.probabilidade_classe_1

    fig, axes = plt.subplots(5, 5, figsize=(24, 15), sharex="col",
                             gridspec_kw={"height_ratios": [2, 1, 1, 1, 2]})
    for col, (collection, short) in enumerate(COLLECTIONS):
        block = base.loc[base.Coleta.eq(collection)].sort_values("Tempo")
        assert block.Classe.eq(0).all()
        x = block.Tempo.to_numpy()
        wrong = block.erro_mq.to_numpy()
        for ax in axes[:, col]:
            ax.fill_between(x, 0, 1, where=wrong, step="mid",
                            transform=ax.get_xaxis_transform(), color="#e99b9b", alpha=.35)
            ax.grid(axis="y", alpha=.16)
            ax.set_xlim(x.min(), x.max())
            ax.tick_params(labelsize=8)
        axes[0, col].set_title(
            f"{short}\n{int(wrong.sum())} erros / {len(block)} leituras", fontsize=11
        )
        for sensor, color in MQ_COLORS.items():
            values = block[sensor].to_numpy()
            low, high = np.quantile(values, [.01, .99])
            scaled = np.clip((values - low) / (high - low), 0, 1) if high > low else np.zeros(len(values))
            axes[0, col].plot(x, scaled, linewidth=.8, alpha=.82,
                              color=color, linestyle="--" if sensor == "MQ7" else "-")
        axes[0, col].set_ylim(-.05, 1.05)

        axes[1, col].plot(x, block.Soil_indice_0_1, color="#6c4ba0", linewidth=1)
        axes[1, col].set_ylim(-.05, 1.05)
        axes[2, col].plot(x, block.Temp_C, color="#ab641c", linewidth=1)
        axes[2, col].set_ylim(30, 47.5)
        axes[3, col].plot(x, block.Pres_kPa, color="#32627b", linewidth=1)
        axes[3, col].set_ylim(93.49, 93.74)
        axes[4, col].plot(x, block.p1_mq, color="#124d80", linewidth=1.05)
        axes[4, col].plot(x, block.p1_full, color="#db7f28", linewidth=.95)
        axes[4, col].axhline(.5, color="#a52323", linestyle="--", linewidth=1)
        axes[4, col].set_ylim(-.04, 1.04)
        axes[4, col].set_xlabel("Tempo (s)", fontsize=9)

    labels = ["MQ normalizados*", "Índice do solo (0–1)", "Temperatura (°C)",
              "Pressão (kPa)", "P(classe 1)"]
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label, fontsize=10)
    handles = [Line2D([0], [0], color=color,
                      linestyle="--" if sensor == "MQ7" else "-", label=sensor)
               for sensor, color in MQ_COLORS.items()]
    handles += [Line2D([0], [0], color="#124d80", label="ET só MQ"),
                Line2D([0], [0], color="#db7f28", label="ET MQ + ambiente"),
                Line2D([0], [0], color="#a52323", linestyle="--", label="Limiar 0,5")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .955), ncol=9, fontsize=9)
    fig.suptitle("Cinco coletas inoculadas: todos os sensores e erros do ET usando só MQ",
                 y=.995, fontsize=16)
    fig.text(.5, .018,
             "Vermelho = leitura errada pelo ET só MQ (real 0, prevista 1). "
             "*Cada curva MQ foi reescalada entre os percentis 1 e 99 da própria coleta; "
             "MQ-7 foi registrado, mas não entrou no ET.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.055, right=.985, top=.88, bottom=.07, hspace=.18, wspace=.20)
    dest = OUT / "cinco_coletas_todos_sensores_uma_imagem.png"
    fig.savefig(dest, dpi=180)
    plt.close(fig)
    print(dest)


if __name__ == "__main__":
    main()
