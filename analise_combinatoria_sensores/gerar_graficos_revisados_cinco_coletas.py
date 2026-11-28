"""Gráficos com escalas fixas e média móvel para as cinco coletas críticas.

As escalas de cada variável são iguais em todas as coletas. A média móvel
centrada de 10 segundos é apenas visual; os erros e probabilidades originais
do LOSO não são recalculados. Lacunas de aquisição > 2 s não são conectadas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "loso_balanceado_52486_solo_binario"
OUT = ROOT / "diagnostico_erros_et_mq_52486" / "graficos_revisados_escalas_fixas"
COLLECTIONS = [
    ("dia 19 - Soja Heterodera Vaso 8", "D19 inoculado 8", "dia19_inoculado8"),
    ("dia 11 - Soja Heterodera Vaso 7", "D11 inoculado 7", "dia11_inoculado7"),
    ("dia 18 - Soja Heterodera Vaso 9", "D18 inoculado 9", "dia18_inoculado9"),
    ("dia 18 - Soja Heterodera Vaso 6", "D18 inoculado 6", "dia18_inoculado6"),
    ("dia 19 - Soja Heterodera Vaso 7", "D19 inoculado 7", "dia19_inoculado7"),
]
SERIES = [
    ("MQ2", "MQ-2 (bruto)", (6_000, 16_500), "#2860aa"),
    ("MQ3", "MQ-3 (bruto)", (11_000, 19_500), "#e08b22"),
    ("MQ7", "MQ-7 (bruto)*", (0, 7_500), "#7b8188"),
    ("MQ8", "MQ-8 (bruto)", (2_500, 9_000), "#2b9a65"),
    ("MQ135", "MQ-135 (bruto)", (10_000, 22_000), "#8352a5"),
    ("MQ138", "MQ-138 (bruto)", (4_500, 14_000), "#ca4655"),
    ("Soil_indice_0_1", "Índice solo (0–1)", (0, 1), "#6c4ba0"),
    ("Temp_C", "Temperatura (°C)", (30, 48), "#a36b22"),
    ("Pres_kPa", "Pressão (kPa)", (93.49, 93.73), "#32627b"),
]
X_LIMITS = (0, 680)
GAP_SECONDS = 2.0
WINDOW_SECONDS = 10.0


def segments(x: np.ndarray) -> list[slice]:
    cuts = np.r_[0, np.flatnonzero(np.diff(x) > GAP_SECONDS) + 1, len(x)]
    return [slice(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]


def moving_average_time(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Média centrada em 10 s, reiniciada depois de lacunas de aquisição."""
    result = np.empty(len(x), dtype=float)
    for section in segments(x):
        xs = x[section]
        ys = y[section]
        left = np.searchsorted(xs, xs - WINDOW_SECONDS / 2, side="left")
        right = np.searchsorted(xs, xs + WINDOW_SECONDS / 2, side="right")
        cumulative = np.r_[0, np.cumsum(ys, dtype=float)]
        result[section] = (cumulative[right] - cumulative[left]) / (right - left)
    return result


def shade_errors(ax: plt.Axes, x: np.ndarray, wrong: np.ndarray) -> None:
    for section in segments(x):
        ax.fill_between(x[section], 0, 1, where=wrong[section], step="mid",
                        transform=ax.get_xaxis_transform(), color="#eaa1a1", alpha=.25)


def plot_collection(axes: np.ndarray, block: pd.DataFrame, short: str) -> None:
    block = block.sort_values("Tempo")
    x = block.Tempo.to_numpy()
    wrong = block.erro_mq.to_numpy()
    for ax in axes:
        shade_errors(ax, x, wrong)
        ax.set_xlim(*X_LIMITS)
        ax.grid(axis="y", alpha=.16)
        ax.tick_params(labelsize=7)
    for ax, (column, _, limits, color) in zip(axes[:9], SERIES):
        y = block[column].to_numpy(dtype=float)
        smooth = moving_average_time(x, y)
        for section in segments(x):
            ax.plot(x[section], y[section], color="#777777", alpha=.24, linewidth=.55)
            ax.plot(x[section], smooth[section], color=color, linewidth=1.25)
        ax.set_ylim(*limits)
    ax = axes[9]
    for column, color in (("p1_mq", "#124d80"), ("p1_full", "#db7f28")):
        y = block[column].to_numpy(dtype=float)
        smooth = moving_average_time(x, y)
        for section in segments(x):
            ax.plot(x[section], y[section], color=color, alpha=.16, linewidth=.55)
            ax.plot(x[section], smooth[section], color=color, linewidth=1.45)
    ax.axhline(.5, color="#a52323", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    axes[0].set_title(f"{short}\n{int(wrong.sum())}/{len(block)} erros", fontsize=10)
    axes[-1].set_xlabel("Tempo registrado (s)", fontsize=9)


def add_legend(fig: plt.Figure, y: float) -> None:
    handles = [
        Line2D([0], [0], color="#777777", alpha=.5, linewidth=1, label="Sinal bruto"),
        Line2D([0], [0], color="#124d80", linewidth=1.4,
               label="Linha colorida: média móvel de 10 s"),
        Patch(facecolor="#eaa1a1", alpha=.4, label="Erro do ET só MQ (sem suavização)"),
        Line2D([0], [0], color="#db7f28", linewidth=1.4,
               label="Probabilidade do ET com MQ + ambiente"),
        Line2D([0], [0], color="#a52323", linestyle="--", label="Limiar 0,5"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, y), ncol=5, fontsize=9)


def save_overview(blocks: list[tuple[pd.DataFrame, str, str]]) -> None:
    fig, axes = plt.subplots(10, 5, figsize=(28, 23), sharex=True)
    for j, (block, short, _) in enumerate(blocks):
        plot_collection(axes[:, j], block, short)
    for i, (_, label, _, _) in enumerate(SERIES):
        axes[i, 0].set_ylabel(label, fontsize=9)
    axes[9, 0].set_ylabel("P(controle)", fontsize=9)
    fig.suptitle("Cinco coletas: sinais brutos e média móvel com escalas fixas por variável",
                 y=.995, fontsize=15)
    add_legend(fig, .975)
    fig.text(.5, .015,
             "Mesmo eixo X e, para cada variável, mesmo eixo Y nas cinco coletas. "
             "*MQ-7 registrado, mas não usado pelo ET. "
             "Lacunas > 2 s não são conectadas. A média móvel não altera os erros.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.07, right=.99, top=.93, bottom=.05, hspace=.18, wspace=.30)
    fig.savefig(OUT / "cinco_coletas_escalas_fixas_media_movel.png", dpi=170)
    fig.savefig(OUT / "cinco_coletas_escalas_fixas_media_movel.pdf")
    plt.close(fig)


def save_individual(block: pd.DataFrame, short: str, filename: str) -> None:
    fig, axes = plt.subplots(10, 1, figsize=(14, 18), sharex=True)
    plot_collection(axes, block, short)
    for i, (_, label, _, _) in enumerate(SERIES):
        axes[i].set_ylabel(label, fontsize=9)
    axes[9].set_ylabel("P(controle)", fontsize=9)
    fig.suptitle(f"{short}: escalas fixas entre as cinco coletas; média móvel de 10 s",
                 y=.995, fontsize=14)
    add_legend(fig, .975)
    fig.text(.5, .018,
             "*MQ-7 registrado, mas não usado pelo ET. Lacunas > 2 s não são conectadas. "
             "Vermelho indica erro calculado sem suavização.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=.12, right=.98, top=.935, bottom=.055, hspace=.15)
    fig.savefig(OUT / f"{filename}_escalas_fixas.png", dpi=160)
    plt.close(fig)


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
    blocks = []
    for collection, short, filename in COLLECTIONS:
        block = base.loc[base.Coleta.eq(collection)].sort_values("Tempo")
        assert len(block) > 0 and block.Classe.eq(0).all()
        for column, _, limits, _ in SERIES:
            values = block[column]
            assert values.ge(limits[0]).all() and values.le(limits[1]).all(), (collection, column)
        blocks.append((block, short, filename))
    save_overview(blocks)
    for block, short, filename in blocks:
        save_individual(block, short, filename)
    print(OUT)


if __name__ == "__main__":
    main()
