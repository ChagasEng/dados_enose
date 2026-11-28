"""Gráficos de todos os sinais registrados nas cinco coletas críticas do ET com MQ."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "loso_balanceado_52486_solo_binario"
OUT = ROOT / "diagnostico_erros_et_mq_52486" / "graficos_todos_sensores"
COLLECTIONS = [
    ("dia 19 - Soja Heterodera Vaso 8", "dia19_inoculado8"),
    ("dia 11 - Soja Heterodera Vaso 7", "dia11_inoculado7"),
    ("dia 18 - Soja Heterodera Vaso 9", "dia18_inoculado9"),
    ("dia 18 - Soja Heterodera Vaso 6", "dia18_inoculado6"),
    ("dia 19 - Soja Heterodera Vaso 7", "dia19_inoculado7"),
]
SIGNALS = [
    ("MQ2", "MQ-2 (bruto)"),
    ("MQ3", "MQ-3 (bruto)"),
    ("MQ7", "MQ-7 (bruto)*"),
    ("MQ8", "MQ-8 (bruto)"),
    ("MQ135", "MQ-135 (bruto)"),
    ("MQ138", "MQ-138 (bruto)"),
    ("Soil_indice_0_1", "Índice do solo (0–1)"),
    ("Temp_C", "Temperatura (°C)"),
    ("Pres_kPa", "Pressão (kPa)"),
]


def make_figure(block: pd.DataFrame, collection: str) -> plt.Figure:
    block = block.sort_values("Tempo")
    x = block.Tempo.to_numpy()
    wrong = block.erro_mq.to_numpy()
    n, errors = len(block), int(wrong.sum())
    fig, axes = plt.subplots(10, 1, figsize=(13, 18), sharex=True,
                             gridspec_kw={"height_ratios": [1] * 9 + [1.5]},
                             constrained_layout=True)
    for ax, (column, label) in zip(axes, SIGNALS):
        ax.fill_between(x, 0, 1, where=wrong, step="mid",
                        transform=ax.get_xaxis_transform(), color="#e9a0a0", alpha=.34)
        ax.plot(x, block[column], linewidth=.95, color="#1e5d87")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=.17)
        ax.tick_params(axis="y", labelsize=8)
    ax = axes[-1]
    ax.fill_between(x, 0, 1, where=wrong, step="mid",
                    transform=ax.get_xaxis_transform(), color="#e9a0a0", alpha=.34)
    ax.plot(x, block.prob_classe_1_mq, linewidth=1.15, color="#0d4b78",
            label="ET usando cinco MQ")
    ax.plot(x, block.prob_classe_1_mq_ambiente, linewidth=1, color="#d37b25",
            label="ET com MQ + solo, temperatura e pressão")
    ax.axhline(.5, linestyle="--", linewidth=1.1, color="#a32222", label="Limiar 0,5")
    ax.set_ylim(-.03, 1.03)
    ax.set_ylabel("P(controle)", fontsize=9)
    ax.set_xlabel("Tempo na coleta (s)")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=.17)
    axes[0].set_xlim(x.min(), x.max())
    fig.suptitle(
        f"{collection} | {errors} erros e {n-errors} acertos com cinco MQ\n"
        "Vermelho = leitura errada (real 0, prevista 1); "
        "*MQ-7 registrado, mas não usado pelo ET; escalas verticais próprias.",
        fontsize=11,
    )
    return fig


def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = pd.read_csv(SRC / "base_balanceada.csv")
    mq = pd.read_csv(SRC / "predicoes_mq_et.csv")
    full = pd.read_csv(SRC / "predicoes_mq_ambiente_et.csv")
    for key in ("linha_balanceada", "Coleta", "Vaso_id", "Classe", "Tempo"):
        assert base[key].equals(mq[key]) and base[key].equals(full[key])
    base["erro_mq"] = base.Classe.ne(mq.predicao)
    base["prob_classe_1_mq"] = mq.probabilidade_classe_1
    base["prob_classe_1_mq_ambiente"] = full.probabilidade_classe_1
    pdf_path = OUT / "cinco_coletas_todos_sensores.pdf"
    with PdfPages(pdf_path) as pdf:
        for collection, shortname in COLLECTIONS:
            block = base.loc[base.Coleta.eq(collection)]
            assert len(block) > 0 and block.Classe.eq(0).all()
            fig = make_figure(block, collection)
            fig.savefig(OUT / f"{shortname}_todos_sensores.png", dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"{shortname}: {len(block)} leituras; {block.erro_mq.sum()} erros")
    print(pdf_path)


if __name__ == "__main__":
    main()
