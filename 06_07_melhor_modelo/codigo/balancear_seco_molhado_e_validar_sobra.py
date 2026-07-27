"""Balanceia a avaliacao seco/molhado e usa o excedente como validacao.

Parte das predicoes independentes ja produzidas para o conjunto Teste. Assim,
nenhuma observacao e movida para o treino e o split original por Coleta e
preservado.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


BASE = Path(__file__).resolve().parents[1]
ENTRADA = BASE / "analise_seco_molhado" / "predicoes_teste_com_faixa_soil.csv"
SAIDA = BASE / "analise_seco_molhado_balanceado"
RANDOM_STATE = 42
TARGET = "Classe"


def metricas(df: pd.DataFrame, nome: str) -> dict[str, object]:
    y_true = df[TARGET]
    y_pred = df["predito"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    recalls = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "conjunto": nome,
        "linhas": int(len(df)),
        "doentes_classe_0": int((y_true == 0).sum()),
        "saudaveis_classe_1": int((y_true == 1).sum()),
        "acertos": int((y_true == y_pred).sum()),
        "erros": int((y_true != y_pred).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "acerto_doentes_classe_0": float(recalls[0]),
        "acerto_saudaveis_classe_1": float(recalls[1]),
        "matriz_confusao": cm.tolist(),
    }


def salvar_matriz(df: pd.DataFrame, nome: str) -> None:
    cm = confusion_matrix(df[TARGET], df["predito"], labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["real_com_nematoide", "real_sem_nematoide"],
        columns=["previsto_com_nematoide", "previsto_sem_nematoide"],
    ).to_csv(SAIDA / f"matriz_confusao_{nome}.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    display = ConfusionMatrixDisplay(cm, display_labels=["Com nematoide", "Sem nematoide"])
    display.plot(ax=ax, cmap="Purples", values_format="d", colorbar=False)
    ax.set_title(nome.replace("_", " ").title())
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    fig.tight_layout()
    fig.savefig(SAIDA / f"matriz_confusao_{nome}.png", dpi=180)
    plt.close(fig)


def balancear_faixa(df: pd.DataFrame, faixa: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    bloco = df.loc[df["faixa_soil"] == faixa].copy()
    doentes = bloco.loc[bloco[TARGET] == 0].copy()
    saudaveis = bloco.loc[bloco[TARGET] == 1].copy()
    tamanho_balanceado = min(len(doentes), len(saudaveis))
    if tamanho_balanceado == 0:
        raise ValueError(f"A faixa {faixa} nao tem as duas classes para balancear.")

    # Mantem integralmente a classe minoritaria e sorteia a mesma quantidade
    # da majoritaria. O excedente, ainda nunca usado para metricas desta rodada,
    # e reservado para validacao complementar.
    if len(doentes) > len(saudaveis):
        maioria, minoria, maior_classe = doentes, saudaveis, 0
    else:
        maioria, minoria, maior_classe = saudaveis, doentes, 1
    maioria_balanceada = maioria.sample(n=tamanho_balanceado, random_state=RANDOM_STATE)
    balanceado = pd.concat([minoria, maioria_balanceada], ignore_index=False).sort_index()
    validacao_sobra = maioria.drop(index=maioria_balanceada.index).sort_index()

    registro = {
        "faixa_soil": faixa,
        "classe_majoritaria_excedente": "doentes_classe_0" if maior_classe == 0 else "saudaveis_classe_1",
        "linhas_antes": int(len(bloco)),
        "por_classe_balanceado": int(tamanho_balanceado),
        "linhas_balanceadas": int(len(balanceado)),
        "linhas_validacao_sobra": int(len(validacao_sobra)),
    }
    return balanceado, validacao_sobra, registro


def salvar_grafico(resumo: pd.DataFrame) -> None:
    plot_df = resumo.set_index("conjunto").loc[["avaliacao_balanceada", "validacao_sobra"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(
        ["Avaliacao\nbalanceada", "Validacao\ncom a sobra"],
        plot_df["balanced_accuracy"] * 100,
        color=["#2d6cdf", "#6f8f72"],
    )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("Modelo: avaliacao balanceada e validacao com sobra", pad=14)
    ax.grid(axis="y", alpha=0.2)
    for barra, linha in zip(barras, plot_df.itertuples()):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 1.5,
            f"{linha.balanced_accuracy:.2%}\n({linha.acertos:,}/{linha.linhas:,})".replace(",", "."),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(SAIDA / "comparacao_balanceado_validacao.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def salvar_painel_metricas(resumo: pd.DataFrame) -> None:
    """Gera um painel amplo, no estilo dos graficos de coletas do projeto."""

    dados = resumo.set_index("conjunto").loc[["avaliacao_balanceada", "validacao_sobra"]]
    colunas = [
        ("accuracy", "Acuracia", "#2d6cdf"),
        ("f1_macro", "F1 macro", "#7950b3"),
        ("acerto_doentes_classe_0", "Recall\ndoentes", "#c85a5a"),
        ("acerto_saudaveis_classe_1", "Recall\nsaudaveis", "#4f9b70"),
    ]
    fundos = ["#eaf2ff", "#eaf7ee"]
    titulos = [
        "Avaliacao balanceada: 7.778 doentes + 7.778 saudaveis",
        "Validacao com a sobra: 2.881 doentes + 3.678 saudaveis",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, sharey=True)
    for ax, (_, linha), fundo, titulo in zip(axes, dados.iterrows(), fundos, titulos):
        ax.set_facecolor(fundo)
        valores = [linha[coluna] * 100 for coluna, _, _ in colunas]
        barras = ax.bar(
            [rotulo for _, rotulo, _ in colunas],
            valores,
            color=[cor for _, _, cor in colunas],
            width=0.62,
            edgecolor="white",
            linewidth=1.2,
        )
        for barra, valor in zip(barras, valores):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                valor + 1.6,
                f"{valor:.2f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="semibold",
            )
        ax.set_title(titulo, loc="left", fontsize=13, fontweight="semibold", pad=10)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.24)
        ax.set_ylabel("Percentual (%)")

    fig.suptitle("Metricas do melhor modelo: avaliacao balanceada e validacao", fontsize=18, y=0.995)
    fig.legend(
        handles=[Patch(color=cor, label=rotulo.replace("\n", " ")) for _, rotulo, cor in colunas],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.98),
        ncol=4,
        frameon=True,
    )
    axes[-1].set_xlabel("Metrica")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(SAIDA / "grafico_metricas_acuracia_f1_recall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def salvar_painel_metricas_seco_molhado(resumo_faixas: pd.DataFrame) -> None:
    """Compara seco e molhado com ambas as classes balanceadas em cada painel."""

    dados = resumo_faixas.set_index("conjunto").loc[["seco_balanceado", "molhado_balanceado"]]
    colunas = [
        ("accuracy", "Acuracia", "#2d6cdf"),
        ("f1_macro", "F1 macro", "#7950b3"),
        ("acerto_doentes_classe_0", "Recall\ndoentes", "#c85a5a"),
        ("acerto_saudaveis_classe_1", "Recall\nsaudaveis", "#4f9b70"),
    ]
    paineis = [
        ("Solo seco (Soil_indice_0_1 <= 0,4)", "#fff1e6"),
        ("Solo molhado (Soil_indice_0_1 > 0,4)", "#eaf7ee"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, sharey=True)
    for ax, (_, linha), (titulo, fundo) in zip(axes, dados.iterrows(), paineis):
        ax.set_facecolor(fundo)
        valores = [linha[coluna] * 100 for coluna, _, _ in colunas]
        barras = ax.bar(
            [rotulo for _, rotulo, _ in colunas],
            valores,
            color=[cor for _, _, cor in colunas],
            width=0.62,
            edgecolor="white",
            linewidth=1.2,
        )
        for barra, valor in zip(barras, valores):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                valor + 1.6,
                f"{valor:.2f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="semibold",
            )
        ax.set_title(
            f"{titulo}: {int(linha.doentes_classe_0):,} doentes + {int(linha.saudaveis_classe_1):,} saudaveis".replace(",", "."),
            loc="left",
            fontsize=13,
            fontweight="semibold",
            pad=10,
        )
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.24)
        ax.set_ylabel("Percentual (%)")

    fig.suptitle("Metricas do melhor modelo por condicao do solo", fontsize=18, y=0.995)
    fig.legend(
        handles=[Patch(color=cor, label=rotulo.replace("\n", " ")) for _, rotulo, cor in colunas],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.98),
        ncol=4,
        frameon=True,
    )
    axes[-1].set_xlabel("Metrica")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(SAIDA / "grafico_metricas_seco_molhado_balanceado.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def salvar_readme(resumo: pd.DataFrame, faixas: pd.DataFrame) -> None:
    avaliacao = resumo.loc[resumo["conjunto"] == "avaliacao_balanceada"].iloc[0]
    validacao = resumo.loc[resumo["conjunto"] == "validacao_sobra"].iloc[0]
    seco = faixas.loc[faixas["faixa_soil"] == "seco"].iloc[0]
    molhado = faixas.loc[faixas["faixa_soil"] == "molhado"].iloc[0]
    texto = f"""# Avaliacao balanceada por condicao de solo

O modelo nao foi retreinado. Foram usadas somente as predicoes do conjunto Teste ja separado por Coleta, preservando a independencia em relacao ao treino.

## Como foi balanceado

- `seco`: {int(seco.por_classe_balanceado)} doentes + {int(seco.por_classe_balanceado)} saudaveis; {int(seco.linhas_validacao_sobra)} saudaveis restantes foram para validacao.
- `molhado`: {int(molhado.por_classe_balanceado)} doentes + {int(molhado.por_classe_balanceado)} saudaveis; {int(molhado.linhas_validacao_sobra)} doentes restantes foram para validacao.
- Amostragem aleatoria reprodutivel com `random_state={RANDOM_STATE}`. Nenhuma linha aparece nos dois conjuntos.

## Resultado

| Conjunto | Doentes | Saudaveis | Acertos | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Avaliacao balanceada | {int(avaliacao.doentes_classe_0)} | {int(avaliacao.saudaveis_classe_1)} | {int(avaliacao.acertos)} / {int(avaliacao.linhas)} | {avaliacao.accuracy:.2%} | {avaliacao.balanced_accuracy:.2%} |
| Validacao com sobra | {int(validacao.doentes_classe_0)} | {int(validacao.saudaveis_classe_1)} | {int(validacao.acertos)} / {int(validacao.linhas)} | {validacao.accuracy:.2%} | {validacao.balanced_accuracy:.2%} |

A validacao com sobra combina os excedentes das duas faixas; por isso ela volta a conter as duas classes. Ainda assim, os resultados devem ser lidos junto com os resultados por faixa, pois cada sobra isolada contem apenas a classe que era majoritaria naquela faixa.

## Arquivos

- `avaliacao_balanceada.csv`: conjunto equilibrado para comparar doente e saudavel.
- `validacao_sobra.csv`: linhas nao usadas na avaliacao balanceada.
- `resumo_metricas.csv`: metricas dos dois conjuntos.
- `grafico_metricas_acuracia_f1_recall.png`: painel de acuracia, F1 e recall por classe.
- `grafico_metricas_seco_molhado_balanceado.png`: comparacao direta seco x molhado, com as classes balanceadas em cada faixa.
- `resumo_metricas_por_faixa_balanceada.csv`: valores exibidos no grafico seco x molhado.
- `matriz_confusao_*.csv/png`: erros por classe.
"""
    (SAIDA / "README.md").write_text(texto, encoding="utf-8")


def main() -> None:
    if not ENTRADA.exists():
        raise FileNotFoundError(f"Predicoes de entrada nao encontradas: {ENTRADA}")
    df = pd.read_csv(ENTRADA)
    required = [TARGET, "predito", "acertou", "faixa_soil"]
    missing = [coluna for coluna in required if coluna not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    partes_balanceadas: list[pd.DataFrame] = []
    partes_validacao: list[pd.DataFrame] = []
    registros_faixa: list[dict[str, object]] = []
    metricas_faixas_balanceadas: list[dict[str, object]] = []
    for faixa in ("seco", "molhado"):
        balanceado, sobra, registro = balancear_faixa(df, faixa)
        partes_balanceadas.append(balanceado)
        partes_validacao.append(sobra)
        registros_faixa.append(registro)
        metricas_faixas_balanceadas.append(metricas(balanceado, f"{faixa}_balanceado"))

    avaliacao = pd.concat(partes_balanceadas).sort_index()
    validacao = pd.concat(partes_validacao).sort_index()
    if set(avaliacao.index).intersection(validacao.index):
        raise AssertionError("Ha sobreposicao entre avaliacao balanceada e validacao.")
    if len(avaliacao) + len(validacao) != len(df):
        raise AssertionError("As linhas nao foram particionadas integralmente.")

    SAIDA.mkdir(exist_ok=True)
    avaliacao.to_csv(SAIDA / "avaliacao_balanceada.csv", index=False, encoding="utf-8-sig")
    validacao.to_csv(SAIDA / "validacao_sobra.csv", index=False, encoding="utf-8-sig")

    metricas_avaliacao = metricas(avaliacao, "avaliacao_balanceada")
    metricas_validacao = metricas(validacao, "validacao_sobra")
    resumo = pd.DataFrame([metricas_avaliacao, metricas_validacao])
    faixas = pd.DataFrame(registros_faixa)
    resumo_faixas_balanceadas = pd.DataFrame(metricas_faixas_balanceadas)
    resumo.to_csv(SAIDA / "resumo_metricas.csv", index=False, encoding="utf-8-sig")
    faixas.to_csv(SAIDA / "resumo_balanceamento_por_faixa.csv", index=False, encoding="utf-8-sig")
    resumo_faixas_balanceadas.to_csv(
        SAIDA / "resumo_metricas_por_faixa_balanceada.csv", index=False, encoding="utf-8-sig"
    )
    salvar_matriz(avaliacao, "avaliacao_balanceada")
    salvar_matriz(validacao, "validacao_sobra")
    salvar_grafico(resumo)
    salvar_painel_metricas(resumo)
    salvar_painel_metricas_seco_molhado(resumo_faixas_balanceadas)
    salvar_readme(resumo, faixas)
    with (SAIDA / "metodo_e_metricas.json").open("w", encoding="utf-8") as arquivo:
        json.dump(
            {
                "entrada": str(ENTRADA.relative_to(BASE)),
                "metodo": "Balanceamento por faixa Soil no Teste; excedentes reservados para validacao complementar.",
                "random_state": RANDOM_STATE,
                "faixas": registros_faixa,
                "metricas": [metricas_avaliacao, metricas_validacao],
                "metricas_por_faixa_balanceada": metricas_faixas_balanceadas,
            },
            arquivo,
            indent=2,
            ensure_ascii=False,
        )
    print(resumo[["conjunto", "linhas", "doentes_classe_0", "saudaveis_classe_1", "accuracy", "balanced_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
