"""Gera um painel com a melhor matriz de confusao para cada quantidade de MQ."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from avaliar_combinacoes_sensores import (
    DATASET,
    TARGET,
    corrigir_mq_pelo_treino,
    criar_modelo,
    features_para,
    localizar_coleta,
)


BASE = Path(__file__).resolve().parents[1]
RESULTADOS = BASE / "ablacao_sensores_sem_C16_treino"


def desenhar_matriz(ax, matriz: np.ndarray, titulo: str) -> None:
    ax.imshow(matriz, cmap="Blues")
    limite = matriz.max() / 2
    for linha in range(2):
        for coluna in range(2):
            ax.text(
                coluna,
                linha,
                f"{matriz[linha, coluna]:,}".replace(",", "."),
                ha="center",
                va="center",
                color="white" if matriz[linha, coluna] > limite else "#172033",
                fontsize=12,
                fontweight="bold",
            )
    ax.set_xticks([0, 1], ["Com\nnematoide", "Sem\nnematoide"])
    ax.set_yticks([0, 1], ["Com\nnematoide", "Sem\nnematoide"])
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(titulo, fontsize=10, fontweight="bold", pad=10)


def main() -> None:
    melhores = pd.read_csv(RESULTADOS / "melhor_combinacao_por_quantidade.csv")
    if sorted(melhores["qtd_sensores"].tolist()) != [1, 2, 3, 4, 5, 6]:
        raise ValueError("O arquivo precisa conter a melhor combinacao para cada quantidade de 1 a 6 MQ.")

    df = pd.read_csv(DATASET)
    coleta_c16 = localizar_coleta("C16")
    mask_treino = df["Conjunto"].eq("Treino") & ~df["Coleta"].eq(coleta_c16)
    mask_teste = df["Conjunto"].eq("Teste")
    df = corrigir_mq_pelo_treino(df, mask_treino)

    fig, eixos = plt.subplots(2, 3, figsize=(16, 10))
    linhas_resumo = []
    for ax, (_, cenario) in zip(eixos.ravel(), melhores.sort_values("qtd_sensores").iterrows()):
        sensores = tuple(cenario["sensores_mantidos"].split(" | "))
        features = features_para(sensores)
        modelo = criar_modelo()
        modelo.fit(df.loc[mask_treino, features], df.loc[mask_treino, TARGET])
        y_real = df.loc[mask_teste, TARGET]
        y_predito = modelo.predict(df.loc[mask_teste, features])
        matriz = confusion_matrix(y_real, y_predito, labels=[0, 1])
        accuracy = accuracy_score(y_real, y_predito)
        balanced = balanced_accuracy_score(y_real, y_predito)
        titulo = (
            f"{len(sensores)} MQ: {' + '.join(sensores)}\n"
            f"Accuracy {accuracy:.2%} | Balanced {balanced:.2%}"
        )
        desenhar_matriz(ax, matriz, titulo)
        linhas_resumo.append(
            {
                "qtd_sensores": len(sensores),
                "sensores": " | ".join(sensores),
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "real_com_previsto_com": int(matriz[0, 0]),
                "real_com_previsto_sem": int(matriz[0, 1]),
                "real_sem_previsto_com": int(matriz[1, 0]),
                "real_sem_previsto_sem": int(matriz[1, 1]),
            }
        )

    fig.suptitle(
        "Matrizes de confusao — melhor combinacao para cada quantidade de sensores\n"
        "Mesmo ExtraTrees e mesmo teste; C16 removida somente do treino; Soil, Temp e Pressao mantidos em todos os cenarios",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RESULTADOS / "painel_matrizes_confusao_melhores_combinacoes.png", dpi=200)
    plt.close(fig)

    pd.DataFrame(linhas_resumo).to_csv(
        RESULTADOS / "resumo_matrizes_confusao_melhores_combinacoes.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
