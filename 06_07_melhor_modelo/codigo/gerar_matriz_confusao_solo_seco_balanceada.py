"""Gera a matriz de confusao balanceada para a condicao de solo seco."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


BASE = Path(__file__).resolve().parents[1]
ENTRADA = BASE / "analise_seco_molhado_balanceado" / "avaliacao_balanceada.csv"
SAIDA = BASE / "analise_seco_molhado_balanceado"
NOME_ARQUIVO = "matriz_confusao_solo_seco_balanceada"
MATRIZ_ESPERADA = [[5864, 643], [0, 6507]]


def main() -> None:
    dados = pd.read_csv(ENTRADA)
    seco = dados.loc[dados["faixa_soil"] == "seco"].copy()
    matriz = confusion_matrix(seco["Classe"], seco["predito"], labels=[0, 1])

    if matriz.tolist() != MATRIZ_ESPERADA:
        raise ValueError(
            f"Matriz obtida {matriz.tolist()} difere da esperada {MATRIZ_ESPERADA}."
        )

    pd.DataFrame(
        matriz,
        index=["real_com_nematoide", "real_sem_nematoide"],
        columns=["previsto_com_nematoide", "previsto_sem_nematoide"],
    ).to_csv(SAIDA / f"{NOME_ARQUIVO}.csv", encoding="utf-8-sig")

    figura, eixo = plt.subplots(figsize=(6.5, 5.5))
    exibicao = ConfusionMatrixDisplay(
        matriz,
        display_labels=["Com nematoide", "Sem nematoide"],
    )
    exibicao.plot(
        ax=eixo,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    eixo.set_title("Melhor modelo - solo seco\n(classes balanceadas)")
    eixo.set_xlabel("Classe prevista")
    eixo.set_ylabel("Classe real")
    figura.tight_layout()
    figura.savefig(SAIDA / f"{NOME_ARQUIVO}.png", dpi=180)
    plt.close(figura)

    print(SAIDA / f"{NOME_ARQUIVO}.png")


if __name__ == "__main__":
    main()
