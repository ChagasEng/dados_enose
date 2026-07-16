"""Cria o PDF do estudo de reducao de sensores MQ (1 a 6 sensores)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


BASE = Path(__file__).resolve().parents[1]
RESULTADOS = BASE / "ablacao_sensores_sem_C16_treino"
PDF = RESULTADOS / "relatorio_reducao_sensores_mq.pdf"


def pagina_texto(pdf: PdfPages, titulo: str, linhas: list[str], tamanho: float = 14) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 horizontal
    fig.text(0.06, 0.91, titulo, fontsize=22, fontweight="bold", color="#12355b")
    fig.text(0.06, 0.84, "\n\n".join(linhas), fontsize=tamanho, va="top", wrap=True)
    fig.text(0.06, 0.05, "Estudo ExtraTrees - 06_07_melhor_modelo", fontsize=9, color="#555555")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def matriz(ax, valores: np.ndarray, titulo: str) -> None:
    ax.imshow(valores, cmap="Blues")
    corte = valores.max() / 2
    for linha in range(2):
        for coluna in range(2):
            ax.text(
                coluna,
                linha,
                f"{valores[linha, coluna]:,}".replace(",", "."),
                ha="center",
                va="center",
                fontsize=23,
                fontweight="bold",
                color="white" if valores[linha, coluna] > corte else "#172033",
            )
    ax.set_xticks([0, 1], ["Com\nnematoide", "Sem\nnematoide"])
    ax.set_yticks([0, 1], ["Com\nnematoide", "Sem\nnematoide"])
    ax.set_xlabel("Classe prevista", fontsize=13)
    ax.set_ylabel("Classe real", fontsize=13)
    ax.set_title(titulo, fontsize=15, fontweight="bold", pad=14)


def pagina_cenario(pdf: PdfPages, linha: pd.Series) -> None:
    qtd = int(linha["qtd_sensores"])
    sensores = str(linha["sensores"])
    valores = np.array(
        [
            [linha["real_com_previsto_com"], linha["real_com_previsto_sem"]],
            [linha["real_sem_previsto_com"], linha["real_sem_previsto_sem"]],
        ],
        dtype=int,
    )
    fig = plt.figure(figsize=(11.69, 8.27))
    eixo_matriz = fig.add_axes([0.08, 0.18, 0.48, 0.62])
    matriz(
        eixo_matriz,
        valores,
        f"{qtd} sensor{'es' if qtd > 1 else ''} MQ: {sensores}\n"
        f"Accuracy {linha['accuracy']:.2%} | Balanced accuracy {linha['balanced_accuracy']:.2%}",
    )

    erros = [
        ["Acertos: com nematoide", f"{int(valores[0, 0]):,}".replace(",", ".")],
        ["Erros: com -> sem", f"{int(valores[0, 1]):,}".replace(",", ".")],
        ["Erros: sem -> com", f"{int(valores[1, 0]):,}".replace(",", ".")],
        ["Acertos: sem nematoide", f"{int(valores[1, 1]):,}".replace(",", ".")],
    ]
    eixo_tabela = fig.add_axes([0.63, 0.26, 0.30, 0.38])
    eixo_tabela.axis("off")
    tabela = eixo_tabela.table(cellText=erros, colLabels=["Resultado", "Linhas"], loc="center", cellLoc="left")
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(11)
    tabela.scale(1.1, 1.8)
    for (linha_t, coluna), celula in tabela.get_celld().items():
        if linha_t == 0:
            celula.set_facecolor("#12355b")
            celula.get_text().set_color("white")
            celula.get_text().set_weight("bold")
        elif linha_t % 2:
            celula.set_facecolor("#eaf2fb")

    fig.text(0.08, 0.93, f"Teste {7 - qtd}/6 - melhor combinacao com {qtd} MQ", fontsize=21, fontweight="bold", color="#12355b")
    fig.text(
        0.63,
        0.72,
        "Sensores usados\n" + sensores.replace(" | ", " + ") + "\n\n"
        "As variaveis Soil, Temp e Pressao\nforam mantidas em todos os cenarios.",
        fontsize=13,
        va="top",
    )
    fig.text(0.08, 0.05, "Mesmo conjunto de teste; C16 removida somente do treino; compensacao ambiental recalculada no treino restante.", fontsize=9, color="#555555")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    resumo = pd.read_csv(RESULTADOS / "resumo_matrizes_confusao_melhores_combinacoes.csv")
    resumo = resumo.sort_values("qtd_sensores")
    if resumo["qtd_sensores"].tolist() != [1, 2, 3, 4, 5, 6]:
        raise ValueError("Resumo incompleto: sao necessarios os cenarios de 1 a 6 sensores.")

    seis = resumo.loc[resumo["qtd_sensores"].eq(6)].iloc[0]
    dupla = resumo.loc[resumo["qtd_sensores"].eq(2)].iloc[0]

    with PdfPages(PDF) as pdf:
        pagina_texto(
            pdf,
            "Relatorio - reducao de sensores MQ",
            [
                "Objetivo\nVerificar se e possivel reduzir a quantidade de sensores de gases sem perder desempenho na classificacao de nematoide.",
                "Cenario avaliado\nExtraTrees com o mesmo teste de 22.115 linhas. A coleta C16 foi removida apenas do treino; a compensacao ambiental foi recalculada usando o treino restante. Soil, temperatura e pressao permaneceram em todos os testes.",
                "Como ler o documento\nCada pagina seguinte mostra a melhor combinacao encontrada para uma quantidade fixa de 6, 5, 4, 3, 2 e 1 sensor MQ.",
            ],
        )

        # Comeca com o conjunto completo e chega na menor combinacao.
        for _, linha in resumo.sort_values("qtd_sensores", ascending=False).iterrows():
            pagina_cenario(pdf, linha)

        fig = plt.figure(figsize=(11.69, 8.27))
        eixo = fig.add_axes([0.06, 0.25, 0.88, 0.55])
        tabela = resumo.sort_values("qtd_sensores", ascending=False).copy()
        tabela["sensores"] = tabela["sensores"].str.replace(" | ", " + ", regex=False)
        tabela["accuracy"] = tabela["accuracy"].map("{:.2%}".format)
        tabela["balanced_accuracy"] = tabela["balanced_accuracy"].map("{:.2%}".format)
        eixo.axis("off")
        quadro = eixo.table(
            cellText=tabela[["qtd_sensores", "sensores", "accuracy", "balanced_accuracy"]].values,
            colLabels=["MQ", "Melhor combinacao", "Accuracy", "Balanced accuracy"],
            cellLoc="center",
            loc="center",
            colWidths=[0.08, 0.55, 0.18, 0.19],
        )
        quadro.auto_set_font_size(False)
        quadro.set_fontsize(11)
        quadro.scale(1, 2.1)
        for (linha_t, _), celula in quadro.get_celld().items():
            if linha_t == 0:
                celula.set_facecolor("#12355b")
                celula.get_text().set_color("white")
                celula.get_text().set_weight("bold")
            elif linha_t % 2:
                celula.set_facecolor("#eaf2fb")
        fig.text(0.06, 0.91, "Comparativo geral", fontsize=22, fontweight="bold", color="#12355b")
        fig.text(0.06, 0.12, "A tabela mostra somente a melhor combinacao de cada quantidade de MQ entre as combinacoes avaliadas.", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        pagina_texto(
            pdf,
            "Resumo final: 6 MQ versus a melhor combinacao",
            [
                f"Conjunto completo - 6 MQ\n{seis['sensores'].replace(' | ', ' + ')}\nAccuracy: {seis['accuracy']:.2%}; balanced accuracy: {seis['balanced_accuracy']:.2%}.\nErros: {int(seis['real_com_previsto_sem']):,} casos com nematoide previstos como sem e {int(seis['real_sem_previsto_com']):,} casos sem nematoide previstos como com.".replace(",", "."),
                f"Melhor combinacao encontrada - 2 MQ\n{dupla['sensores'].replace(' | ', ' + ')}\nAccuracy: {dupla['accuracy']:.2%}; balanced accuracy: {dupla['balanced_accuracy']:.2%}.\nErros: {int(dupla['real_com_previsto_sem']):,} casos com nematoide previstos como sem e {int(dupla['real_sem_previsto_com']):,} casos sem nematoide previstos como com.".replace(",", "."),
                "Interpretacao\nMQ3 + MQ135 apresentou sinal mais consistente neste teste. Remover sensores pode reduzir ruido, redundancia e variacao especifica de algumas coletas, ajudando o ExtraTrees a construir separacoes mais estaveis.",
                "Cuidado metodologico\nForam avaliadas varias combinacoes no mesmo teste e a melhor foi escolhida por esse resultado. Portanto, os 94,12% ainda podem estar otimistas. Antes de reduzir o hardware, a dupla deve ser validada em coletas ou ensaios independentes, sem usar esse teste para escolher os sensores.",
            ],
            tamanho=12.5,
        )

    print(PDF)


if __name__ == "__main__":
    main()
