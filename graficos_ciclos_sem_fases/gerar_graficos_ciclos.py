from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent

# Os datasets de entrada ja tiveram 15% do inicio e 15% do fim removidos.
# No protocolo ilustrado, a coleta ocupa aproximadamente 15%--80% do ciclo.
# Portanto, ainda restam 5% do ciclo original (1/14 do dataset processado)
# pertencentes ao inicio da dessaturacao.
FRACAO_FINAL_A_REMOVER = 1 / 14
TEMPO_INICIAL_S = 180
TEMPO_FINAL_S = 600
LIMITE_Y = 30_000
PICO_ALVO_VISUAL = 27_000

SINAIS = ["Soil", "Temp.", "Pres.", "MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]

CORES = {
    "Soil": "#4d4d4d",
    "Temp.": "#e41a1c",
    "Pres.": "#2ca02c",
    "MQ2": "#1f77b4",
    "MQ3": "#17becf",
    "MQ7": "#e377c2",
    "MQ8": "#9467bd",
    "MQ135": "#bcbd22",
    "MQ138": "#8c564b",
}


@dataclass(frozen=True)
class Cenario:
    arquivo: Path
    coleta: str
    titulo: str
    nome_saida: str
    condicao_solo: str
    estado_planta: str
    condicao_pressao: str
    suavizar: bool = False


CENARIOS = [
    Cenario(
        arquivo=ROOT
        / "dataset_processado_por_dia_vaso_sem_vref0"
        / "dataset_unico_por_dia_vaso_sem_vref0.csv",
        coleta="dia 12 - Soja Saudavel Vaso 2",
        titulo="Solo seco — soja saudável — com pressão",
        nome_saida="01_solo_seco_saudavel_com_pressao.png",
        condicao_solo="seco",
        estado_planta="saudável",
        condicao_pressao="com pressão",
    ),
    Cenario(
        arquivo=ROOT
        / "dataset_processado_por_dia_vaso_sem_vref0"
        / "dataset_unico_por_dia_vaso_sem_vref0.csv",
        coleta="dia 12 - Soja Heterodera Vaso 1",
        titulo="Solo seco — soja com nematoide — com pressão",
        nome_saida="02_solo_seco_nematoide_com_pressao.png",
        condicao_solo="seco",
        estado_planta="com nematoide",
        condicao_pressao="com pressão",
    ),
    Cenario(
        arquivo=ROOT
        / "15_07_dia_20_mais_completo"
        / "dados"
        / "dataset_dia_20_mais_com_ambiente.csv",
        coleta="dia 21 sem pressao - Soja Sauda",
        titulo="Solo molhado — soja saudável — sem pressão",
        nome_saida="03_solo_molhado_saudavel_sem_pressao.png",
        condicao_solo="molhado",
        estado_planta="saudável",
        condicao_pressao="sem pressão",
    ),
    Cenario(
        arquivo=ROOT
        / "15_07_dia_20_mais_completo"
        / "dados"
        / "dataset_dia_20_mais_com_ambiente.csv",
        coleta="Página8",
        titulo="Solo molhado — soja com nematoide — sem pressão",
        nome_saida="04_solo_molhado_nematoide_sem_pressao.png",
        condicao_solo="molhado",
        estado_planta="com nematoide",
        condicao_pressao="sem pressão",
        suavizar=True,
    ),
    Cenario(
        arquivo=ROOT
        / "dataset_processado_por_dia_vaso_sem_vref0"
        / "dataset_unico_por_dia_vaso_sem_vref0.csv",
        coleta="dia 18 - Soja Saudavel Vaso 4",
        titulo="Solo molhado — soja saudável — com pressão",
        nome_saida="05_solo_molhado_saudavel_com_pressao.png",
        condicao_solo="molhado",
        estado_planta="saudável",
        condicao_pressao="com pressão",
    ),
    Cenario(
        arquivo=ROOT
        / "dataset_processado_por_dia_vaso_sem_vref0"
        / "dataset_unico_por_dia_vaso_sem_vref0.csv",
        coleta="dia 18 - Soja Heterodera Vaso 9",
        titulo="Solo molhado — soja com nematoide — com pressão",
        nome_saida="06_solo_molhado_nematoide_com_pressao.png",
        condicao_solo="molhado",
        estado_planta="com nematoide",
        condicao_pressao="com pressão",
    ),
]


def milhares_ptbr(valor: float, _pos: int) -> str:
    return f"{valor:,.0f}".replace(",", ".")


def carregar_cenarios() -> list[tuple[Cenario, pd.DataFrame, int]]:
    cache: dict[Path, pd.DataFrame] = {}
    resultado: list[tuple[Cenario, pd.DataFrame, int]] = []

    for cenario in CENARIOS:
        if cenario.arquivo not in cache:
            cache[cenario.arquivo] = pd.read_csv(cenario.arquivo)

        dados = cache[cenario.arquivo]
        faltantes = [coluna for coluna in ["Coleta", *SINAIS] if coluna not in dados.columns]
        if faltantes:
            raise ValueError(f"Colunas ausentes em {cenario.arquivo}: {faltantes}")

        coleta = dados.loc[dados["Coleta"].eq(cenario.coleta), SINAIS].copy()
        if coleta.empty:
            raise ValueError(f"Coleta nao encontrada: {cenario.coleta}")

        coleta = coleta.apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both")
        total_processado = len(coleta)
        remover_final = round(total_processado * FRACAO_FINAL_A_REMOVER)
        if remover_final:
            coleta = coleta.iloc[:-remover_final].reset_index(drop=True)

        if cenario.suavizar:
            # A mediana curta remove picos isolados; a media movel reduz o ruido
            # de alta frequencia sem apagar a tendencia temporal dos sensores.
            coleta[SINAIS] = (
                coleta[SINAIS]
                .rolling(window=21, center=True, min_periods=1)
                .median()
                .rolling(window=25, center=True, min_periods=1)
                .mean()
            )

        # O usuario solicitou a apresentacao da fase de coleta entre 180 e 600 s.
        coleta.index = pd.Index(
            np.linspace(TEMPO_INICIAL_S, TEMPO_FINAL_S, len(coleta)),
            name="Tempo_s",
        )
        resultado.append((cenario, coleta, total_processado))

    return resultado


def configurar_eixo(
    ax: plt.Axes,
    titulo: str,
    mostrar_eixo_y: bool = True,
) -> None:
    ax.set_title(titulo, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Tempo (s)", fontsize=10)
    if mostrar_eixo_y:
        ax.set_ylabel("Resposta dos sensores (escala comum reescalonada)", fontsize=10)
    ax.set_xlim(TEMPO_INICIAL_S, TEMPO_FINAL_S)
    ax.set_ylim(0, LIMITE_Y)
    ax.set_xticks(np.arange(TEMPO_INICIAL_S, TEMPO_FINAL_S + 1, 60))
    ax.yaxis.set_major_locator(MultipleLocator(5_000))
    ax.yaxis.set_major_formatter(FuncFormatter(milhares_ptbr))
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.7, alpha=0.9)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.45, linestyle=":", alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def desenhar_sinais(ax: plt.Axes, dados: pd.DataFrame, fator_escala: float) -> None:
    for sinal in SINAIS:
        ax.plot(
            dados.index,
            dados[sinal] * fator_escala,
            color=CORES[sinal],
            linewidth=1.15,
            label=sinal,
        )


def gerar_figuras_individuais(
    dados_cenarios: list[tuple[Cenario, pd.DataFrame, int]], fator_escala: float
) -> None:
    for cenario, dados, _total_processado in dados_cenarios:
        fig, ax = plt.subplots(figsize=(11.5, 6.3), constrained_layout=True)
        desenhar_sinais(ax, dados, fator_escala)
        configurar_eixo(ax, cenario.titulo)
        ax.legend(
            title="Sinal",
            bbox_to_anchor=(1.015, 0.5),
            loc="center left",
            frameon=True,
            fontsize=9,
            title_fontsize=9,
        )
        fig.savefig(OUTPUT_DIR / cenario.nome_saida, dpi=300, facecolor="white")
        plt.close(fig)


def gerar_painel(
    dados_cenarios: list[tuple[Cenario, pd.DataFrame, int]], fator_escala: float
) -> None:
    fig, eixos = plt.subplots(
        2,
        2,
        figsize=(16, 10),
        sharey=True,
        constrained_layout=True,
    )

    for indice, (ax, (cenario, dados, _total_processado)) in enumerate(
        zip(eixos.flat, dados_cenarios)
    ):
        desenhar_sinais(ax, dados, fator_escala)
        configurar_eixo(
            ax,
            cenario.titulo,
            mostrar_eixo_y=indice % 2 == 0,
        )

    legenda = [
        Line2D([0], [0], color=CORES[sinal], linewidth=1.5, label=sinal) for sinal in SINAIS
    ]
    fig.legend(
        handles=legenda,
        title="Sinal",
        loc="outside right center",
        frameon=True,
        fontsize=9,
        title_fontsize=9,
    )
    fig.suptitle(
        "Sinais do nariz eletrônico durante a fase de coleta",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_DIR / "00_painel_quatro_cenarios.png", dpi=300, facecolor="white")
    plt.close(fig)


def gerar_painel_oito_cenarios(
    dados_cenarios: list[tuple[Cenario, pd.DataFrame, int]], fator_escala: float
) -> None:
    """Mostra o desenho 2 x 2 x 2 e explicita as combinacoes nao coletadas."""

    dados_por_condicao = {
        (cenario.condicao_solo, cenario.condicao_pressao, cenario.estado_planta): dados
        for cenario, dados, _total_processado in dados_cenarios
    }
    linhas = [
        ("seco", "com pressão"),
        ("seco", "sem pressão"),
        ("molhado", "com pressão"),
        ("molhado", "sem pressão"),
    ]
    colunas = ["saudável", "com nematoide"]

    fig, eixos = plt.subplots(
        4,
        2,
        figsize=(16, 18),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for indice_linha, (solo, pressao) in enumerate(linhas):
        for indice_coluna, planta in enumerate(colunas):
            ax = eixos[indice_linha, indice_coluna]
            titulo_planta = "soja saudável" if planta == "saudável" else "soja com nematoide"
            titulo = f"Solo {solo} — {titulo_planta} — {pressao}"
            configurar_eixo(ax, titulo, mostrar_eixo_y=indice_coluna == 0)

            dados = dados_por_condicao.get((solo, pressao, planta))
            if dados is not None:
                desenhar_sinais(ax, dados, fator_escala)
                continue

            ax.set_facecolor("#fafafa")
            ax.text(
                0.5,
                0.56,
                "DADOS NÃO COLETADOS",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="#9b2c2c",
            )
            ax.text(
                0.5,
                0.44,
                "Não há coleta de solo seco sem pressão\nnos arquivos experimentais.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#555555",
            )

    legenda = [
        Line2D([0], [0], color=CORES[sinal], linewidth=1.5, label=sinal) for sinal in SINAIS
    ]
    fig.legend(
        handles=legenda,
        title="Sinal",
        loc="outside right center",
        frameon=True,
        fontsize=9,
        title_fontsize=9,
    )
    fig.suptitle(
        "Sinais do nariz eletrônico — combinações de solo, pressão e sanidade",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_DIR / "00_painel_oito_cenarios.png", dpi=300, facecolor="white")
    plt.close(fig)


def salvar_resumo(
    dados_cenarios: list[tuple[Cenario, pd.DataFrame, int]], fator_escala: float
) -> None:
    linhas = []
    for cenario, dados, total_processado in dados_cenarios:
        linhas.append(
            {
                "arquivo_grafico": cenario.nome_saida,
                "coleta_origem": cenario.coleta,
                "solo": cenario.condicao_solo,
                "planta": cenario.estado_planta,
                "pressao": cenario.condicao_pressao,
                "amostras_arquivo_processado": total_processado,
                "amostras_grafico_somente_coleta": len(dados),
                "amostras_finais_removidas_dessaturacao_residual": total_processado - len(dados),
                "soil_mediana": dados["Soil"].median(),
                "tempo_inicial_s": TEMPO_INICIAL_S,
                "tempo_final_s": TEMPO_FINAL_S,
                "suavizacao_aplicada": "sim" if cenario.suavizar else "não",
                "fator_escala_visual_comum": fator_escala,
                "pico_alvo_visual": PICO_ALVO_VISUAL,
                "tempo_inicial_origem": "fase de aquecimento ja removida no dataset processado",
            }
        )

    pd.DataFrame(linhas).to_csv(
        OUTPUT_DIR / "resumo_graficos.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlecolor": "#202020",
            "axes.labelcolor": "#303030",
            "xtick.color": "#303030",
            "ytick.color": "#303030",
        }
    )

    dados_cenarios = carregar_cenarios()
    # Mantem exatamente a escala visual definida pelo painel original. Os
    # quatro primeiros cenarios sao as coletas que originaram esse fator.
    maior_pico_bruto = max(
        dados[SINAIS].max().max() for _, dados, _ in dados_cenarios[:4]
    )
    fator_escala = PICO_ALVO_VISUAL / maior_pico_bruto

    gerar_figuras_individuais(dados_cenarios, fator_escala)
    gerar_painel(dados_cenarios, fator_escala)
    gerar_painel_oito_cenarios(dados_cenarios, fator_escala)
    salvar_resumo(dados_cenarios, fator_escala)

    print(f"Graficos gerados em: {OUTPUT_DIR}")
    print(f"Fator de escala visual comum: {fator_escala:.6f}")
    for cenario, dados, total_processado in dados_cenarios:
        print(
            f"- {cenario.nome_saida}: {len(dados)} amostras "
            f"({total_processado - len(dados)} removidas do final residual)"
        )
    print(
        "- Solo seco sem pressão (saudável e com nematoide): "
        "dados não coletados; células identificadas no painel completo"
    )


if __name__ == "__main__":
    main()
