"""Compara os acertos do melhor ExtraTrees em solo molhado e seco.

O corte e feito apenas para estratificar o conjunto de teste ja definido por
coleta. O modelo nao e retreinado para cada faixa: assim a comparacao mostra
em que condicao o mesmo modelo generaliza melhor.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
DATASET = BASE / "dados" / "dataset_melhor_modelo_sensores_corrigidos.csv"
MODELO = BASE / "modelo" / "modelo_extra_trees_melhor_93_20.joblib"
SAIDA = BASE / "analise_seco_molhado"
TARGET = "Classe"
SOIL = "Soil_indice_0_1"
CORTE_SOIL = 0.4
FEATURES = [
    "MQ2_corrigido_env",
    "MQ3_corrigido_env",
    "MQ7_corrigido_env",
    "MQ8_corrigido_env",
    "MQ135_corrigido_env",
    "MQ138_corrigido_env",
    SOIL,
    "Temp_C",
    "Pres_kPa",
]


def faixa_umidade(indice: float) -> str:
    """Converte o indice operacional do sensor em uma faixa comparavel.

    Esta e a regra operacional definida para este experimento: indice ate 0,4
    e seco; acima de 0,4 e molhado. Ela substitui qualquer convencao generica
    do sensor e deve ser mantida em todas as comparacoes desta rodada.
    """

    return "seco" if indice <= CORTE_SOIL else "molhado"


def metricas_faixa(df: pd.DataFrame, faixa: str) -> dict[str, object]:
    y_true = df[TARGET]
    y_pred = df["predito"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "faixa": faixa,
        "regra_soil": f"{SOIL} {'<=' if faixa == 'seco' else '>'} {CORTE_SOIL}",
        "linhas_teste": int(len(df)),
        "soil_indice_min": float(df[SOIL].min()),
        "soil_indice_max": float(df[SOIL].max()),
        "soil_indice_mediana": float(df[SOIL].median()),
        "classe_0_com_nematoide": int((y_true == 0).sum()),
        "classe_1_sem_nematoide": int((y_true == 1).sum()),
        "acertos": int(df["acertou"].sum()),
        "erros": int((~df["acertou"]).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "acerto_classe_0": float(recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)[0]),
        "acerto_classe_1": float(recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)[1]),
        "matriz_confusao": cm.tolist(),
    }


def salvar_matriz(df: pd.DataFrame, faixa: str) -> None:
    cm = confusion_matrix(df[TARGET], df["predito"], labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["real_com_nematoide", "real_sem_nematoide"],
        columns=["previsto_com_nematoide", "previsto_sem_nematoide"],
    ).to_csv(SAIDA / f"matriz_confusao_{faixa}.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    display = ConfusionMatrixDisplay(cm, display_labels=["Com nematoide", "Sem nematoide"])
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Melhor modelo - solo {faixa}")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    fig.tight_layout()
    fig.savefig(SAIDA / f"matriz_confusao_{faixa}.png", dpi=180)
    plt.close(fig)


def salvar_grafico_resumo(resumo: pd.DataFrame) -> None:
    plot_df = resumo.set_index("faixa").loc[["seco", "molhado"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(plot_df.index, plot_df["accuracy"] * 100, color=["#d9822b", "#247ba0"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Acertos no teste (%)")
    ax.set_title("Acertos do melhor modelo por condicao do solo", pad=14)
    ax.grid(axis="y", alpha=0.2)
    for bar, row in zip(bars, plot_df.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{row.accuracy:.2%}\n({row.acertos:,}/{row.linhas_teste:,})".replace(",", "."),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(SAIDA / "comparacao_acertos_seco_molhado.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def salvar_readme(resumo: pd.DataFrame) -> None:
    molhado = resumo.loc[resumo["faixa"] == "molhado"].iloc[0]
    seco = resumo.loc[resumo["faixa"] == "seco"].iloc[0]
    diferenca = (molhado["accuracy"] - seco["accuracy"]) * 100
    texto = f"""# Acertos por condicao do solo

Esta analise usa o mesmo ExtraTrees e o mesmo conjunto de teste por coleta do melhor modelo. Ela nao retreina um modelo para solo seco ou molhado: apenas separa as predicoes do teste para comparar a generalizacao em cada condicao.

## Regra operacional

- `seco`: `{SOIL} <= {CORTE_SOIL}`
- `molhado`: `{SOIL} > {CORTE_SOIL}`

O indice e uma normalizacao min-max da leitura analogica `Soil`. A convencao desta rodada foi definida como indice ate 0,4 = seco e acima de 0,4 = molhado. Portanto, os nomes representam faixas operacionais do experimento, nao percentual fisico de agua no solo.

## Resultado no teste

| Faixa | Linhas | Acertos | Accuracy | Balanced accuracy | Acerto classe 0 | Acerto classe 1 |
|---|---:|---:|---:|---:|---:|---:|
| Molhado | {int(molhado.linhas_teste)} | {int(molhado.acertos)} | {molhado.accuracy:.2%} | {molhado.balanced_accuracy:.2%} | {molhado.acerto_classe_0:.2%} | {molhado.acerto_classe_1:.2%} |
| Seco | {int(seco.linhas_teste)} | {int(seco.acertos)} | {seco.accuracy:.2%} | {seco.balanced_accuracy:.2%} | {seco.acerto_classe_0:.2%} | {seco.acerto_classe_1:.2%} |

A accuracy no solo molhado ficou {diferenca:.2f} pontos percentuais acima da faixa seca. Como as faixas possuem proporcoes diferentes das classes, a balanced accuracy e indispensavel para comparar o desempenho sem deixar a classe majoritaria mascarar o resultado.

## Arquivos

- `resumo_acertos_seco_molhado.csv`: comparacao consolidada.
- `predicoes_teste_com_faixa_soil.csv`: cada predicao do teste, com o indice e a faixa.
- `matriz_confusao_molhado.*` e `matriz_confusao_seco.*`: erros por classe em cada faixa.
- `comparacao_acertos_seco_molhado.png`: grafico de acertos.
"""
    (SAIDA / "README.md").write_text(texto, encoding="utf-8")


def main() -> None:
    df = pd.read_csv(DATASET)
    required = [TARGET, "Conjunto", "Coleta", "Tempo", "Soil", *FEATURES]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset local: {missing}")
    if not MODELO.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {MODELO}")

    teste = df.loc[df["Conjunto"] == "Teste"].copy()
    modelo = joblib.load(MODELO)
    teste["predito"] = modelo.predict(teste[FEATURES])
    teste["prob_sem_nematoide"] = modelo.predict_proba(teste[FEATURES])[:, list(modelo.classes_).index(1)]
    teste["acertou"] = teste[TARGET] == teste["predito"]
    teste["faixa_soil"] = teste[SOIL].map(faixa_umidade)

    SAIDA.mkdir(exist_ok=True)
    colunas_saida = [
        "Coleta", "Dia", "Vaso", "Tempo", "Soil", SOIL, "faixa_soil", TARGET,
        "predito", "prob_sem_nematoide", "acertou",
    ]
    teste[colunas_saida].to_csv(
        SAIDA / "predicoes_teste_com_faixa_soil.csv", index=False, encoding="utf-8-sig"
    )

    registros = []
    for faixa in ("molhado", "seco"):
        bloco = teste.loc[teste["faixa_soil"] == faixa].copy()
        if bloco.empty:
            raise ValueError(f"Nenhuma linha de teste na faixa {faixa}.")
        registros.append(metricas_faixa(bloco, faixa))
        salvar_matriz(bloco, faixa)

    resumo = pd.DataFrame(registros)
    resumo.to_csv(SAIDA / "resumo_acertos_seco_molhado.csv", index=False, encoding="utf-8-sig")
    with (SAIDA / "metricas_acertos_seco_molhado.json").open("w", encoding="utf-8") as arquivo:
        json.dump(
            {
                "modelo": str(MODELO.relative_to(BASE)),
                "dataset": str(DATASET.relative_to(BASE)),
                "metodo": "Mesmo modelo treinado no conjunto Treino; estratificacao posterior apenas no conjunto Teste.",
                "corte_soil_indice_0_1": CORTE_SOIL,
                "convencao": "Regra operacional definida: indice <= 0,4 = seco; indice > 0,4 = molhado.",
                "resultados": registros,
            },
            arquivo,
            indent=2,
            ensure_ascii=False,
        )
    salvar_grafico_resumo(resumo)
    salvar_readme(resumo)

    print(resumo[["faixa", "linhas_teste", "acertos", "accuracy", "balanced_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
