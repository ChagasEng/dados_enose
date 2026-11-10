"""Resume previsões LOSO já salvas para a base completa, sem novo treino.

Unidades de avaliação: leitura, coleta e vaso. A decisão por vaso é a
média simples das probabilidades das suas coletas, para não privilegiar
coletas com maior número de leituras.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "busca_mq_ambiente_loso_leituras" / "predicoes_vencedor_cru.csv"
OUT = ROOT / "avaliacao_parecer_base_completa"


def metricas(frame: pd.DataFrame, unidade: str) -> dict:
    return {
        "unidade": unidade,
        "n": len(frame),
        "acertos": int(frame.Classe.eq(frame.predicao).sum()),
        "acuracia": float(accuracy_score(frame.Classe, frame.predicao)),
        "acuracia_balanceada": float(balanced_accuracy_score(frame.Classe, frame.predicao)),
        "f1_macro": float(f1_score(frame.Classe, frame.predicao, average="macro")),
    }


def bootstrap_por_vaso(frame: pd.DataFrame, unidade: str,
                       repeticoes: int = 10000) -> dict:
    """Reamostra cinco vasos de cada classe; modelo e subconjunto ficam fixos."""
    blocos = (frame.assign(correto=frame.Classe.eq(frame.predicao).astype(int))
              .groupby(["Classe", "Vaso_id"], as_index=False)
              .agg(n=("correto", "size"), acertos=("correto", "sum")))
    if blocos.groupby("Classe").size().to_dict() != {0: 5, 1: 5}:
        raise AssertionError("Esperados cinco vasos em cada classe")
    rng = np.random.default_rng(42)
    draws = []
    for classe in (0, 1):
        subset = blocos.loc[blocos.Classe.eq(classe), ["n", "acertos"]].to_numpy()
        indices = rng.integers(0, 5, size=(repeticoes, 5))
        selected = subset[indices]
        draws.append((selected[:, :, 1].sum(axis=1),
                      selected[:, :, 0].sum(axis=1)))
    balanced = (draws[0][0] / draws[0][1] +
                draws[1][0] / draws[1][1]) / 2
    return {
        "unidade": unidade,
        "repeticoes": repeticoes,
        "limite_inferior_95": float(np.quantile(balanced, .025)),
        "limite_superior_95": float(np.quantile(balanced, .975)),
        "metodo": "bootstrap estratificado por classe, reamostrando vasos; "
                  "IC condicional ao modelo e as entradas ja escolhidas",
    }


def main() -> None:
    df = pd.read_csv(PREDICTIONS)
    if len(df) != 63492 or df.Coleta.nunique() != 36 or df.Vaso_id.nunique() != 10:
        raise AssertionError("Base LOSO completa inesperada")
    if df.groupby("Coleta").agg(vasos=("Vaso_id", "nunique"),
                                classes=("Classe", "nunique")).max().max() != 1:
        raise AssertionError("Coleta com múltiplos vasos ou classes")

    por_coleta = (df.groupby(["Coleta", "Vaso_id", "Classe"], as_index=False)
                  .agg(n_leituras=("Classe", "size"),
                       probabilidade_classe_1=("probabilidade_classe_1", "mean")))
    por_coleta["predicao"] = (por_coleta.probabilidade_classe_1 >= .5).astype(int)

    por_vaso = (por_coleta.groupby(["Vaso_id", "Classe"], as_index=False)
                .agg(n_coletas=("Coleta", "size"),
                     probabilidade_classe_1=("probabilidade_classe_1", "mean")))
    por_vaso["predicao"] = (por_vaso.probabilidade_classe_1 >= .5).astype(int)

    OUT.mkdir(exist_ok=True)
    por_coleta.to_csv(OUT / "previsoes_por_coleta.csv", index=False)
    por_vaso.to_csv(OUT / "previsoes_por_vaso.csv", index=False)
    resumo = pd.DataFrame([
        metricas(df, "leitura"),
        metricas(por_coleta, "coleta"),
        metricas(por_vaso, "vaso"),
    ])
    resumo.to_csv(OUT / "metricas_tres_unidades.csv", index=False)
    intervalos = pd.DataFrame([
        bootstrap_por_vaso(df, "leitura"),
        bootstrap_por_vaso(por_coleta, "coleta"),
        bootstrap_por_vaso(por_vaso, "vaso"),
    ])
    intervalos.to_csv(OUT / "intervalos_bootstrap_vaso.csv", index=False)
    print(resumo.to_string(index=False))
    print(intervalos[["unidade", "limite_inferior_95", "limite_superior_95"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
