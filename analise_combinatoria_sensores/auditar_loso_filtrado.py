"""Audita as previsoes LOSO do ET com cinco MQ e ambiente na base filtrada.

Uso: .venv/Scripts/python.exe analise_combinatoria_sensores/auditar_loso_filtrado.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "comparacao_auc_modelos_filtrada" / "predicoes_et_mq_ambiente.csv"
EXCLUSIONS = ROOT / "validacao_sem_cinco_coletas" / "coletas_excluidas.csv"
OUT = ROOT / "auditoria_loso_filtrado"


def summarize(frame: pd.DataFrame, unit: str) -> dict:
    truth = frame.Classe.to_numpy(dtype=int)
    predicted = frame.predicao.to_numpy(dtype=int)
    matrix = np.zeros((2, 2), dtype=int)
    np.add.at(matrix, (truth, predicted), 1)
    f1 = []
    for label in (0, 1):
        tp = matrix[label, label]
        fp = matrix[1 - label, label]
        fn = matrix[label, 1 - label]
        f1.append(2 * tp / (2 * tp + fp + fn))
    return {
        "unidade": unit,
        "n": len(frame),
        "acertos": int(np.trace(matrix)),
        "acuracia": float(np.trace(matrix) / matrix.sum()),
        "acuracia_balanceada": float(np.mean(np.diag(matrix) / matrix.sum(axis=1))),
        "f1_classe_0": float(f1[0]),
        "f1_classe_1": float(f1[1]),
        "f1_macro": float(np.mean(f1)),
        "real_0_prev_0": int(matrix[0, 0]),
        "real_0_prev_1": int(matrix[0, 1]),
        "real_1_prev_0": int(matrix[1, 0]),
        "real_1_prev_1": int(matrix[1, 1]),
    }


def bootstrap_balanced_accuracy(readings: pd.DataFrame, repetitions: int = 10_000) -> dict:
    """Reamostra vasos completos; preserva cinco vasos em cada classe."""
    blocks = (readings.assign(correct=readings.Classe.eq(readings.predicao).astype(int))
              .groupby(["Classe", "Vaso_id"], as_index=False)
              .agg(n=("correct", "size"), correct=("correct", "sum")))
    rng = np.random.default_rng(42)
    recalls = []
    for label in (0, 1):
        values = blocks.loc[blocks.Classe.eq(label), ["n", "correct"]].to_numpy()
        assert values.shape == (5, 2)
        selected = values[rng.integers(0, 5, size=(repetitions, 5))]
        recalls.append(selected[:, :, 1].sum(axis=1) / selected[:, :, 0].sum(axis=1))
    scores = (recalls[0] + recalls[1]) / 2
    lower, upper = np.quantile(scores, [.025, .975])
    return {
        "unidade": "leitura",
        "estatistica": "acuracia_balanceada",
        "repeticoes": repetitions,
        "semente": 42,
        "limite_inferior_95": float(lower),
        "limite_superior_95": float(upper),
        "metodo": "percentil; cinco vasos por classe com reposicao; previsoes LOSO fixas",
    }


def main() -> None:
    readings = pd.read_csv(PREDICTIONS)
    excluded = set(pd.read_csv(EXCLUSIONS).Coleta)
    assert len(readings) == 55_946
    assert readings.Coleta.nunique() == 31
    assert readings.Vaso_id.nunique() == 10
    assert readings.fold.nunique() == 10
    assert not (set(readings.Coleta) & excluded)
    assert readings.groupby("fold").Vaso_id.nunique().eq(1).all()
    assert readings.groupby("Vaso_id").fold.nunique().eq(1).all()
    assert readings.groupby("Coleta")[["Vaso_id", "Classe", "fold"]].nunique().eq(1).all().all()
    assert readings.groupby("Classe").Vaso_id.nunique().to_dict() == {0: 5, 1: 5}
    assert readings.probabilidade_classe_1.between(0, 1).all()

    readings["predicao"] = (readings.probabilidade_classe_1 >= .5).astype(int)
    collections = (readings.groupby(["Coleta", "Vaso_id", "Classe"], as_index=False)
                   .agg(n_leituras=("Classe", "size"),
                        probabilidade_classe_1=("probabilidade_classe_1", "mean")))
    collections["predicao"] = (collections.probabilidade_classe_1 >= .5).astype(int)
    vessels = (collections.groupby(["Vaso_id", "Classe"], as_index=False)
               .agg(n_coletas=("Coleta", "size"),
                    probabilidade_classe_1=("probabilidade_classe_1", "mean")))
    vessels["predicao"] = (vessels.probabilidade_classe_1 >= .5).astype(int)

    reading_counts = (readings.assign(correct=readings.Classe.eq(readings.predicao).astype(int))
                      .groupby("Vaso_id", as_index=False)
                      .agg(n_leituras=("correct", "size"), acertos_leituras=("correct", "sum")))
    collection_counts = (collections.assign(correct=collections.Classe.eq(collections.predicao).astype(int))
                         .groupby("Vaso_id", as_index=False)
                         .agg(acertos_coletas=("correct", "sum")))
    vessels = vessels.merge(reading_counts, on="Vaso_id", validate="one_to_one")
    vessels = vessels.merge(collection_counts, on="Vaso_id", validate="one_to_one")
    vessels["acerto_vaso"] = vessels.Classe.eq(vessels.predicao).astype(int)

    metrics = pd.DataFrame([
        summarize(readings, "leitura"),
        summarize(collections, "coleta"),
        summarize(vessels, "vaso"),
    ])
    k = int(vessels.acerto_vaso.sum())
    n = len(vessels)
    cp = pd.DataFrame([{
        "unidade": "vaso", "estatistica": "acuracia", "acertos": k, "n": n,
        "limite_inferior_95": 0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
        "limite_superior_95": 1.0 if k == n else float(beta.ppf(.975, k + 1, n - k)),
        "metodo": "Clopper-Pearson bilateral; previsoes LOSO fixas",
    }])
    bootstrap = pd.DataFrame([bootstrap_balanced_accuracy(readings)])

    OUT.mkdir(exist_ok=True)
    vessels.to_csv(OUT / "resultados_dez_vasos.csv", index=False)
    metrics.to_csv(OUT / "metricas.csv", index=False)
    cp.to_csv(OUT / "clopper_pearson.csv", index=False)
    bootstrap.to_csv(OUT / "bootstrap_por_vaso.csv", index=False)
    print(metrics.to_string(index=False))
    print(cp.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
