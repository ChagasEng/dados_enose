"""Busca exaustiva de MQ, com e sem ambiente, sem selecionar no teste.

Uso: .venv/Scripts/python.exe analise_combinatoria_sensores/rodar_combinacoes.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "06_07_melhor_modelo" / "dados" / "dataset_melhor_modelo_sensores_corrigidos.csv"
OUT = Path(__file__).resolve().parent
MQ = ("MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138")
ENV = ("Soil_indice_0_1", "Temp_C", "Pres_kPa")
SEED = 42
N_TREES = 900


def model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=N_TREES, random_state=SEED, n_jobs=-1,
        max_features="sqrt", min_samples_leaf=10, bootstrap=False,
    )


def metrics(y_true, y_pred, prefix: str) -> dict:
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def score(frame: pd.DataFrame, probabilities: np.ndarray) -> tuple[dict, pd.DataFrame]:
    readings = frame[["Coleta", "Classe"]].copy()
    readings["probabilidade_classe_1"] = probabilities
    by_collection = readings.groupby("Coleta", sort=True).agg(
        Classe=("Classe", "first"),
        probabilidade_classe_1=("probabilidade_classe_1", "mean"),
        n_leituras=("Classe", "size"),
    ).reset_index()
    by_collection["predicao"] = (by_collection.probabilidade_classe_1 >= 0.5).astype(int)
    result = metrics(by_collection.Classe, by_collection.predicao, "coleta")
    return result, by_collection


def collection_folds(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = train.groupby("Coleta", sort=True).Classe.first().reset_index()
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    folds = []
    for train_groups, valid_groups in splitter.split(groups.Coleta, groups.Classe):
        valid_names = set(groups.iloc[valid_groups].Coleta)
        train_names = set(groups.iloc[train_groups].Coleta)
        assert not (valid_names & train_names)
        folds.append((np.flatnonzero(train.Coleta.isin(train_names)),
                      np.flatnonzero(train.Coleta.isin(valid_names))))
    assert sorted(np.concatenate([v for _, v in folds]).tolist()) == list(range(len(train)))
    return folds


def main() -> None:
    df = pd.read_csv(DATASET)
    required = ["Coleta", "Classe", "Conjunto", *MQ, *ENV]
    if df[required].isna().any().any():
        raise ValueError("Valores ausentes em entradas, rotulos ou identificadores")
    if set(df.Conjunto) != {"Treino", "Teste"}:
        raise ValueError("A coluna Conjunto deve conter Treino e Teste")
    if df.groupby("Coleta").Classe.nunique().max() != 1:
        raise ValueError("Uma coleta contem rotulos distintos")
    train = df.loc[df.Conjunto.eq("Treino")].reset_index(drop=True)
    test = df.loc[df.Conjunto.eq("Teste")].reset_index(drop=True)
    if set(train.Coleta) & set(test.Coleta):
        raise ValueError("Uma coleta aparece no treino e no teste")
    folds = collection_folds(train)
    subsets = [s for size in range(1, len(MQ) + 1)
               for s in itertools.combinations(MQ, size)]
    rows = []
    for scenario in ("mq", "mq_ambiente"):
        for number, subset in enumerate(subsets, 1):
            features = list(subset) + (list(ENV) if scenario == "mq_ambiente" else [])
            oof = np.full(len(train), np.nan)
            for fit_idx, valid_idx in folds:
                fitted = model().fit(train.loc[fit_idx, features], train.loc[fit_idx, "Classe"])
                oof[valid_idx] = fitted.predict_proba(train.loc[valid_idx, features])[:, 1]
            if not np.isfinite(oof).all():
                raise AssertionError("Predicoes de validacao cruzada incompletas")
            result, _ = score(train, oof)
            rows.append({"cenario": scenario, "sensores_mq": " | ".join(subset),
                         "n_mq": len(subset), "n_features": len(features), **result})
            print(f"{scenario}: {number}/{len(subsets)} - {'+'.join(subset)}", flush=True)

    results = pd.DataFrame(rows).sort_values(
        ["cenario", "coleta_balanced_accuracy", "coleta_f1_macro",
         "n_mq", "sensores_mq"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    results.to_csv(OUT / "validacao_cruzada_126_combinacoes.csv", index=False)

    # Seleciona o melhor da CV e o baseline com os seis MQ em cada cenario.
    # A selecao nao consulta as 12 coletas do teste reservado.
    selected = []
    for scenario in ("mq", "mq_ambiente"):
        scenario_rows = results.loc[results.cenario.eq(scenario)].copy()
        best = scenario_rows.iloc[0]
        selected.append((scenario, "melhor_cv", best))
        baseline = scenario_rows.loc[scenario_rows.n_mq.eq(6)].iloc[0]
        selected.append((scenario, "todos_6_mq", baseline))

    evaluations = []
    predictions = []
    seen = set()
    for scenario, selection, row in selected:
        subset = tuple(row.sensores_mq.split(" | "))
        features = list(subset) + (list(ENV) if scenario == "mq_ambiente" else [])
        key = (scenario, subset)
        if key in seen:
            continue
        seen.add(key)
        fitted = model().fit(train[features], train.Classe)
        test_probability = fitted.predict_proba(test[features])[:, 1]
        test_metrics, by_collection = score(test, test_probability)
        by_collection.insert(0, "cenario", scenario)
        by_collection.insert(1, "selecao", selection)
        by_collection.insert(2, "sensores_mq", row.sensores_mq)
        predictions.append(by_collection)
        evaluations.append({"cenario": scenario, "selecao": selection,
                            "sensores_mq": row.sensores_mq, "n_mq": len(subset),
                            "cv_coleta_balanced_accuracy": row.coleta_balanced_accuracy,
                            **test_metrics})
    pd.DataFrame(evaluations).to_csv(OUT / "avaliacao_teste_reservado.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "predicoes_teste_por_coleta.csv", index=False)

    paired = results.pivot(index="sensores_mq", columns="cenario",
                           values="coleta_balanced_accuracy").reset_index()
    paired["delta_ambiente_pp"] = 100 * (paired.mq_ambiente - paired.mq)
    paired.to_csv(OUT / "efeito_ambiente_mesmos_mq_cv.csv", index=False)
    metadata = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "linhas_treino": len(train), "coletas_treino": train.Coleta.nunique(),
        "linhas_teste": len(test), "coletas_teste": test.Coleta.nunique(),
        "combinacoes_por_cenario": len(subsets), "seed": SEED,
        "modelo": f"ExtraTreesClassifier(n_estimators={N_TREES}, max_features='sqrt', min_samples_leaf=10, bootstrap=False)",
        "validacao": "4 folds estratificados por rotulo da coleta, apenas nas 24 coletas de treino; predicoes fora da amostra agregadas por coleta",
        "selecao": "Maior acuracia balanceada por coleta na validacao; desempates por F1 macro por coleta, menor n_mq e nome",
        "teste": "Teste preexistente de 12 coletas; avaliado apenas para o melhor da CV e o baseline de 6 MQ de cada cenario; modelos identicos avaliados uma vez",
        "mq": "Sinais brutos; sem compensacao ambiental nos dois cenarios",
        "ambiente": list(ENV),
        "unidade_de_avaliacao": "Coleta em todos os cenarios, folds, rankings e metricas; nenhuma metrica por leitura",
        "predicao_coleta": "Media das probabilidades da classe 1 das leituras da coleta; limiar 0.5",
        "classe_0": "Com nematoide", "classe_1": "Sem nematoide",
    }
    (OUT / "metodo.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(evaluations).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
