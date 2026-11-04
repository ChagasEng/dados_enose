"""RF em solo seco sob pressao, nos mesmos folds por coleta da X-TCN."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from avaliar_xtcn_original_por_umidade import DATASET, MQ, ROOT, SOIL_CUTOFF


OUTPUT = ROOT / "08_10_2026_xtcn_solo_medio/resultados_media_solo_seco_pressao"
SCENARIOS = {
    "RF MQ": (RandomForestClassifier, MQ),
    "RF MQ + Soil bruto": (RandomForestClassifier, MQ + ["Soil"]),
    "RF MQ + media de Soil": (RandomForestClassifier, MQ + ["Soil_media_coleta"]),
    "Extra Trees MQ + Soil bruto": (ExtraTreesClassifier, MQ + ["Soil"]),
}


def metrics(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}


def main():
    df = pd.read_csv(DATASET).dropna(subset=["Coleta", "Classe", "Soil", *MQ]).copy()
    df.Classe = df.Classe.astype(int)
    soil_min, soil_max = float(df.Soil.min()), float(df.Soil.max())
    groups = df.groupby("Coleta", as_index=False).agg(
        Classe=("Classe", "first"), Soil_media_coleta=("Soil", "mean"))
    groups["indice"] = (groups.Soil_media_coleta - soil_min) / (soil_max - soil_min)
    groups = groups[groups.indice <= SOIL_CUTOFF].copy()
    dry = df[df.Coleta.isin(groups.Coleta)].merge(
        groups[["Coleta", "Soil_media_coleta"]], on="Coleta", validate="many_to_one")
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    folds = list(cv.split(groups[["Soil_media_coleta"]], groups.Classe, groups.Coleta))
    results, all_predictions = [], []
    for name, (model_type, features) in SCENARIOS.items():
        reading_predictions, collection_predictions = [], []
        for fold, (train_idx, test_idx) in enumerate(folds, start=1):
            train_groups = groups.iloc[train_idx]
            test_groups = groups.iloc[test_idx]
            train = dry[dry.Coleta.isin(train_groups.Coleta)]
            test = dry[dry.Coleta.isin(test_groups.Coleta)]
            model = model_type(n_estimators=300, random_state=42, n_jobs=-1,
                               class_weight="balanced")
            model.fit(train[features], train.Classe)
            probabilities = model.predict_proba(test[features])[:, 1]
            readings = pd.DataFrame({"Coleta": test.Coleta.to_numpy(),
                                     "classe_real": test.Classe.to_numpy(),
                                     "probabilidade_saudavel": probabilities,
                                     "classe_prevista": (probabilities >= .5).astype(int),
                                     "modelo": name, "fold": fold})
            collections = readings.groupby("Coleta", as_index=False).agg(
                classe_real=("classe_real", "first"),
                probabilidade_saudavel=("probabilidade_saudavel", "mean"))
            collections["classe_prevista"] = (collections.probabilidade_saudavel >= .5).astype(int)
            collections["modelo"], collections["fold"] = name, fold
            reading_predictions.append(readings)
            collection_predictions.append(collections)
        for unit, chunks in (("leitura", reading_predictions), ("coleta", collection_predictions)):
            pred = pd.concat(chunks, ignore_index=True)
            if unit == "coleta" and pred.Coleta.nunique() != len(groups):
                raise AssertionError("Coleta ausente na validacao")
            results.append({"modelo": name, "unidade": unit, "n": len(pred),
                            **metrics(pred.classe_real, pred.classe_prevista)})
            all_predictions.append(pred.assign(unidade=unit))
        print(f"Concluido: {name}", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT / "comparacao_rf_extratrees_seco_cv.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        OUTPUT / "predicoes_rf_extratrees_seco_cv.csv", index=False)
    (OUTPUT / "metodo_rf_cv.json").write_text(json.dumps({
        "dataset": str(DATASET.relative_to(ROOT)),
        "grupo": "Coleta", "faixa": "media de Soil por coleta, indice <=0.4",
        "folds": "4-fold StratifiedGroupKFold em 17 coletas secas, seed 42",
        "n_arvores": 300, "class_weight": "balanced", "features": {k:v[1] for k,v in SCENARIOS.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
