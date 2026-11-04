"""Audita a acuracia seco/umido da X-TCN em quatro folds por coleta."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedGroupKFold

from avaliar_xtcn_original_por_umidade import (
    DATASET, FEATURES, OUTPUT, SOIL_CUTOFF, metrics, train_and_predict,
)


def main() -> None:
    df = pd.read_csv(DATASET).dropna(subset=["Coleta", "Tempo", "Classe", *FEATURES]).copy()
    df["Classe"] = df.Classe.astype(int)
    soil_min, soil_max = float(df.Soil.min()), float(df.Soil.max())
    collections = df.groupby("Coleta", as_index=False).agg(
        classe_real=("Classe", "first"), soil_medio=("Soil", "mean"))
    collections["soil_indice_medio"] = (collections.soil_medio - soil_min) / (soil_max - soil_min)
    collections["faixa"] = np.where(collections.soil_indice_medio <= SOIL_CUTOFF, "seco", "umido")
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    all_windows, all_collections = [], []
    for fold, (train_idx, test_idx) in enumerate(cv.split(df[FEATURES], df.Classe, df.Coleta), start=1):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        predictions = train_and_predict(train, test)
        predictions["fold"] = fold
        predictions = predictions.merge(collections[["Coleta", "soil_medio", "soil_indice_medio", "faixa"]],
                                        on="Coleta", validate="many_to_one")
        by_collection = predictions.groupby("Coleta", as_index=False).agg(
            classe_real=("classe_real", "first"),
            probabilidade_saudavel=("probabilidade_saudavel", "mean"),
            n_janelas=("classe_real", "size"),
            faixa=("faixa", "first"), soil_medio=("soil_medio", "first"),
            soil_indice_medio=("soil_indice_medio", "first"))
        by_collection["classe_prevista"] = (by_collection.probabilidade_saudavel >= .5).astype(int)
        by_collection["fold"] = fold
        all_windows.append(predictions)
        all_collections.append(by_collection)
        print(f"Fold {fold}: {len(by_collection)} coletas; "
              f"acuracia por coleta {metrics(by_collection)['accuracy']:.3f}", flush=True)
    windows = pd.concat(all_windows, ignore_index=True)
    by_collection = pd.concat(all_collections, ignore_index=True)
    if by_collection.Coleta.nunique() != df.Coleta.nunique():
        raise AssertionError("Nem todas as coletas receberam previsao fora do treino")
    rows = []
    for faixa in ("total", "seco", "umido"):
        w = windows if faixa == "total" else windows[windows.faixa == faixa]
        g = by_collection if faixa == "total" else by_collection[by_collection.faixa == faixa]
        wm, gm = metrics(w), metrics(g)
        rows.append({"faixa": faixa, "n_coletas": len(g), "n_janelas": len(w),
                     "coletas_classe_0": int((g.classe_real == 0).sum()),
                     "coletas_classe_1": int((g.classe_real == 1).sum()),
                     "media_soil_das_coletas": float(g.soil_medio.mean()),
                     "accuracy_janela": wm["accuracy"],
                     "f1_macro_janela": wm["f1_macro"],
                     "recall_macro_janela": float(recall_score(w.classe_real, w.classe_prevista, average="macro")),
                     "balanced_accuracy_janela": wm["balanced_accuracy"],
                     "accuracy_coleta": gm["accuracy"],
                     "f1_macro_coleta": gm["f1_macro"],
                     "recall_macro_coleta": float(recall_score(g.classe_real, g.classe_prevista, average="macro")),
                     "balanced_accuracy_coleta": gm["balanced_accuracy"]})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "tabela_validacao_cruzada_4_folds.csv", index=False)
    windows.to_csv(OUTPUT / "predicoes_janelas_cv.csv", index=False)
    by_collection.to_csv(OUTPUT / "predicoes_coletas_cv.csv", index=False)
    (OUTPUT / "metodo_cv.json").write_text(json.dumps({
        "split": "StratifiedGroupKFold 4 folds, shuffle=True, random_state=42, groups=Coleta",
        "features": FEATURES, "faixa": "media de Soil por coleta; indice <=0.4 seco, >0.4 umido",
        "metricas": "predicoes fora do treino agregadas sobre todos os folds",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
