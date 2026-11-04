"""X-TCN MQ + media de Soil por coleta, somente no solo seco sob pressao."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from avaliar_xtcn_original_por_umidade import DATASET, MQ, ROOT, SOIL_CUTOFF, train_and_predict


OUTPUT = ROOT / "08_10_2026_xtcn_solo_medio/resultados_media_solo_seco_pressao"
FEATURES = MQ + ["Soil_media_coleta"]
FEATURES_RAW = MQ + ["Soil"]


def metricas(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro")),
            "recall_classe_0": float(recall_score(y, pred, pos_label=0)),
            "recall_classe_1": float(recall_score(y, pred, pos_label=1))}


def main() -> None:
    df = pd.read_csv(DATASET).dropna(subset=["Coleta", "Classe", "Tempo", "Soil", *MQ]).copy()
    df["Classe"] = df.Classe.astype(int)
    if df.groupby("Coleta").Classe.nunique().max() != 1:
        raise ValueError("Coleta com mais de uma classe")
    soil_min, soil_max = float(df.Soil.min()), float(df.Soil.max())
    groups = df.groupby("Coleta", as_index=False).agg(
        Classe=("Classe", "first"), Soil_media_coleta=("Soil", "mean"),
        n_leituras=("Soil", "size"))
    groups["Soil_indice_medio"] = (groups.Soil_media_coleta - soil_min) / (soil_max - soil_min)
    dry_groups = groups[groups.Soil_indice_medio <= SOIL_CUTOFF].copy()
    dry = df[df.Coleta.isin(dry_groups.Coleta)].merge(
        dry_groups[["Coleta", "Soil_media_coleta"]], on="Coleta", validate="many_to_one")

    summary = dry_groups.groupby("Classe", as_index=False).agg(
        coletas=("Coleta", "size"), leituras=("n_leituras", "sum"),
        media_solo_por_coleta=("Soil_media_coleta", "mean"),
        dp_solo_por_coleta=("Soil_media_coleta", "std"),
        menor_media_coleta=("Soil_media_coleta", "min"),
        maior_media_coleta=("Soil_media_coleta", "max"))
    summary["condicao"] = summary.Classe.map({0: "com nematoide", 1: "sem nematoide"})

    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    pred_windows, pred_groups, raw_windows, raw_groups, stump_groups = [], [], [], [], []
    for fold, (train_idx, test_idx) in enumerate(
        cv.split(dry_groups[["Soil_media_coleta"]], dry_groups.Classe, dry_groups.Coleta), start=1
    ):
        training_groups = dry_groups.iloc[train_idx]
        testing_groups = dry_groups.iloc[test_idx]
        train = dry[dry.Coleta.isin(training_groups.Coleta)]
        test = dry[dry.Coleta.isin(testing_groups.Coleta)]
        windows = train_and_predict(train, test, FEATURES)
        windows["fold"] = fold
        by_group = windows.groupby("Coleta", as_index=False).agg(
            classe_real=("classe_real", "first"),
            probabilidade_saudavel=("probabilidade_saudavel", "mean"),
            n_janelas=("classe_real", "size"))
        by_group["classe_prevista"] = (by_group.probabilidade_saudavel >= .5).astype(int)
        by_group["fold"] = fold
        by_group = by_group.merge(testing_groups[["Coleta", "Soil_media_coleta"]], on="Coleta", validate="one_to_one")
        pred_windows.append(windows)
        pred_groups.append(by_group)
        original_windows = train_and_predict(train, test, FEATURES_RAW)
        original_windows["fold"] = fold
        original_groups = original_windows.groupby("Coleta", as_index=False).agg(
            classe_real=("classe_real", "first"),
            probabilidade_saudavel=("probabilidade_saudavel", "mean"),
            n_janelas=("classe_real", "size"))
        original_groups["classe_prevista"] = (original_groups.probabilidade_saudavel >= .5).astype(int)
        original_groups["fold"] = fold
        raw_windows.append(original_windows)
        raw_groups.append(original_groups)
        soil_only = DecisionTreeClassifier(max_depth=1, random_state=42)
        soil_only.fit(training_groups[["Soil_media_coleta"]], training_groups.Classe)
        stump_groups.append(pd.DataFrame({
            "Coleta": testing_groups.Coleta, "classe_real": testing_groups.Classe,
            "classe_prevista": soil_only.predict(testing_groups[["Soil_media_coleta"]]),
            "fold": fold, "Soil_media_coleta": testing_groups.Soil_media_coleta,
        }))
        print(f"Fold {fold}: {len(testing_groups)} coletas secas", flush=True)

    windows = pd.concat(pred_windows, ignore_index=True)
    collections = pd.concat(pred_groups, ignore_index=True)
    raw_window_predictions = pd.concat(raw_windows, ignore_index=True)
    raw_collection_predictions = pd.concat(raw_groups, ignore_index=True)
    stump = pd.concat(stump_groups, ignore_index=True)
    if collections.Coleta.nunique() != len(dry_groups):
        raise AssertionError("Faltaram coletas na validacao")
    result = pd.DataFrame([
        {"modelo": "X-TCN MQ + Soil bruto", "unidade": "janela", "n": len(raw_window_predictions),
         **metricas(raw_window_predictions.classe_real, raw_window_predictions.classe_prevista)},
        {"modelo": "X-TCN MQ + Soil bruto", "unidade": "coleta", "n": len(raw_collection_predictions),
         **metricas(raw_collection_predictions.classe_real, raw_collection_predictions.classe_prevista)},
        {"modelo": "X-TCN MQ + media de Soil por coleta", "unidade": "janela", "n": len(windows),
         **metricas(windows.classe_real, windows.classe_prevista)},
        {"modelo": "X-TCN MQ + media de Soil por coleta", "unidade": "coleta", "n": len(collections),
         **metricas(collections.classe_real, collections.classe_prevista)},
        {"modelo": "Arvore de 1 corte, somente media de Soil", "unidade": "coleta", "n": len(stump),
         **metricas(stump.classe_real, stump.classe_prevista)},
    ])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "medias_solo_seco_por_classe.csv", index=False)
    result.to_csv(OUTPUT / "metricas_validacao_cruzada.csv", index=False)
    windows.to_csv(OUTPUT / "predicoes_janelas.csv", index=False)
    collections.to_csv(OUTPUT / "predicoes_coletas.csv", index=False)
    raw_window_predictions.to_csv(OUTPUT / "predicoes_janelas_solo_bruto.csv", index=False)
    raw_collection_predictions.to_csv(OUTPUT / "predicoes_coletas_solo_bruto.csv", index=False)
    stump.to_csv(OUTPUT / "controle_somente_solo.csv", index=False)
    (OUTPUT / "metodo.json").write_text(json.dumps({
        "dataset": str(DATASET.relative_to(ROOT)),
        "regime": "coletas antes do dia 20, sob pressao forçada, apos filtragem de variacoes de pressao",
        "faixa_seca": "indice min-max da media de Soil por coleta <= 0.4",
        "soil_min": soil_min, "soil_max": soil_max,
        "media_como_entrada": "media de Soil da propria coleta, calculada sem usar Classe",
        "resumo_por_classe": "apenas descritivo, calculado apos selecionar as coletas secas",
        "validacao": "4-fold StratifiedGroupKFold por Coleta, seed 42",
        "features_xtcn": FEATURES,
        "features_controle_solo_bruto": FEATURES_RAW,
        "epocas_xtcn": 60,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
