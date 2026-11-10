"""Busca LOSO de 63 subconjuntos MQ, com e sem ambiente, em 32 coletas.

Mantem dez vasos. Salva checkpoint apos cada combinacao para retomada.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from rodar_combinacoes import DATASET, ENV, MQ, model
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validacao_sem_quatro_coletas_erradas"
BAD = (
    "dia 12 - Soja Saudavel Vaso 2",
    "dia 19 - Soja Heterodera Vaso 9",
    "dia 19 - Soja Saudavel Vaso 2",
    "dia 19 - Soja Saudavel Vaso 1",
)
EXTRA_NOISY = "dia 19 - Soja Heterodera Vaso 1"


def evaluate(df: pd.DataFrame, features: list[str], folds) -> tuple[int, pd.DataFrame]:
    parts = []
    for fold, (fit_idx, test_idx) in enumerate(folds, 1):
        train, test = df.iloc[fit_idx], df.iloc[test_idx]
        if set(train.Vaso_id) & set(test.Vaso_id):
            raise AssertionError("Vaso compartilhado entre treino e teste")
        fitted = model().fit(train[features], train.Classe)
        predicted = (fitted.predict_proba(test[features])[:, 1] >= 0.5).astype(int)
        parts.append(pd.DataFrame({
            "Coleta": test.Coleta.to_numpy(),
            "Vaso_id": test.Vaso_id.to_numpy(),
            "Classe": test.Classe.to_numpy(),
            "predicao": predicted,
            "fold": fold,
        }))
    pred = pd.concat(parts, ignore_index=True)
    if len(pred) != len(df) or pred.Coleta.nunique() != df.Coleta.nunique():
        raise AssertionError("Predicoes LOSO incompletas")
    correct = pred.Classe.eq(pred.predicao)
    by_collection = (
        pred.assign(correta=correct.astype(int))
        .groupby(["Coleta", "Vaso_id", "Classe"], as_index=False)
        .agg(n_leituras=("correta", "size"), n_acertos=("correta", "sum"))
    )
    return int(correct.sum()), by_collection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--somar-ruido", action="store_true",
                        help="Tambem retirar a coleta inoculada ruidosa do dia 19")
    args = parser.parse_args()
    excluded = (*BAD, EXTRA_NOISY) if args.somar_ruido else BAD
    out = (OUT.parent / "validacao_sem_cinco_coletas"
           if args.somar_ruido else OUT)
    out.mkdir(exist_ok=True)
    raw = pd.read_csv(DATASET)
    raw["Vaso_id"] = vessel_id(raw)
    df = raw.loc[~raw.Coleta.isin(excluded)].copy()
    expected_rows, expected_collections = (55946, 31) if args.somar_ruido else (57207, 32)
    if len(df) != expected_rows or df.Coleta.nunique() != expected_collections or df.Vaso_id.nunique() != 10:
        raise AssertionError("Base filtrada inesperada")
    if df[["Coleta", "Vaso_id", "Classe", *MQ, *ENV]].isna().any().any():
        raise ValueError("Valores ausentes")
    folds = list(LeaveOneGroupOut().split(df, df.Classe, df.Vaso_id))
    if len(folds) != 10:
        raise AssertionError("Esperados dez folds")
    subsets = [x for n in range(1, len(MQ) + 1) for x in itertools.combinations(MQ, n)]
    for scenario, environment in (("mq", ()), ("mq_ambiente", ENV)):
        ranking_path = out / f"ranking_63_{scenario}.csv"
        collection_path = out / f"acertos_por_coleta_63_{scenario}.csv"
        results = pd.read_csv(ranking_path).drop(columns="ranking").to_dict("records") if ranking_path.exists() else []
        collections = pd.read_csv(collection_path) if collection_path.exists() else pd.DataFrame()
        finished = {row["sensores_mq"] for row in results}
        if finished:
            counts = collections.groupby("sensores_mq").size().to_dict()
            if any(counts.get(label) != expected_collections for label in finished):
                raise ValueError("Checkpoint inconsistente")
        for number, subset in enumerate(subsets, 1):
            label = " | ".join(subset)
            if label in finished:
                continue
            features = [*subset, *environment]
            n_correct, by_collection = evaluate(df, features, folds)
            by_collection.insert(0, "sensores_mq", label)
            collections = pd.concat([collections, by_collection], ignore_index=True)
            results.append({"sensores_mq": label, "n_mq": len(subset),
                            "n_leituras": len(df), "n_acertos": n_correct,
                            "acuracia": n_correct / len(df)})
            ranked = (pd.DataFrame(results)
                      .sort_values(["acuracia", "n_mq", "sensores_mq"],
                                   ascending=[False, True, True])
                      .reset_index(drop=True))
            ranked.insert(0, "ranking", range(1, len(ranked) + 1))
            collections.to_csv(collection_path, index=False)
            ranked.to_csv(ranking_path, index=False)
            print(f"{scenario} {number}/63 {label}: {n_correct}/{len(df)}", flush=True)
        final = pd.read_csv(ranking_path)
        if len(final) != 63:
            raise AssertionError("Busca incompleta")
        print(f"{scenario} vencedor:\n{final.head(3).to_string(index=False)}", flush=True)
    (out / "metodo_busca.json").write_text(json.dumps({
        "dataset": str(DATASET),
        "coletas_excluidas": excluded,
        "modelo": "ExtraTreesClassifier, 900 arvores, max_features=sqrt, min_samples_leaf=10, seed=42",
        "validacao": f"Dez folds LOSO por vaso, {expected_collections} coletas, {expected_rows} leituras",
        "busca": "63 subconjuntos MQ; cenarios sem ambiente e com Soil_indice_0_1, Temp_C e Pres_kPa",
        "metrica": "Acuracia normal por leitura; desempate por menor numero de MQ",
        "limite": "Quatro coletas selecionadas pelos erros do modelo e uma pelo ruido na mesma base; busca e validacao nos mesmos vasos" if args.somar_ruido else "Coletas selecionadas pelos erros do modelo na mesma base; busca e validacao nos mesmos vasos",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
