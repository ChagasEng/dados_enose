"""Busca exaustiva de MQ no LOSO, com ambiente opcional, por leitura/bloco."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from rodar_combinacoes import DATASET, ENV, MQ, model
from validacao_por_leitura import make_mean4
from validacao_por_vaso import vessel_id


def rank(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(
        ["acuracia_normal", "n_mq", "sensores_mq"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    result.insert(0, "ranking", range(1, len(result) + 1))
    return result


def evaluate(df: pd.DataFrame, subset: tuple[str, ...], folds,
             environment: tuple[str, ...] = ()) -> tuple[dict, pd.DataFrame]:
    features = [*subset, *environment]
    parts = []
    for fold, (fit_idx, test_idx) in enumerate(folds, 1):
        train, test = df.iloc[fit_idx], df.iloc[test_idx]
        assert not (set(train.Vaso_id) & set(test.Vaso_id))
        fitted = model().fit(train[features], train.Classe)
        probability = fitted.predict_proba(test[features])[:, 1]
        predicted = (probability >= 0.5).astype(np.int8)
        parts.append(pd.DataFrame({
            "Coleta": test.Coleta.to_numpy(),
            "Vaso_id": test.Vaso_id.to_numpy(),
            "Classe": test.Classe.to_numpy(),
            "predicao": predicted,
            "fold": fold,
        }))
    predictions = pd.concat(parts, ignore_index=True)
    if len(predictions) != len(df) or predictions.Coleta.nunique() != 36:
        raise AssertionError("Predicoes LOSO incompletas")
    correct = predictions.Classe.eq(predictions.predicao)
    summary = {
        "sensores_mq": " | ".join(subset),
        "n_mq": len(subset),
        "n_amostras": len(predictions),
        "n_acertos": int(correct.sum()),
        "acuracia_normal": float(correct.mean()),
    }
    by_collection = (
        predictions.assign(correta=correct.astype(int))
        .groupby(["Coleta", "Vaso_id", "Classe"], as_index=False)
        .agg(n_amostras=("correta", "size"), n_acertos=("correta", "sum"))
    )
    by_collection.insert(0, "sensores_mq", summary["sensores_mq"])
    by_collection["acuracia_dentro_da_coleta"] = (
        by_collection.n_acertos / by_collection.n_amostras
    )
    return summary, by_collection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--com-ambiente", action="store_true",
                        help="Acrescenta solo, temperatura e pressao a cada subconjunto MQ")
    args = parser.parse_args()
    out = (Path(__file__).resolve().parent /
           ("busca_mq_ambiente_loso_leituras" if args.com_ambiente
            else "busca_mq_loso_leituras"))
    out.mkdir(exist_ok=True)
    environment = tuple(ENV) if args.com_ambiente else ()
    raw = pd.read_csv(DATASET)
    if raw[["Coleta", "Vaso", "Classe", "Tempo", *MQ]].isna().any().any():
        raise ValueError("Valores ausentes")
    raw["Vaso_id"] = vessel_id(raw)
    mean4 = make_mean4(raw)
    subsets = [s for size in range(1, len(MQ) + 1)
               for s in itertools.combinations(MQ, size)]

    winners = {}
    for representation, df in (("cru", raw), ("media_4", mean4)):
        result_path = out / f"ranking_63_mq_{representation}.csv"
        collection_path = out / f"acertos_por_coleta_63_mq_{representation}.csv"
        results = (pd.read_csv(result_path).drop(columns="ranking").to_dict("records")
                   if result_path.exists() else [])
        collections = pd.read_csv(collection_path) if collection_path.exists() else pd.DataFrame()
        finished = {x["sensores_mq"] for x in results}
        if finished:
            count = collections.groupby("sensores_mq").size().to_dict()
            if any(count.get(label) != 36 for label in finished):
                raise ValueError("Checkpoint inconsistente")
        folds = list(LeaveOneGroupOut().split(df, df.Classe, df.Vaso_id))
        if len(folds) != 10:
            raise AssertionError("Esperados 10 folds LOSO")
        for number, subset in enumerate(subsets, 1):
            label = " | ".join(subset)
            if label in finished:
                print(representation, f"{number}/63", "ja calculado", label, flush=True)
                continue
            summary, by_collection = evaluate(df, subset, folds, environment)
            results.append(summary)
            collections = pd.concat([collections, by_collection], ignore_index=True)
            # Salva checkpoint por combinacao para permitir retomar a busca.
            collections.to_csv(collection_path, index=False)
            rank(pd.DataFrame(results)).to_csv(result_path, index=False)
            print(representation, f"{number}/63", label,
                  f"{summary['n_acertos']}/{summary['n_amostras']}", flush=True)
        ranking = rank(pd.DataFrame(results))
        if len(ranking) != 63:
            raise AssertionError("Busca incompleta")
        winners[representation] = ranking.iloc[0].to_dict()
        print(representation, "vencedor", ranking.head(8).to_string(index=False), flush=True)

    (out / "metodo.json").write_text(json.dumps({
        "dataset": str(DATASET),
        "mq": list(MQ),
        "ambiente": list(environment) if environment else "Excluido de todos os modelos",
        "combinacoes_por_representacao": 63,
        "representacoes": {"cru": len(raw), "media_4": len(mean4)},
        "media_4": "Medias de quatro leituras consecutivas da mesma coleta, ordenadas por Tempo; ultimo bloco incompleto descartado",
        "validacao": "10 folds LOSO por vaso biologico; todas as amostras e coletas do vaso ficam fora do treino",
        "metrica": "Acuracia normal por leitura crua ou por bloco medio de 4; nao agrega previsoes por coleta",
        "desempate": "Menor numero de MQ; depois ordem alfabetica do conjunto",
        "modelo": "ExtraTreesClassifier, 900 arvores, max_features=sqrt, min_samples_leaf=10, seed=42",
        "vencedores": winners,
        "limite": "Selecao e avaliacao usam os mesmos dez vasos; amostras dentro de vasos sao correlacionadas",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
