"""Busca exaustiva dos seis MQ por LOSO de vaso, acuracia normal por coleta.

Uso: .venv/Scripts/python.exe analise_combinatoria_sensores/buscar_mq_loso.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from rodar_combinacoes import DATASET, MQ
from validacao_por_vaso import predict_collections, vessel_id


OUT = Path(__file__).resolve().parent / "busca_mq_loso"
RESULTS = OUT / "ranking_63_combinacoes_mq_loso.csv"
PREDICTIONS = OUT / "predicoes_63_combinacoes_mq_loso.csv"


def rank(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(
        ["acuracia_coleta", "n_mq", "sensores_mq"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    frame.insert(0, "ranking", range(1, len(frame) + 1))
    return frame


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATASET)
    required = ["Coleta", "Vaso", "Classe", *MQ]
    if df[required].isna().any().any():
        raise ValueError("Valores ausentes")
    df["Vaso_id"] = vessel_id(df)
    if df.Vaso_id.nunique() != 10 or df.Coleta.nunique() != 36:
        raise ValueError("Esperados 10 vasos e 36 coletas")
    logo = LeaveOneGroupOut()
    folds = list(logo.split(df, df.Classe, df.Vaso_id))
    if len(folds) != 10:
        raise AssertionError("LOSO deveria ter 10 folds")

    subsets = [subset for size in range(1, len(MQ) + 1)
               for subset in itertools.combinations(MQ, size)]
    rows = pd.read_csv(RESULTS).drop(columns="ranking").to_dict("records") if RESULTS.exists() else []
    saved_predictions = pd.read_csv(PREDICTIONS) if PREDICTIONS.exists() else pd.DataFrame()
    finished = {row["sensores_mq"] for row in rows}
    if finished and saved_predictions.empty:
        raise ValueError("Resultados parciais sem predicoes correspondentes")

    for number, subset in enumerate(subsets, 1):
        label = " | ".join(subset)
        if label in finished:
            print(f"{number}/{len(subsets)} ja calculado: {label}", flush=True)
            continue
        fold_predictions = []
        for fold, (fit_idx, test_idx) in enumerate(folds, 1):
            train, test = df.iloc[fit_idx], df.iloc[test_idx]
            assert not (set(train.Vaso_id) & set(test.Vaso_id))
            predicted = predict_collections(train, test, list(subset))
            predicted.insert(0, "fold", fold)
            fold_predictions.append(predicted)
        predictions = pd.concat(fold_predictions, ignore_index=True)
        if len(predictions) != 36 or predictions.Coleta.nunique() != 36:
            raise AssertionError("Faltam coletas nas predicoes LOSO")
        predictions.insert(0, "sensores_mq", label)
        predictions.insert(1, "n_mq", len(subset))
        correct = predictions.Classe.eq(predictions.predicao)
        rows.append({
            "sensores_mq": label,
            "n_mq": len(subset),
            "n_coletas": len(predictions),
            "n_acertos": int(correct.sum()),
            "acuracia_coleta": float(correct.mean()),
            "vasos_sem_erros": int(predictions.assign(correta=correct).groupby("Vaso_id").correta.all().sum()),
        })
        saved_predictions = pd.concat([saved_predictions, predictions], ignore_index=True)
        rank(pd.DataFrame(rows)).to_csv(RESULTS, index=False)
        saved_predictions.to_csv(PREDICTIONS, index=False)
        print(f"{number}/{len(subsets)} {label}: {int(correct.sum())}/36", flush=True)

    final = rank(pd.DataFrame(rows))
    if len(final) != len(subsets):
        raise AssertionError("Busca incompleta")
    winner = final.iloc[0]
    (OUT / "metodo.json").write_text(json.dumps({
        "dataset": str(DATASET),
        "sensores_avaliados": list(MQ),
        "variaveis_ambientais": "Excluidas de todos os modelos",
        "numero_combinacoes": len(subsets),
        "validacao": "LOSO por vaso biologico (10 folds); cada vaso e todas as suas coletas ficam fora do treino uma vez",
        "unidade_de_avaliacao": "Coleta; media das probabilidades das leituras, limiar 0.5",
        "metrica_principal": "Acuracia normal por coleta = numero de coletas corretas / 36",
        "desempate": "Menor numero de MQ; depois ordem alfabetica do conjunto",
        "modelo": "ExtraTreesClassifier, 900 arvores, max_features=sqrt, min_samples_leaf=10, seed=42",
        "melhor_observado": winner.to_dict(),
        "limite": "O conjunto e escolhido com os mesmos dez vasos usados para estimar seu desempenho; necessita confirmacao em vasos novos",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(final.head(15).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
