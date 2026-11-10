"""Busca exaustiva de todas as 511 combinacoes dos seis MQ e tres variaveis ambientais.

Reaproveita os 126 resultados das buscas so MQ e MQ + ambiente completo,
treinando as outras 385 combinacoes no mesmo LOSO de 31 coletas e dez vasos.
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from rodar_combinacoes import DATASET, ENV, MQ, model
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "validacao_sem_cinco_coletas"
OUT = ROOT / "busca_nove_variaveis_loso"
EXPECTED_ROWS = 55946
EXPECTED_COLLECTIONS = 31
EXPECTED_VASES = 10
G_DF = None
G_FOLDS = None
G_THREADS = None


def key(mq: tuple[str, ...], environment: tuple[str, ...]) -> str:
    return " | ".join((*mq, *environment))


def all_candidates():
    sensors = (*MQ, *ENV)
    return [
        (tuple(x for x in subset if x in MQ), tuple(x for x in subset if x in ENV))
        for size in range(1, len(sensors) + 1)
        for subset in itertools.combinations(sensors, size)
    ]


def init_worker(threads: int) -> None:
    global G_DF, G_FOLDS, G_THREADS
    excluded = set(pd.read_csv(SOURCE / "coletas_excluidas.csv").Coleta)
    df = pd.read_csv(DATASET)
    df = df.loc[~df.Coleta.isin(excluded)].copy()
    df["Vaso_id"] = vessel_id(df)
    if len(df) != EXPECTED_ROWS or df.Coleta.nunique() != EXPECTED_COLLECTIONS or df.Vaso_id.nunique() != EXPECTED_VASES:
        raise AssertionError("Base filtrada inesperada")
    if df[["Coleta", "Classe", "Vaso_id", *MQ, *ENV]].isna().any().any():
        raise ValueError("Valores ausentes")
    G_DF = df
    G_FOLDS = list(LeaveOneGroupOut().split(df, df.Classe, df.Vaso_id))
    G_THREADS = threads


def evaluate(candidate):
    mq, environment = candidate
    features = [*mq, *environment]
    df = G_DF
    parts = []
    for fold, (fit_idx, test_idx) in enumerate(G_FOLDS, 1):
        train, test = df.iloc[fit_idx], df.iloc[test_idx]
        if set(train.Vaso_id) & set(test.Vaso_id):
            raise AssertionError("Vaso compartilhado entre treino e teste")
        fitted = model().set_params(n_jobs=G_THREADS).fit(train[features], train.Classe)
        pred = (fitted.predict_proba(test[features])[:, 1] >= 0.5).astype(int)
        parts.append(pd.DataFrame({
            "Coleta": test.Coleta.to_numpy(),
            "Vaso_id": test.Vaso_id.to_numpy(),
            "Classe": test.Classe.to_numpy(),
            "predicao": pred,
            "fold": fold,
        }))
    predictions = pd.concat(parts, ignore_index=True)
    if len(predictions) != EXPECTED_ROWS or predictions.Coleta.nunique() != EXPECTED_COLLECTIONS:
        raise AssertionError("Predicoes incompletas")
    correct = predictions.Classe.eq(predictions.predicao)
    n_correct = int(correct.sum())
    by_collection = (
        predictions.assign(correta=correct.astype(int))
        .groupby(["Coleta", "Vaso_id", "Classe"], as_index=False)
        .agg(n_leituras=("correta", "size"), n_acertos=("correta", "sum"))
    )
    by_collection.insert(0, "sensores_mq", " | ".join(mq))
    by_collection.insert(1, "sensores_ambiente", " | ".join(environment))
    row = {"sensores_mq": " | ".join(mq),
           "sensores_ambiente": " | ".join(environment),
           "n_mq": len(mq), "n_ambiente": len(environment),
           "n_variaveis": len(features), "n_leituras": EXPECTED_ROWS,
           "n_acertos": n_correct, "acuracia": n_correct / EXPECTED_ROWS}
    return row, by_collection


def load_existing() -> tuple[list[dict], pd.DataFrame]:
    if OUT.joinpath("ranking_511.csv").exists():
        saved = pd.read_csv(OUT / "ranking_511.csv")
        saved[["sensores_mq", "sensores_ambiente"]] = saved[["sensores_mq", "sensores_ambiente"]].fillna("")
        rows = saved.drop(columns="ranking").to_dict("records")
        collections = pd.read_csv(OUT / "acertos_por_coleta_511.csv").fillna(
            {"sensores_mq": "", "sensores_ambiente": ""})
        return rows, collections
    rows = []
    groups = []
    for suffix, environment in (("mq", ()), ("mq_ambiente", tuple(ENV))):
        rank = pd.read_csv(SOURCE / f"ranking_63_{suffix}.csv")
        col = pd.read_csv(SOURCE / f"acertos_por_coleta_63_{suffix}.csv")
        if len(rank) != 63 or len(col) != 63 * EXPECTED_COLLECTIONS:
            raise AssertionError("Resultados anteriores incompletos")
        for record in rank.itertuples(index=False):
            mq = tuple(record.sensores_mq.split(" | "))
            rows.append({"sensores_mq": record.sensores_mq,
                         "sensores_ambiente": " | ".join(environment),
                         "n_mq": len(mq), "n_ambiente": len(environment),
                         "n_variaveis": len(mq) + len(environment),
                         "n_leituras": int(record.n_leituras),
                         "n_acertos": int(record.n_acertos),
                         "acuracia": float(record.acuracia)})
        col.insert(1, "sensores_ambiente", " | ".join(environment))
        groups.append(col)
    return rows, pd.concat(groups, ignore_index=True)


def save(rows: list[dict], collections: pd.DataFrame) -> None:
    rank = (pd.DataFrame(rows)
            .sort_values(["acuracia", "n_variaveis", "n_mq", "sensores_mq", "sensores_ambiente"],
                         ascending=[False, True, False, True, True])
            .reset_index(drop=True))
    rank.insert(0, "ranking", range(1, len(rank) + 1))
    temp_rank = OUT / "ranking_511.tmp.csv"
    temp_col = OUT / "acertos_por_coleta_511.tmp.csv"
    rank.to_csv(temp_rank, index=False)
    collections.to_csv(temp_col, index=False)
    temp_rank.replace(OUT / "ranking_511.csv")
    temp_col.replace(OUT / "acertos_por_coleta_511.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    rows, collections = load_existing()
    finished = {(r["sensores_mq"], r["sensores_ambiente"]) for r in rows}
    if len(finished) != len(rows):
        raise AssertionError("Combinacoes duplicadas")
    if len(collections) != len(rows) * EXPECTED_COLLECTIONS:
        raise AssertionError("Checkpoint inconsistente")
    candidates = [c for c in all_candidates()
                  if (" | ".join(c[0]), " | ".join(c[1])) not in finished]
    if len(rows) + len(candidates) != 511:
        raise AssertionError("Busca nao cobre 511 combinacoes")
    save(rows, collections)
    print(f"{len(rows)} resultados existentes; {len(candidates)} faltantes", flush=True)
    if candidates:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=init_worker,
                                 initargs=(args.threads,)) as pool:
            futures = {pool.submit(evaluate, c): c for c in candidates}
            for index, future in enumerate(as_completed(futures), 1):
                row, by_collection = future.result()
                rows.append(row)
                collections = pd.concat([collections, by_collection], ignore_index=True)
                if index % 5 == 0 or index == len(candidates):
                    save(rows, collections)
                print(f"{len(rows)}/511 {key(futures[future][0], futures[future][1])}: "
                      f"{row['n_acertos']}/{EXPECTED_ROWS}", flush=True)
    if len(rows) != 511 or len(collections) != 511 * EXPECTED_COLLECTIONS:
        raise AssertionError("Busca incompleta")
    (OUT / "metodo.json").write_text(json.dumps({
        "dataset": str(DATASET), "coletas_excluidas": str(SOURCE / "coletas_excluidas.csv"),
        "validacao": "LOSO por vaso, 10 folds, 31 coletas, 55946 leituras",
        "modelo": "ExtraTreesClassifier, 900 arvores, max_features=sqrt, min_samples_leaf=10, seed=42",
        "busca": "Todos os subconjuntos nao vazios de seis MQ + tres variaveis ambientais: 511",
        "reaproveitados": 126,
        "novos": 385,
        "ranking": "Acuracia normal por leitura, desempate por menos variaveis e depois mais MQ",
        "limite": "Busca e avaliacao nos mesmos vasos; cinco coletas retiradas apos inspecao dos dados, tres por erros do modelo",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.read_csv(OUT / "ranking_511.csv").head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
