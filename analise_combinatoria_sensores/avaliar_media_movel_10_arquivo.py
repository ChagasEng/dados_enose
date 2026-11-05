"""Aplica media movel causal de 10 leituras a uma coleta e repete LOSO.

As primeiras nove leituras ficam inalteradas. Da decima em diante, cada
entrada passa a ser a media das dez leituras originais que terminam nela.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from rodar_combinacoes import DATASET, ENV, MQ, model
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parent
ORIGINAL_PREDICTIONS = (
    ROOT / "busca_mq_ambiente_loso_leituras" / "predicoes_vencedor_cru.csv"
)
FEATURES = ["MQ2", "MQ3", "MQ8", "MQ135", "MQ138", *ENV]
SMOOTH_COLUMNS = [*MQ, *ENV]


def smooth_one_collection(df: pd.DataFrame, collection: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = df.loc[df.Coleta.eq(collection)].sort_values("Tempo", kind="stable")
    if matches.empty:
        raise ValueError(f"Coleta nao encontrada: {collection}")
    if matches.Vaso_id.nunique() != 1 or matches.Classe.nunique() != 1:
        raise ValueError("Coleta com vaso ou classe inconsistentes")
    original = matches[SMOOTH_COLUMNS].astype(float).copy()
    rolling = original.rolling(window=10, min_periods=10).mean()
    transformed = original.copy()
    transformed.iloc[9:] = rolling.iloc[9:].to_numpy()
    if not transformed.iloc[:9].equals(original.iloc[:9]):
        raise AssertionError("As primeiras nove leituras devem permanecer intactas")
    if not np.allclose(transformed.iloc[9].to_numpy(dtype=float),
                       original.iloc[:10].mean().to_numpy(dtype=float)):
        raise AssertionError("A decima leitura nao e a media de 1 a 10")
    result = df.copy()
    result[SMOOTH_COLUMNS] = result[SMOOTH_COLUMNS].astype(float)
    result.loc[matches.index, SMOOTH_COLUMNS] = transformed.to_numpy()
    audit = matches[["Coleta", "Vaso_id", "Tempo", "Classe"]].copy()
    audit.insert(0, "posicao_na_coleta", np.arange(1, len(audit) + 1))
    for feature in SMOOTH_COLUMNS:
        audit[f"{feature}_original"] = original[feature].to_numpy()
        audit[f"{feature}_media_movel_10"] = transformed[feature].to_numpy()
    return result, audit


def loso_predictions(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for fold, (fit_idx, test_idx) in enumerate(
        LeaveOneGroupOut().split(df, df.Classe, df.Vaso_id), 1
    ):
        train, test = df.iloc[fit_idx], df.iloc[test_idx]
        assert not (set(train.Vaso_id) & set(test.Vaso_id))
        fitted = model().fit(train[FEATURES], train.Classe)
        probabilities = fitted.predict_proba(test[FEATURES])[:, 1]
        frame = test[["Coleta", "Vaso_id", "Tempo", "Classe"]].copy()
        frame["probabilidade_classe_1"] = probabilities
        frame["predicao"] = (probabilities >= 0.5).astype(int)
        frame.insert(0, "fold", fold)
        parts.append(frame)
        print("LOSO", fold, test.Vaso_id.iloc[0], flush=True)
    predicted = pd.concat(parts, ignore_index=True)
    if len(predicted) != len(df) or predicted.Coleta.nunique() != 36:
        raise AssertionError("LOSO incompleto")
    return predicted


def result_row(predictions: pd.DataFrame, label: str, collection: str) -> dict:
    correct = predictions.Classe.eq(predictions.predicao)
    target = predictions.loc[predictions.Coleta.eq(collection)]
    target_correct = target.Classe.eq(target.predicao)
    return {
        "cenario": label,
        "n_leituras_total": len(predictions),
        "n_acertos_total": int(correct.sum()),
        "acuracia_normal_total": float(correct.mean()),
        "coleta_modificada": collection,
        "n_leituras_coleta": len(target),
        "n_acertos_coleta": int(target_correct.sum()),
        "acuracia_normal_coleta": float(target_correct.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coleta", required=True,
                        help="Nome exato da coluna Coleta que corresponde ao arquivo destacado")
    args = parser.parse_args()
    df = pd.read_csv(DATASET)
    df["Vaso_id"] = vessel_id(df)
    transformed, audit = smooth_one_collection(df, args.coleta)
    output = ROOT / "media_movel_10_arquivo" / args.coleta.replace("/", "_").replace("\\", "_")
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "arquivo_original_e_suavizado.csv", index=False)

    original = pd.read_csv(ORIGINAL_PREDICTIONS)
    if len(original) != len(df):
        raise AssertionError("Predicoes originais nao correspondem ao dataset")
    new = loso_predictions(transformed)
    new.sort_values(["Coleta", "Tempo"]).to_csv(
        output / "predicoes_loso_com_media_movel.csv", index=False,
    )
    comparison = pd.DataFrame([
        result_row(original, "original_sem_media_movel", args.coleta),
        result_row(new, "uma_coleta_com_media_movel_10", args.coleta),
    ])
    comparison.to_csv(output / "comparacao_acuracia.csv", index=False)
    (output / "metodo.json").write_text(json.dumps({
        "dataset": str(DATASET),
        "coleta_modificada": args.coleta,
        "vaso_modificado": audit.Vaso_id.iloc[0],
        "regra": "Posicoes 1 a 9 intactas; posicao 10 = media original 1-10; posicao 11 = media original 2-11; e assim sucessivamente",
        "colunas_suavizadas": SMOOTH_COLUMNS,
        "metadados_preservados": ["Coleta", "Vaso_id", "Tempo", "Classe"],
        "modelo": "Mesmo ExtraTrees 900 arvores e mesmos sensores do vencedor com ambiente em dados crus",
        "validacao": "LOSO por vaso em todas as 36 coletas; apenas a coleta indicada recebe media movel",
        "metrica": "Acuracia normal por leitura crua",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
