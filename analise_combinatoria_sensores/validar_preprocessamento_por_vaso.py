"""LOSO com parametros de solo e pressao ajustados apenas no treino.

Compara a faixa estrita de pressao (mediana do treino +/- 0,50 kPa) com
o mesmo conjunto apos o filtro fixo de saltos, sem a faixa estrita.
As configuracoes de entradas sao fixas: este script nao faz selecao aninhada.

Uso: .venv/Scripts/python.exe analise_combinatoria_sensores/validar_preprocessamento_por_vaso.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "comparacao" / "pressao_filtrada" / "antes_dia_20_pressao_filtrada.csv"
OUT = Path(__file__).resolve().parent / "validacao_preprocessamento_por_vaso"
MQ = ("MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138")
ENV = ("Soil_indice_0_1", "Temp_C", "Pres_kPa")
CONFIGS = {
    "nove_entradas": (*MQ, *ENV),
    "somente_ambientais": ENV,
    "somente_mq": MQ,
    "cinco_mq_ambiente_exploratorio": ("MQ2", "MQ3", "MQ8", "MQ135", "MQ138", *ENV),
}
SCENARIOS = ("faixa_estrita_treino", "sem_faixa_estrita")


def metrics(frame: pd.DataFrame, scenario: str, config: str, unit: str) -> dict:
    truth = frame["Classe"].to_numpy()
    predicted = frame["predicao"].to_numpy()
    return {
        "cenario": scenario,
        "configuracao": config,
        "unidade": unit,
        "n": int(len(frame)),
        "acertos": int(np.sum(truth == predicted)),
        "acuracia": float(accuracy_score(truth, predicted)),
        "acuracia_balanceada": float(balanced_accuracy_score(truth, predicted)),
        "f1_macro": float(f1_score(truth, predicted, average="macro")),
    }


def prepare_fold(train: pd.DataFrame, test: pd.DataFrame, scenario: str):
    train = train.copy()
    test = test.copy()
    median = float(train["Pres."].median())
    if scenario == "faixa_estrita_treino":
        low, high = median - 0.50, median + 0.50
        train = train.loc[train["Pres."].between(low, high)].copy()
        test = test.loc[test["Pres."].between(low, high)].copy()
    else:
        low = high = None

    soil_min = float(train["Soil"].min())
    soil_max = float(train["Soil"].max())
    if soil_min == soil_max:
        raise ValueError("Indice de solo sem amplitude no treino")
    for part in (train, test):
        part["Soil_indice_0_1"] = (part["Soil"] - soil_min) / (soil_max - soil_min)
        part["Temp_C"] = part["Temp."]
        part["Pres_kPa"] = part["Pres."]
    return train, test, {
        "mediana_pressao_treino_kpa": median,
        "limite_pressao_inferior_kpa": low,
        "limite_pressao_superior_kpa": high,
        "solo_min_treino": soil_min,
        "solo_max_treino": soil_max,
        "leituras_treino_apos_filtro": len(train),
        "leituras_teste_apos_filtro": len(test),
        "teste_solo_fora_escala_0_1": int((~test["Soil_indice_0_1"].between(0, 1)).sum()),
    }


def summarize_predictions(predictions: pd.DataFrame, scenario: str, config: str):
    by_collection = predictions.groupby(["Coleta", "Vaso_id", "Classe"], as_index=False).agg(
        probabilidade_classe_1=("probabilidade_classe_1", "mean"),
        n_leituras=("Classe", "size"),
    )
    by_collection["predicao"] = (by_collection.probabilidade_classe_1 >= 0.5).astype(int)
    by_vessel = by_collection.groupby(["Vaso_id", "Classe"], as_index=False).agg(
        probabilidade_classe_1=("probabilidade_classe_1", "mean"),
        n_coletas=("Coleta", "size"),
    )
    by_vessel["predicao"] = (by_vessel.probabilidade_classe_1 >= 0.5).astype(int)
    if by_collection.Coleta.nunique() != 36 or by_vessel.Vaso_id.nunique() != 10:
        raise AssertionError("O filtro deixou alguma coleta ou vaso sem previsao")
    rows = [metrics(predictions, scenario, config, "leitura"),
            metrics(by_collection, scenario, config, "coleta"),
            metrics(by_vessel, scenario, config, "vaso")]
    return rows, by_collection, by_vessel


def paired_sign_flip(reading_by_vessel: pd.DataFrame, scenario: str) -> dict:
    """Compara acuracia por leitura dentro de cada um dos dez vasos."""
    pivot = reading_by_vessel.pivot(index="Vaso_id", columns="configuracao", values="acuracia")
    diff = (pivot.nove_entradas - pivot.somente_ambientais).to_numpy()
    observed = float(diff.mean())
    flips = np.asarray(list(itertools.product((-1, 1), repeat=len(diff))))
    permuted = (flips * diff).mean(axis=1)
    return {"cenario": scenario, "diferenca_media_acuracia_leitura_por_vaso": observed,
            "p_bilateral_inversao_sinal": float(np.mean(np.abs(permuted) >= abs(observed) - 1e-12))}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = pd.read_csv(SOURCE)
    data["Vaso_id"] = "classe_" + data.Classe.astype(str) + "_vaso_" + data.Vaso.astype(int).astype(str)
    required = ["Coleta", "Classe", "Vaso_id", "Soil", "Temp.", "Pres.", *MQ]
    if data[required].isna().any().any():
        raise ValueError("Dados ausentes nas colunas necessarias")
    if len(data) != 66081 or data.Coleta.nunique() != 36 or data.Vaso_id.nunique() != 10:
        raise AssertionError("Base antes da faixa estrita inesperada")
    if data.groupby("Coleta").Vaso_id.nunique().max() != 1:
        raise AssertionError("Coleta ligada a mais de um vaso")

    folds = list(LeaveOneGroupOut().split(data, data.Classe, data.Vaso_id))
    if len(folds) != 10:
        raise AssertionError("Esperadas dez rodadas por vaso")
    metadata, all_metrics, all_collections, all_vessels, all_reading_vessels = [], [], [], [], []
    for scenario in SCENARIOS:
        predictions_by_config = {name: [] for name in CONFIGS}
        for fold, (fit_idx, test_idx) in enumerate(folds, 1):
            train_raw = data.iloc[fit_idx]
            test_raw = data.iloc[test_idx]
            assert not (set(train_raw.Vaso_id) & set(test_raw.Vaso_id))
            train, test, details = prepare_fold(train_raw, test_raw, scenario)
            if train.empty or test.empty or train.Classe.nunique() != 2:
                raise AssertionError("Particao vazia ou treino com uma classe")
            metadata.append({"cenario": scenario, "fold": fold,
                             "vaso_teste": test_raw.Vaso_id.iloc[0],
                             "leituras_treino_antes_filtro": len(train_raw),
                             "leituras_teste_antes_filtro": len(test_raw), **details})
            for config, features in CONFIGS.items():
                fitted = ExtraTreesClassifier(
                    n_estimators=900, max_features="sqrt", min_samples_leaf=10,
                    bootstrap=False, random_state=42, n_jobs=6,
                ).fit(train.loc[:, features], train.Classe)
                prediction = test[["Coleta", "Vaso_id", "Classe", "Dia", "Tempo"]].copy()
                prediction["probabilidade_classe_1"] = fitted.predict_proba(test.loc[:, features])[:, 1]
                prediction["predicao"] = (prediction.probabilidade_classe_1 >= 0.5).astype(int)
                prediction["fold"] = fold
                predictions_by_config[config].append(prediction)
                print(f"{scenario}: vaso {fold}/10, {config}, teste {len(test)} leituras", flush=True)

        for config, parts in predictions_by_config.items():
            predictions = pd.concat(parts, ignore_index=True)
            if predictions.Coleta.nunique() != 36 or predictions.Vaso_id.nunique() != 10:
                raise AssertionError("Predicoes LOSO incompletas")
            rows, collections, vessels = summarize_predictions(predictions, scenario, config)
            all_metrics.extend(rows)
            collections.insert(0, "configuracao", config)
            collections.insert(0, "cenario", scenario)
            vessels.insert(0, "configuracao", config)
            vessels.insert(0, "cenario", scenario)
            all_collections.append(collections)
            all_vessels.append(vessels)
            reading_vessels = predictions.assign(
                acerto=(predictions.Classe == predictions.predicao).astype(float)
            ).groupby("Vaso_id", as_index=False).agg(
                acuracia=("acerto", "mean"), n_leituras=("acerto", "size")
            )
            reading_vessels.insert(0, "configuracao", config)
            reading_vessels.insert(0, "cenario", scenario)
            all_reading_vessels.append(reading_vessels)
            predictions.to_csv(OUT / f"predicoes_{scenario}_{config}.csv.gz", index=False,
                               compression="gzip")

    metadata_df = pd.DataFrame(metadata)
    metrics_df = pd.DataFrame(all_metrics)
    vessels_df = pd.concat(all_vessels, ignore_index=True)
    reading_vessels_df = pd.concat(all_reading_vessels, ignore_index=True)
    metadata_df.to_csv(OUT / "parametros_por_rodada.csv", index=False)
    metrics_df.to_csv(OUT / "metricas.csv", index=False)
    pd.concat(all_collections, ignore_index=True).to_csv(OUT / "coletas.csv", index=False)
    vessels_df.to_csv(OUT / "vasos.csv", index=False)
    reading_vessels_df.to_csv(OUT / "acuracia_leitura_por_vaso.csv", index=False)
    pd.DataFrame([paired_sign_flip(reading_vessels_df[reading_vessels_df.cenario == scenario], scenario)
                  for scenario in SCENARIOS]).to_csv(OUT / "comparacao_pareada_vasos.csv", index=False)
    (OUT / "metodo.json").write_text(json.dumps({
        "fonte": str(SOURCE.relative_to(ROOT)),
        "fonte_preprocessada": "Recorte central de 70% e filtro fixo de saltos de pressao ja aplicados por coleta",
        "validacao": "Dez rodadas LeaveOneGroupOut por vaso; parametros ajustados so nos nove vasos de treino",
        "solo": "Minimo e maximo do Soil bruto no treino apos eventual faixa estrita; teste pode ficar fora de [0,1]",
        "faixa_estrita": "Mediana de Pres. somente no treino antes da faixa +/- 0,50 kPa; aplicada a treino e teste",
        "sem_faixa_estrita": "Mantem filtro fixo de saltos; nao aplica banda por mediana",
        "modelo": "ExtraTreesClassifier 900 arvores, max_features=sqrt, min_samples_leaf=10, bootstrap=False, seed=42",
        "configuracoes": {key: list(value) for key, value in CONFIGS.items()},
        "limites": "Cinco MQ foram selecionados anteriormente nos mesmos dados; esse cenario continua exploratorio. Nao ha selecao aninhada de entradas.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
