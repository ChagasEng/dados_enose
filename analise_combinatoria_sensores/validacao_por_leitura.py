"""Valida leituras cruas e medias de 4, com splits por coleta/vaso e LOSO.

Todas as previsoes sao por leitura ou bloco de quatro leituras. A separacao
entre treino e teste continua agrupada por vaso no LOSO e no 70/30 por vaso.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, train_test_split

from rodar_combinacoes import DATASET, ENV, MQ, model
from validacao_por_vaso import CONFIGS as PREVIOUS_CONFIGS, vessel_id


OUT = Path(__file__).resolve().parent / "validacao_por_leitura"
SEED = 42
CONFIGS = {
    "melhor_mq4_loso": ["MQ2", "MQ3", "MQ135", "MQ138"],
    **PREVIOUS_CONFIGS,
}


def make_mean4(raw: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in raw.groupby("Coleta", sort=False):
        group = group.sort_values("Tempo", kind="stable")
        usable = len(group) // 4 * 4
        trimmed = group.iloc[:usable].copy()
        trimmed["bloco"] = np.arange(usable) // 4
        averaged = trimmed.groupby("bloco", sort=False).agg({
            "Coleta": "first", "Classe": "first", "Vaso_id": "first",
            "Dia": "first", "Tempo": "mean",
            **{feature: "mean" for feature in [*MQ, *ENV]},
        }).reset_index()
        pieces.append(averaged)
    return pd.concat(pieces, ignore_index=True)


def predict_samples(train: pd.DataFrame, test: pd.DataFrame,
                    features: list[str], protocol: str,
                    representation: str, name: str) -> pd.DataFrame:
    fitted = model().fit(train[features], train.Classe)
    result = test[["Coleta", "Vaso_id", "Dia", "Tempo", "Classe"]].copy()
    result["probabilidade_classe_1"] = fitted.predict_proba(test[features])[:, 1]
    result["predicao"] = (result.probabilidade_classe_1 >= 0.5).astype(int)
    result.insert(0, "representacao", representation)
    result.insert(1, "protocolo", protocol)
    result.insert(2, "configuracao", name)
    return result


def summarize(predictions: pd.DataFrame, features: list[str], n_folds: int,
              n_vessels_train: int, n_vessels_test: int) -> dict:
    correct = predictions.Classe.eq(predictions.predicao)
    return {
        "representacao": predictions.representacao.iloc[0],
        "protocolo": predictions.protocolo.iloc[0],
        "configuracao": predictions.configuracao.iloc[0],
        "sensores": " | ".join(features),
        "n_sensores": len(features),
        "n_folds": n_folds,
        "n_vasos_treino_por_fold": n_vessels_train,
        "n_vasos_teste_por_fold": n_vessels_test,
        "n_amostras_teste": len(predictions),
        "n_acertos": int(correct.sum()),
        "acuracia_normal": float(correct.mean()),
        "n_coletas_teste": predictions.Coleta.nunique(),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(DATASET)
    required = ["Coleta", "Vaso", "Dia", "Tempo", "Classe", "Conjunto", *MQ, *ENV]
    if raw[required].isna().any().any():
        raise ValueError("Valores ausentes nas entradas ou identificadores")
    raw["Vaso_id"] = vessel_id(raw)
    if raw.Vaso_id.nunique() != 10 or raw.Coleta.nunique() != 36:
        raise ValueError("Esperados 10 vasos e 36 coletas")
    mean4 = make_mean4(raw)
    if len(mean4) != sum(raw.groupby("Coleta").size() // 4):
        raise AssertionError("Numero de blocos incorreto")
    vessels = raw.groupby("Vaso_id", sort=True).Classe.first()
    train_vessels, test_vessels = train_test_split(
        vessels.index.to_numpy(), test_size=0.3,
        stratify=vessels.to_numpy(), random_state=SEED,
    )
    test_vessels = set(test_vessels)
    all_predictions, metrics = [], []

    for representation, df in (("cru", raw), ("media_4", mean4)):
        old_train = df.loc[df.Conjunto.eq("Treino")] if "Conjunto" in df else None
        old_test = df.loc[df.Conjunto.eq("Teste")] if "Conjunto" in df else None
        # O atributo Conjunto e mantido nos blocos de 4 a partir da coleta.
        if old_train is None:
            split_lookup = raw.groupby("Coleta").Conjunto.first()
            df = df.copy()
            df["Conjunto"] = df.Coleta.map(split_lookup)
            old_train = df.loc[df.Conjunto.eq("Treino")]
            old_test = df.loc[df.Conjunto.eq("Teste")]
        group_train = df.loc[~df.Vaso_id.isin(test_vessels)]
        group_test = df.loc[df.Vaso_id.isin(test_vessels)]
        assert not (set(group_train.Vaso_id) & set(group_test.Vaso_id))

        for name, features in CONFIGS.items():
            for protocol, train, test in (
                ("70_30_coleta_historico", old_train, old_test),
                ("70_30_vaso", group_train, group_test),
            ):
                predictions = predict_samples(train, test, features, protocol,
                                              representation, name)
                all_predictions.append(predictions)
                metrics.append(summarize(
                    predictions, features, 1,
                    train.Vaso_id.nunique(), test.Vaso_id.nunique(),
                ))
                print(representation, protocol, name,
                      f"{metrics[-1]['n_acertos']}/{metrics[-1]['n_amostras_teste']}", flush=True)

            fold_predictions = []
            logo = LeaveOneGroupOut()
            for fold, (fit_idx, test_idx) in enumerate(
                logo.split(df, df.Classe, df.Vaso_id), 1
            ):
                train, test = df.iloc[fit_idx], df.iloc[test_idx]
                assert not (set(train.Vaso_id) & set(test.Vaso_id))
                predicted = predict_samples(train, test, features,
                                            "loso_vaso", representation, name)
                predicted.insert(3, "fold", fold)
                fold_predictions.append(predicted)
            predictions = pd.concat(fold_predictions, ignore_index=True)
            if len(predictions) != len(df) or predictions.Coleta.nunique() != 36:
                raise AssertionError("LOSO nao cobriu todas as amostras e coletas")
            all_predictions.append(predictions)
            metrics.append(summarize(predictions, features, 10, 9, 1))
            print(representation, "LOSO", name,
                  f"{metrics[-1]['n_acertos']}/{metrics[-1]['n_amostras_teste']}", flush=True)

    summary = pd.DataFrame(metrics)
    summary.to_csv(OUT / "metricas_por_amostra.csv", index=False)
    predictions = pd.concat(all_predictions, ignore_index=True)
    # Uma linha por amostra para inspecionar erros dentro de cada arquivo.
    predictions.to_csv(OUT / "predicoes_por_amostra.csv", index=False)
    per_collection = (
        predictions.assign(correta=lambda x: x.Classe.eq(x.predicao).astype(int))
        .groupby(["representacao", "protocolo", "configuracao", "Coleta", "Vaso_id", "Classe"], as_index=False)
        .agg(n_amostras=("correta", "size"), n_acertos=("correta", "sum"))
    )
    per_collection["acuracia_dentro_da_coleta"] = per_collection.n_acertos / per_collection.n_amostras
    per_collection.to_csv(OUT / "acertos_dentro_de_cada_coleta.csv", index=False)
    (OUT / "metodo.json").write_text(json.dumps({
        "dataset": str(DATASET),
        "configuracoes": CONFIGS,
        "total_leituras_cruas": len(raw),
        "total_blocos_media_4": len(mean4),
        "leituras_descartadas_media_4": len(raw) - 4 * len(mean4),
        "media_4": "Quatro leituras consecutivas da mesma coleta, ordenadas por Tempo; medias dos atributos; ultimo bloco incompleto descartado",
        "unidade_de_avaliacao": "Leitura crua ou bloco medio de 4, conforme a representacao",
        "unidade_de_separacao_loso": "Vaso biologico inteiro (10 folds); todas as coletas e amostras do vaso ficam fora do treino",
        "unidade_de_separacao_70_30_vaso": "Sete vasos treino, tres teste; semente 42",
        "split_historico": "Por coleta, 24 treino e 12 teste; oito vasos repetidos nos dois lados",
        "metrica": "Acuracia normal = amostras corretas / amostras avaliadas",
        "modelo": "ExtraTreesClassifier, 900 arvores, max_features=sqrt, min_samples_leaf=10, seed=42",
        "limite": "Leituras dentro da mesma coleta e coletas do mesmo vaso sao correlacionadas; quantidade de leituras nao equivale a quantidade de unidades biologicas independentes",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
