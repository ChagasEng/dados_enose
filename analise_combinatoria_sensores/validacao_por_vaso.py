"""Valida configuracoes fixas por coleta, com split 70/30 e LOSO por vaso.

Uso: .venv/Scripts/python.exe analise_combinatoria_sensores/validacao_por_vaso.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split

from rodar_combinacoes import DATASET, ENV, MQ, model


OUT = Path(__file__).resolve().parent / "validacao_por_vaso"
SEED = 42
CONFIGS = {
    "candidato_mq_ambiente": ["MQ2", "MQ135", "MQ138", *ENV],
    "candidato_so_mq": ["MQ2", "MQ3", "MQ8", "MQ135", "MQ138"],
    "seis_mq": list(MQ),
    "seis_mq_ambiente": [*MQ, *ENV],
    "mq7_ambiente": ["MQ7", *ENV],
}


def vessel_id(frame: pd.DataFrame) -> pd.Series:
    # O numero Vaso=1 ocorre nas duas classes; sao unidades biologicas distintas.
    return "classe_" + frame.Classe.astype(str) + "_vaso_" + frame.Vaso.astype(int).astype(str)


def predict_collections(train: pd.DataFrame, test: pd.DataFrame,
                        features: list[str]) -> pd.DataFrame:
    fitted = model().fit(train[features], train.Classe)
    readings = test[["Coleta", "Classe"]].copy()
    readings["probabilidade_classe_1"] = fitted.predict_proba(test[features])[:, 1]
    predictions = readings.groupby("Coleta", sort=True).agg(
        Classe=("Classe", "first"),
        probabilidade_classe_1=("probabilidade_classe_1", "mean"),
        n_leituras=("Classe", "size"),
    ).reset_index()
    predictions["predicao"] = (predictions.probabilidade_classe_1 >= 0.5).astype(int)
    details = test[["Coleta", "Vaso_id", "Dia"]].drop_duplicates("Coleta")
    return predictions.merge(details, on="Coleta", validate="one_to_one")


def summarize(predictions: pd.DataFrame, protocol: str, name: str,
              features: list[str], n_train_vessels: int,
              n_test_vessels: int, n_folds: int) -> dict:
    y, yhat = predictions.Classe, predictions.predicao
    return {
        "protocolo": protocol, "configuracao": name,
        "sensores": " | ".join(features),
        "n_sensores": len(features), "n_folds": n_folds,
        "n_vasos_treino_por_fold": n_train_vessels,
        "n_vasos_teste_por_fold": n_test_vessels,
        "n_coletas_avaliadas": len(predictions),
        "n_coletas_corretas": int((y == yhat).sum()),
        "coleta_accuracy": float(accuracy_score(y, yhat)),
        "coleta_balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "coleta_f1_macro": float(f1_score(y, yhat, average="macro")),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATASET)
    required = ["Coleta", "Vaso", "Dia", "Classe", "Conjunto", *MQ, *ENV]
    if df[required].isna().any().any():
        raise ValueError("Valores ausentes no dataset")
    if df.groupby("Coleta").Classe.nunique().max() != 1:
        raise ValueError("Rotulos inconsistentes em uma coleta")
    df["Vaso_id"] = vessel_id(df)
    if df.groupby("Coleta").Vaso_id.nunique().max() != 1:
        raise ValueError("Mais de um vaso em uma coleta")
    vessels = df.groupby("Vaso_id", sort=True).Classe.first()
    if len(vessels) != 10 or vessels.value_counts().to_dict() != {0: 5, 1: 5}:
        raise ValueError("Esperados cinco vasos em cada classe")

    old_train = df.loc[df.Conjunto.eq("Treino")]
    old_test = df.loc[df.Conjunto.eq("Teste")]
    if set(old_train.Coleta) & set(old_test.Coleta):
        raise AssertionError("Coleta compartilhada no split historico")
    overlap = sorted(set(old_train.Vaso_id) & set(old_test.Vaso_id))

    train_vessels, test_vessels = train_test_split(
        vessels.index.to_numpy(), test_size=0.3,
        stratify=vessels.to_numpy(), random_state=SEED,
    )
    train_vessels, test_vessels = set(train_vessels), set(test_vessels)
    group_train = df.loc[df.Vaso_id.isin(train_vessels)]
    group_test = df.loc[df.Vaso_id.isin(test_vessels)]
    assert not (set(group_train.Vaso_id) & set(group_test.Vaso_id))

    logo = LeaveOneGroupOut()
    all_predictions, summaries = [], []
    for name, features in CONFIGS.items():
        for protocol, train, test in (
            ("70_30_coleta_historico", old_train, old_test),
            ("70_30_vaso", group_train, group_test),
        ):
            predictions = predict_collections(train, test, features)
            predictions.insert(0, "protocolo", protocol)
            predictions.insert(1, "configuracao", name)
            all_predictions.append(predictions)
            summaries.append(summarize(
                predictions, protocol, name, features,
                train.Vaso_id.nunique(), test.Vaso_id.nunique(), 1,
            ))
            print(protocol, name, summaries[-1]["coleta_balanced_accuracy"], flush=True)

        fold_predictions = []
        for fold, (fit_idx, test_idx) in enumerate(logo.split(df, df.Classe, df.Vaso_id), 1):
            fit, test = df.iloc[fit_idx], df.iloc[test_idx]
            if set(fit.Vaso_id) & set(test.Vaso_id):
                raise AssertionError("Vaso compartilhado no LOSO")
            predictions = predict_collections(fit, test, features)
            predictions.insert(0, "fold", fold)
            fold_predictions.append(predictions)
            print("LOSO", name, fold, test.Vaso_id.iloc[0], flush=True)
        loso_predictions = pd.concat(fold_predictions, ignore_index=True)
        if loso_predictions.Coleta.nunique() != df.Coleta.nunique():
            raise AssertionError("LOSO nao previu todas as coletas")
        loso_predictions.insert(0, "protocolo", "loso_vaso")
        loso_predictions.insert(1, "configuracao", name)
        all_predictions.append(loso_predictions)
        summaries.append(summarize(
            loso_predictions, "loso_vaso", name, features, 9, 1, 10,
        ))
        print("LOSO final", name, summaries[-1]["coleta_balanced_accuracy"], flush=True)

    pred_df = pd.concat(all_predictions, ignore_index=True)
    pred_df.to_csv(OUT / "predicoes_por_coleta.csv", index=False)
    result_df = pd.DataFrame(summaries)
    result_df.to_csv(OUT / "metricas_por_coleta.csv", index=False)
    per_vessel = (
        pred_df.loc[pred_df.protocolo.eq("loso_vaso")]
        .assign(correta=lambda x: x.Classe.eq(x.predicao).astype(int))
        .groupby(["configuracao", "Vaso_id", "Classe"], as_index=False)
        .agg(coletas=("Coleta", "size"), acertos=("correta", "sum"))
    )
    per_vessel.to_csv(OUT / "acertos_loso_por_vaso.csv", index=False)
    (OUT / "metodo.json").write_text(json.dumps({
        "dataset": str(DATASET),
        "modelo": "Mesmo ExtraTrees 900 arvores da analise combinatoria",
        "configuracoes": CONFIGS,
        "unidade_de_avaliacao": "Coleta; media das probabilidades das leituras, limiar 0.5",
        "vaso_id": "Classe + numero Vaso; numero 1 aparece nas duas classes e representa vasos distintos",
        "70_30_coleta_historico": "Coluna Conjunto; 24 coletas treino, 12 teste; pode repetir vaso nos dois lados",
        "vasos_em_ambos_lados_historico": overlap,
        "70_30_vaso": "train_test_split estratificado por classe do vaso; 7 vasos treino, 3 teste; seed 42",
        "vasos_teste_70_30": sorted(test_vessels),
        "loso": "10 folds LeaveOneGroupOut por vaso; 9 vasos treino, 1 vaso teste; metricas calculadas nas 36 predicoes de coleta reunidas",
        "observacao": "Cada fold LOSO testa uma unica classe; balanced accuracy somente apos reunir as previsoes das 10 rodadas",
        "limite": "Conjuntos de sensores escolhidos anteriormente nos mesmos dados; esta validacao e exploratoria, nao confirmacao externa",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
