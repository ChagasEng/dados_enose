"""LOSO e 70/30 por leitura apos excluir tres vasos com mais erros."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, train_test_split

from rodar_combinacoes import DATASET, ENV, model
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validacao_sem_vasos_criticos"
OLD_PRED = ROOT / "busca_mq_ambiente_loso_leituras" / "predicoes_vencedor_cru.csv"
FEATURES = ["MQ2", "MQ3", "MQ8", "MQ135", "MQ138", *ENV]
EXCLUDED = {"classe_1_vaso_1", "classe_1_vaso_2", "classe_0_vaso_9"}


def predict(train: pd.DataFrame, test: pd.DataFrame, protocol: str) -> pd.DataFrame:
    fitted = model().fit(train[FEATURES], train.Classe)
    probabilities = fitted.predict_proba(test[FEATURES])[:, 1]
    predicted = test[["Coleta", "Vaso_id", "Tempo", "Classe"]].copy()
    predicted["probabilidade_classe_1"] = probabilities
    predicted["predicao"] = (probabilities >= 0.5).astype(int)
    predicted.insert(0, "protocolo", protocol)
    return predicted


def summary(predictions: pd.DataFrame, label: str) -> dict:
    correct = predictions.Classe.eq(predictions.predicao)
    return {
        "cenario": label,
        "n_vasos": predictions.Vaso_id.nunique(),
        "n_coletas": predictions.Coleta.nunique(),
        "n_leituras_teste": len(predictions),
        "n_acertos": int(correct.sum()),
        "n_erros": int((~correct).sum()),
        "acuracia_normal": float(correct.mean()),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATASET)
    if df[["Coleta", "Vaso", "Classe", "Tempo", *FEATURES]].isna().any().any():
        raise ValueError("Valores ausentes")
    df["Vaso_id"] = vessel_id(df)
    if not EXCLUDED.issubset(set(df.Vaso_id)):
        raise ValueError("Vasos a excluir nao encontrados")
    reduced = df.loc[~df.Vaso_id.isin(EXCLUDED)].copy()
    if reduced.Vaso_id.nunique() != 7:
        raise AssertionError("Esperados sete vasos restantes")

    loso_parts = []
    for fold, (fit_idx, test_idx) in enumerate(
        LeaveOneGroupOut().split(reduced, reduced.Classe, reduced.Vaso_id), 1
    ):
        train, test = reduced.iloc[fit_idx], reduced.iloc[test_idx]
        assert not (set(train.Vaso_id) & set(test.Vaso_id))
        predicted = predict(train, test, "loso_7_vasos")
        predicted.insert(1, "fold", fold)
        loso_parts.append(predicted)
        print("LOSO", fold, test.Vaso_id.iloc[0], flush=True)
    loso = pd.concat(loso_parts, ignore_index=True)
    if len(loso) != len(reduced) or loso.Coleta.nunique() != reduced.Coleta.nunique():
        raise AssertionError("LOSO incompleto")

    old = pd.read_csv(OLD_PRED)
    old_remaining = old.loc[~old.Vaso_id.isin(EXCLUDED)].copy()
    if len(old_remaining) != len(reduced):
        raise AssertionError("Conjunto de teste antigo nao corresponde aos vasos restantes")

    train_idx, test_idx = train_test_split(
        np.arange(len(reduced)), test_size=0.30,
        stratify=reduced.Classe, random_state=42,
    )
    train, test = reduced.iloc[train_idx], reduced.iloc[test_idx]
    read_split = predict(train, test, "70_30_por_leitura_7_vasos")

    results = pd.DataFrame([
        summary(old, "loso_10_vasos_original"),
        summary(old_remaining, "loso_10_vasos_so_7_testes_restantes"),
        summary(loso, "loso_7_vasos_retreinado"),
        summary(read_split, "70_30_por_leitura_7_vasos"),
    ])
    results.to_csv(OUT / "comparacao_metricas.csv", index=False)
    loso.sort_values(["Coleta", "Tempo"]).to_csv(
        OUT / "predicoes_loso_7_vasos.csv", index=False,
    )
    read_split.sort_values(["Coleta", "Tempo"]).to_csv(
        OUT / "predicoes_70_30_por_leitura_7_vasos.csv", index=False,
    )
    by_vessel = (
        loso.assign(correta=lambda x: x.Classe.eq(x.predicao).astype(int))
        .groupby(["Vaso_id", "Classe"], as_index=False)
        .agg(leituras=("correta", "size"), acertos=("correta", "sum"))
    )
    by_vessel["erros"] = by_vessel.leituras - by_vessel.acertos
    by_vessel["acuracia_normal"] = by_vessel.acertos / by_vessel.leituras
    by_vessel.to_csv(OUT / "acertos_loso_por_vaso.csv", index=False)
    (OUT / "metodo.json").write_text(json.dumps({
        "dataset": str(DATASET),
        "modelo": "ExtraTreesClassifier, 900 arvores, max_features=sqrt, min_samples_leaf=10, seed=42",
        "sensores": FEATURES,
        "vasos_excluidos": sorted(EXCLUDED),
        "criterio_exclusao": "Tres vasos citados na analise anterior de erros do vencedor com ambiente",
        "loso_reduzido": "Sete folds; treina em seis vasos e testa todas as leituras do setimo",
        "comparacao_pareada": "Previsoes LOSO originais do modelo treinado com nove vasos, filtradas para as mesmas leituras dos sete vasos restantes",
        "70_30_reduzido": "train_test_split por leitura, test_size=0.30, stratify=Classe, random_state=42; novo split no conjunto reduzido",
        "metrica": "Acuracia normal por leitura crua",
        "limite": "Excluir vasos selecionados por seus erros muda a populacao de teste; aumento da acuracia nao comprova melhor generalizacao",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
