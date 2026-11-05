"""Sensibilidade do LOSO: manter os dez vasos e retirar coletas com ruido extremo.

Uso: .venv/Scripts/python.exe analise_combinatoria_sensores/avaliar_loso_qualidade_por_coleta.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from rodar_combinacoes import DATASET, ENV, model
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "validacao_qualidade_por_coleta"
AUDIT = ROOT / "auditoria_ruido_todas_coletas.csv"
ORIGINAL = (Path(__file__).resolve().parent /
            "busca_mq_ambiente_loso_leituras" / "predicoes_vencedor_cru.csv")
FEATURES = ["MQ2", "MQ3", "MQ8", "MQ135", "MQ138", *ENV]
MAX_MULTISENSOR_FRACTION = 0.05


def summary(frame: pd.DataFrame, scenario: str) -> dict:
    correct = frame.Classe.eq(frame.predicao)
    return {
        "cenario": scenario,
        "n_vasos": int(frame.Vaso_id.nunique()),
        "n_coletas": int(frame.Coleta.nunique()),
        "n_leituras": int(len(frame)),
        "acertos": int(correct.sum()),
        "erros": int((~correct).sum()),
        "acuracia_por_leitura": float(correct.mean()),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    audit = pd.read_csv(AUDIT)
    audit["fracao_multissensor"] = audit.leituras_multissensor / audit.n_leituras
    bad = set(audit.loc[audit.fracao_multissensor >= MAX_MULTISENSOR_FRACTION, "Coleta"])
    if len(bad) != 2:
        raise AssertionError(f"Esperadas duas coletas com ruido extremo, obtidas: {bad}")

    df = pd.read_csv(DATASET)
    df["Vaso_id"] = vessel_id(df)
    if df.Coleta.nunique() != 36 or df.Vaso_id.nunique() != 10:
        raise AssertionError("Base original inesperada")
    clean = df.loc[~df.Coleta.isin(bad)].copy()
    if clean.Coleta.nunique() != 34 or clean.Vaso_id.nunique() != 10:
        raise AssertionError("A filtragem eliminou algum vaso ou numero incorreto de coletas")

    pieces = []
    for fold, (fit_idx, test_idx) in enumerate(
        LeaveOneGroupOut().split(clean, clean.Classe, clean.Vaso_id), 1
    ):
        train, test = clean.iloc[fit_idx], clean.iloc[test_idx]
        if set(train.Vaso_id) & set(test.Vaso_id):
            raise AssertionError("Vaso compartilhado entre treino e teste")
        fitted = model().fit(train[FEATURES], train.Classe)
        p = fitted.predict_proba(test[FEATURES])[:, 1]
        predicted = test[["Coleta", "Vaso_id", "Dia", "Tempo", "Classe"]].copy()
        predicted["probabilidade_classe_1"] = p
        predicted["predicao"] = (p >= 0.5).astype(int)
        predicted.insert(0, "fold", fold)
        pieces.append(predicted)
        print(f"fold {fold}/10: {test.Vaso_id.iloc[0]}", flush=True)
    new = pd.concat(pieces, ignore_index=True)
    if len(new) != len(clean) or new.Coleta.nunique() != 34:
        raise AssertionError("Predicoes incompletas")

    original = pd.read_csv(ORIGINAL)
    old_same = original.loc[~original.Coleta.isin(bad)].copy()
    if len(old_same) != len(new):
        raise AssertionError("Previsoes antigas nao correspondem as novas leituras")
    if not old_same.groupby("Coleta").size().sort_index().equals(
        new.groupby("Coleta").size().sort_index()
    ):
        raise AssertionError("Composicao por coleta difere")

    results = pd.DataFrame([
        summary(original, "loso_original_36_coletas"),
        summary(old_same, "loso_original_mesmas_34_coletas"),
        summary(new, "loso_retreinado_34_coletas_10_vasos"),
    ])
    results.to_csv(OUT / "comparacao_metricas.csv", index=False)
    new.sort_values(["Coleta", "Tempo"]).to_csv(OUT / "predicoes_loso.csv", index=False)
    audit.loc[audit.Coleta.isin(bad)].to_csv(OUT / "coletas_sinalizadas.csv", index=False)
    (OUT / "metodo.json").write_text(json.dumps({
        "objetivo": "Sensibilidade a ruido por coleta, mantendo todos os dez vasos",
        "dataset": str(DATASET),
        "modelo": "ExtraTreesClassifier, 900 arvores, max_features=sqrt, min_samples_leaf=10, seed=42",
        "entradas": FEATURES,
        "regra_exploratoria": "Sinalizar coleta com >=5% das leituras com salto em dois ou mais MQ, segundo auditoria de eventos p99.5",
        "fracao_minima": MAX_MULTISENSOR_FRACTION,
        "coletas_sinalizadas": sorted(bad),
        "validacao": "LOSO por vaso, 10 folds, nove vasos treino e um teste; uma predicao por leitura",
        "comparacao": "Previsoes originais restritas as mesmas 34 coletas versus novo treino com essas 34 coletas",
        "limite": "Regra de qualidade definida apos inspecao dos dados; analise exploratoria, nao desempenho externo",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
