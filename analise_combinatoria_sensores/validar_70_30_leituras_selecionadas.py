"""70/30 estratificado por leitura crua para tres configuracoes selecionadas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rodar_combinacoes import DATASET, ENV, MQ, model
from validacao_por_vaso import vessel_id


OUT = Path(__file__).resolve().parent / "validacao_70_30_por_leitura"
SEED = 42
CONFIGS = {
    "melhor_mq_loso_cru": ["MQ2", "MQ7", "MQ8", "MQ135", "MQ138"],
    "cinco_mq_pareado": ["MQ2", "MQ3", "MQ8", "MQ135", "MQ138"],
    "cinco_mq_ambiente": ["MQ2", "MQ3", "MQ8", "MQ135", "MQ138", *ENV],
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATASET)
    required = ["Coleta", "Vaso", "Classe", "Tempo", *MQ, *ENV]
    if df[required].isna().any().any():
        raise ValueError("Valores ausentes nas entradas ou identificadores")
    df["Vaso_id"] = vessel_id(df)
    df["linha_original"] = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.30, stratify=df.Classe,
        random_state=SEED,
    )
    if set(train_idx) & set(test_idx):
        raise AssertionError("Leitura presente no treino e no teste")
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    split = df[["linha_original", "Coleta", "Vaso_id", "Classe"]].copy()
    split["conjunto"] = "Treino"
    split.loc[test_idx, "conjunto"] = "Teste"
    split.to_csv(OUT / "split_por_leitura.csv", index=False)

    results, predictions = [], []
    for name, features in CONFIGS.items():
        fitted = model().fit(train[features], train.Classe)
        probability = fitted.predict_proba(test[features])[:, 1]
        predicted = (probability >= 0.5).astype(int)
        correct = test.Classe.to_numpy() == predicted
        results.append({
            "configuracao": name,
            "sensores": " | ".join(features),
            "n_sensores": len(features),
            "n_leituras_treino": len(train),
            "n_leituras_teste": len(test),
            "n_acertos_teste": int(correct.sum()),
            "acuracia_normal": float(correct.mean()),
            "n_coletas_treino": train.Coleta.nunique(),
            "n_coletas_teste": test.Coleta.nunique(),
            "n_vasos_treino": train.Vaso_id.nunique(),
            "n_vasos_teste": test.Vaso_id.nunique(),
        })
        frame = test[["linha_original", "Coleta", "Vaso_id", "Tempo", "Classe"]].copy()
        frame.insert(0, "configuracao", name)
        frame["probabilidade_classe_1"] = probability
        frame["predicao"] = predicted
        predictions.append(frame)
        print(name, f"{int(correct.sum())}/{len(test)}", flush=True)

    pd.DataFrame(results).to_csv(OUT / "metricas_por_leitura.csv", index=False)
    predicted_all = pd.concat(predictions, ignore_index=True)
    predicted_all.sort_values(["configuracao", "linha_original"]).to_csv(
        OUT / "predicoes_teste_por_leitura.csv", index=False,
    )
    by_collection = (
        predicted_all.assign(correta=lambda x: x.Classe.eq(x.predicao).astype(int))
        .groupby(["configuracao", "Coleta", "Vaso_id", "Classe"], as_index=False)
        .agg(leituras_teste=("correta", "size"), acertos=("correta", "sum"))
    )
    by_collection["acuracia_dentro_da_coleta"] = (
        by_collection.acertos / by_collection.leituras_teste
    )
    by_collection.to_csv(OUT / "acertos_dentro_de_cada_coleta.csv", index=False)
    overlap_collection = sorted(set(train.Coleta) & set(test.Coleta))
    overlap_vessel = sorted(set(train.Vaso_id) & set(test.Vaso_id))
    (OUT / "metodo.json").write_text(json.dumps({
        "dataset": str(DATASET),
        "configuracoes": CONFIGS,
        "divisao": "train_test_split por indice de leitura crua, test_size=0.30, stratify=Classe, random_state=42",
        "metrica": "Acuracia normal por leitura crua, limiar de probabilidade 0.5",
        "modelo": "ExtraTreesClassifier, 900 arvores, max_features=sqrt, min_samples_leaf=10, seed=42",
        "coletas_compartilhadas_treino_teste": len(overlap_collection),
        "vasos_compartilhados_treino_teste": len(overlap_vessel),
        "interpretacao": "Avalia novas leituras de coletas e vasos ja vistos no treino; LOSO por vaso avalia vasos nao vistos",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
