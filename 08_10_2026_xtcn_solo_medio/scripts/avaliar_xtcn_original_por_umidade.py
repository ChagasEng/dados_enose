"""Treina MQ + Soil e avalia o mesmo modelo por solo seco/umido.

A faixa e definida pela media de Soil de cada coleta. A X-TCN continua
recebendo Soil leitura a leitura, sem o rotulo da faixa como entrada.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from comparar_rf_xtcn_medias_ambiente import (
    BATCH, DATASET, EPOCHS, MQ, ROOT, SEED, STRIDE, WINDOW, XTCN,
    make_windows, split_groups,
)


OUTPUT = ROOT / "08_10_2026_xtcn_solo_medio/resultados_estratificados_mq_solo"
FEATURES = MQ + ["Soil"]
SOIL_CUTOFF = 0.4


def metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    if frame.empty:
        return {"accuracy": None, "balanced_accuracy": None, "f1_macro": None}
    truth = frame["classe_real"].to_numpy()
    pred = frame["classe_prevista"].to_numpy()
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)) if len(np.unique(truth)) == 2 else None,
        "f1_macro": float(f1_score(truth, pred, average="macro")),
    }


def train_and_predict(train: pd.DataFrame, test: pd.DataFrame,
                      features: list[str] | None = None) -> pd.DataFrame:
    if features is None:
        features = FEATURES
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    scaler = StandardScaler().fit(train[features].to_numpy(dtype=np.float32))
    xtr, ytr, _ = make_windows(train, features, scaler)
    xte, yte, gte = make_windows(test, features, scaler)
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XTCN(len(features)).to(device)
    counts = np.bincount(ytr, minlength=2)
    weights = torch.tensor(len(ytr) / (2 * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        probs = np.concatenate([
            torch.softmax(model(torch.from_numpy(xte[i:i+BATCH]).to(device)), dim=1)[:, 1].cpu().numpy()
            for i in range(0, len(xte), BATCH)
        ])
    return pd.DataFrame({"Coleta": gte, "classe_real": yte,
                         "probabilidade_saudavel": probs,
                         "classe_prevista": (probs >= .5).astype(int)})


def main() -> None:
    df = pd.read_csv(DATASET).dropna(subset=["Coleta", "Tempo", "Classe", *FEATURES]).copy()
    df["Classe"] = df["Classe"].astype(int)
    if df.groupby("Coleta")["Classe"].nunique().max() != 1:
        raise ValueError("Coleta com mais de uma classe")

    soil_min, soil_max = float(df.Soil.min()), float(df.Soil.max())
    if soil_max == soil_min:
        raise ValueError("Soil nao varia; nao e possivel separar faixas")
    collections = df.groupby("Coleta", as_index=False).agg(
        classe_real=("Classe", "first"), soil_medio=("Soil", "mean"),
        n_leituras=("Soil", "size"))
    collections["soil_indice_medio"] = (collections.soil_medio - soil_min) / (soil_max - soil_min)
    collections["faixa"] = np.where(collections.soil_indice_medio <= SOIL_CUTOFF, "seco", "umido")

    train, test = split_groups(df)
    collections["conjunto"] = np.where(collections.Coleta.isin(train.Coleta), "treino", "teste")
    test_collections = collections[collections.conjunto == "teste"].copy()
    predictions = train_and_predict(train, test)
    predictions = predictions.merge(
        test_collections[["Coleta", "soil_medio", "soil_indice_medio", "faixa"]],
        on="Coleta", validate="many_to_one")
    by_collection = predictions.groupby("Coleta", as_index=False).agg(
        classe_real=("classe_real", "first"), probabilidade_saudavel=("probabilidade_saudavel", "mean"),
        n_janelas=("classe_real", "size"), soil_medio=("soil_medio", "first"),
        soil_indice_medio=("soil_indice_medio", "first"), faixa=("faixa", "first"))
    by_collection["classe_prevista"] = (by_collection.probabilidade_saudavel >= .5).astype(int)

    overall_windows = metrics(predictions)
    overall_collections = metrics(by_collection)
    overall = pd.DataFrame([{
        "experimento": "X-TCN MQ + Soil leitura a leitura", "origem": "nova_execucao",
        "n_coletas_treino": train.Coleta.nunique(), "n_coletas_teste": len(test_collections),
        "n_janelas_teste": len(predictions),
        "accuracy_janela": overall_windows["accuracy"],
        "balanced_accuracy_janela": overall_windows["balanced_accuracy"],
        "f1_macro_janela": overall_windows["f1_macro"],
        "accuracy_coleta": overall_collections["accuracy"],
        "balanced_accuracy_coleta": overall_collections["balanced_accuracy"],
        "f1_macro_coleta": overall_collections["f1_macro"],
    }])
    strata = []
    for faixa in ("seco", "umido"):
        subset_windows = predictions[predictions.faixa == faixa]
        subset_collections = by_collection[by_collection.faixa == faixa]
        window_metrics = metrics(subset_windows)
        collection_metrics = metrics(subset_collections)
        strata.append({
            "faixa": faixa, "n_coletas": len(subset_collections), "n_janelas": len(subset_windows),
            "coletas_classe_0": int((subset_collections.classe_real == 0).sum()),
            "coletas_classe_1": int((subset_collections.classe_real == 1).sum()),
            "media_soil_das_coletas": float(subset_collections.soil_medio.mean()) if len(subset_collections) else None,
            "media_indice_soil_das_coletas": float(subset_collections.soil_indice_medio.mean()) if len(subset_collections) else None,
            "accuracy_janela": window_metrics["accuracy"],
            "balanced_accuracy_janela": window_metrics["balanced_accuracy"],
            "accuracy_coleta": collection_metrics["accuracy"],
            "balanced_accuracy_coleta": collection_metrics["balanced_accuracy"],
        })
    strata = pd.DataFrame(strata)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUTPUT / "tabela_geral.csv", index=False)
    strata.to_csv(OUTPUT / "tabela_seco_umido.csv", index=False)
    collections.to_csv(OUTPUT / "medias_e_faixas_por_coleta.csv", index=False)
    predictions.to_csv(OUTPUT / "predicoes_por_janela.csv", index=False)
    by_collection.to_csv(OUTPUT / "predicoes_por_coleta.csv", index=False)
    method = {
        "dataset": str(DATASET.relative_to(ROOT)), "features_xtcn": FEATURES,
        "modelo": "mesma arquitetura X-TCN de comparar_rf_xtcn_medias_ambiente.py",
        "split": "70/30 por Coleta dentro de cada Classe, seed 42", "epocas": EPOCHS,
        "janela": WINDOW, "stride": STRIDE,
        "faixa": "media de Soil da coleta convertida em indice min-max global; seco <= 0.4, umido > 0.4",
        "soil_min": soil_min, "soil_max": soil_max,
        "media_por_faixa": "media nao ponderada das medias de Soil das coletas de teste em cada faixa",
        "observacao": "faixa usada apenas na avaliacao; nao fornecida ao modelo",
    }
    (OUTPUT / "metodo.json").write_text(json.dumps(method, ensure_ascii=False, indent=2), encoding="utf-8")
    print(overall.to_string(index=False))
    print(strata.to_string(index=False))


if __name__ == "__main__":
    main()
