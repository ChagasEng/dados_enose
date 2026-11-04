"""Ablacao de entradas ambientais e importancia por permutacao no teste completo.

Usa o mesmo split por Coleta e as mesmas leituras para Extra Trees e X-TCN.
Os MQ sao brutos nos dois cenarios, pois MQ corrigidos ja incorporam ambiente.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_10_2026_xtcn_solo_medio" / "scripts"))
from comparar_rf_xtcn_medias_ambiente import (  # noqa: E402
    BATCH, EPOCHS, SEED, STRIDE, WINDOW, XTCN, make_windows,
)
from comparar_xtcn_tabela22_por_leitura import windows_to_readings  # noqa: E402

DATASET = ROOT / "06_07_melhor_modelo" / "dados" / "dataset_melhor_modelo_sensores_corrigidos.csv"
OUT = Path(__file__).resolve().parent
MQ = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
ENV = ["Soil_indice_0_1", "Temp_C", "Pres_kPa"]
SCENARIOS = {"mq": MQ, "mq_ambiente": MQ + ENV}
REPEATS = 3


def metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "f1_macro": float(f1_score(y, p, average="macro")),
    }


def fit_xtcn(train, features):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    scaler = StandardScaler().fit(train[features].to_numpy(dtype=np.float32))
    xtr, ytr, _ = make_windows(train, features, scaler)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
        batch_size=BATCH, shuffle=True,
    )
    device = torch.device("cpu")
    model = XTCN(len(features)).to(device)
    counts = np.bincount(ytr, minlength=2)
    weights = torch.tensor(len(ytr) / (2 * counts), dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"X-TCN {len(features)} entradas: epoca {epoch + 1}/{EPOCHS}", flush=True)
    model.eval()
    return model, scaler


def predict_xtcn(test, features, model, scaler):
    xte, yte, groups = make_windows(test, features, scaler)
    with torch.no_grad():
        probs = np.concatenate([
            torch.softmax(model(torch.from_numpy(xte[i:i+BATCH])), dim=1)[:, 1].numpy()
            for i in range(0, len(xte), BATCH)
        ])
    windows = pd.DataFrame({
        "Coleta": groups, "classe_real": yte,
        "probabilidade_saudavel": probs,
    })
    readings = windows_to_readings(test, windows)
    # windows_to_readings ordena por Coleta/Tempo; a metrica usa a ordem
    # original do DataFrame de teste. Reindexar evita comparar rotulos de
    # leituras diferentes.
    return readings.set_index("index").loc[test.index, "classe_prevista_xtcn"].to_numpy()


def score_model(model_name, model, test, features, scaler=None):
    if model_name == "Extra Trees":
        pred = model.predict(test[features])
    else:
        pred = predict_xtcn(test, features, model, scaler)
    return metrics(test["Classe"].to_numpy(), pred)


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATASET)
    required = ["Coleta", "Tempo", "Classe", "Conjunto", *MQ, *ENV]
    if df[required].isna().any().any():
        raise ValueError("Ha valores ausentes nas entradas ou nas colunas de split")
    train = df.loc[df.Conjunto == "Treino"].copy()
    test = df.loc[df.Conjunto == "Teste"].copy()
    if set(train.Coleta) & set(test.Coleta):
        raise AssertionError("Uma Coleta aparece no treino e no teste")
    # Faixas definidas somente para avaliar o teste; ambos os modelos treinam
    # com todas as coletas do treino.
    dry = test.loc[test.Soil_indice_0_1 <= 0.4].copy()
    wet = test.loc[test.Soil_indice_0_1 > 0.4].copy()
    if (len(dry), len(wet)) != (16692, 5423):
        raise AssertionError("O recorte nao coincide com o da Tabela 2")
    if set(dry.Coleta) & set(wet.Coleta):
        raise AssertionError("Uma Coleta cruza as faixas seco/umido")

    summary, importance = [], []
    for scenario, features in SCENARIOS.items():
        forest = ExtraTreesClassifier(
            n_estimators=900, random_state=SEED, n_jobs=-1,
            max_features="sqrt", min_samples_leaf=10, bootstrap=False,
        )
        forest.fit(train[features], train.Classe)
        xtcn, scaler = fit_xtcn(train, features)
        for name, model, fitted_scaler in (
            ("Extra Trees", forest, None), ("X-TCN", xtcn, scaler),
        ):
            baseline = {}
            for group, frame in (("seco", dry), ("umido", wet), ("total", test)):
                result = score_model(name, model, frame, features, fitted_scaler)
                summary.append({
                    "cenario": scenario, "modelo": name, "faixa": group,
                    "n_leituras": len(frame), "n_coletas": frame.Coleta.nunique(),
                    **result,
                })
                if group == "total":
                    baseline = result
            print(scenario, name, "total", baseline, flush=True)
            for feature in features:
                drops = []
                for repeat in range(REPEATS):
                    rng = np.random.default_rng(SEED + repeat)
                    altered = test.copy()
                    altered[feature] = rng.permutation(altered[feature].to_numpy())
                    score = score_model(name, model, altered, features, fitted_scaler)
                    drops.append(100 * (baseline["balanced_accuracy"] - score["balanced_accuracy"]))
                importance.append({
                    "cenario": scenario, "modelo": name, "faixa": "total",
                    "variavel": feature, "queda_balanceada_pp": float(np.mean(drops)),
                    "desvio_pp": float(np.std(drops, ddof=1)),
                    "n_repeticoes": REPEATS,
                })
                print(scenario, name, feature, round(float(np.mean(drops)), 3), flush=True)
    pd.DataFrame(summary).to_csv(OUT / "desempenho_com_sem_ambiente.csv", index=False)
    pd.DataFrame(importance).to_csv(OUT / "importancia_permutacao.csv", index=False)
    (OUT / "metodo.json").write_text(json.dumps({
        "dataset": str(DATASET.relative_to(ROOT)),
        "split": "Coluna Conjunto preexistente; 24 coletas treino, 12 teste",
        "entradas": SCENARIOS,
        "mq": "brutos em ambos os cenarios; sem compensacao ambiental",
        "floresta": "ExtraTreesClassifier, 900 arvores, max_features=sqrt, min_samples_leaf=10, seed=42",
        "xtcn": f"60 epocas, janela={WINDOW}, passo={STRIDE}, lote={BATCH}, seed={SEED}",
        "metrica": "por leitura; X-TCN usa media das probabilidades de janelas que cobrem a leitura",
        "importancia": "queda da acuracia balanceada no teste completo ao permutar uma coluna de leituras; mesmo algoritmo nos dois modelos",
        "repeticoes": REPEATS,
        "limiar_solo": "Soil_indice_0_1 <= 0.4; indice operacional, nao teor calibrado de agua",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
