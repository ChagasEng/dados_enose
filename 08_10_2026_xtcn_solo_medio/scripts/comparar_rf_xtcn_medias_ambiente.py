"""Compara RF e X-TCN com medias ambientais robustas por coleta.

As leituras MQ permanecem intactas. Soil (sensor de umidade do solo) e Temp.
sao substituidos pela media dos valores dentro das cercas de Tukey de cada
coleta, calculadas separadamente para cada variavel.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "06_07/4_polimento_inicial_modelagem/datasets_limpos/antes_dia_20_pressao_filtrada_estrito.csv"
OUTPUT = ROOT / "08_10_2026_xtcn_solo_medio/resultados_medias_robustas"
MQ = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
SCENARIOS = {
    "mq": MQ,
    "mq_solo_bruto": MQ + ["Soil"],
    "mq_solo_media_sem_filtro": MQ + ["Soil_media_sem_filtro"],
    "mq_solo_medio": MQ + ["Soil_medio"],
    "mq_temperatura_media": MQ + ["Temp_media"],
    "mq_solo_temperatura_medios": MQ + ["Soil_medio", "Temp_media"],
}
SEED, WINDOW, STRIDE, EPOCHS, BATCH = 42, 128, 64, 60, 32


def mean_without_outliers(values: pd.Series) -> tuple[float, int, int]:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        raise ValueError("Coleta sem leituras ambientais validas")
    q1, q3 = valid.quantile([0.25, 0.75])
    iqr = q3 - q1
    kept = valid.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return float(valid[kept].mean()), int(len(valid)), int((~kept).sum())


def add_group_means(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for coleta, block in df.groupby("Coleta", sort=False):
        soil, ns, os = mean_without_outliers(block["Soil"])
        temp, nt, ot = mean_without_outliers(block["Temp."])
        records.append({"Coleta": coleta, "Soil_medio": soil,
                        "Soil_media_sem_filtro": float(block["Soil"].mean()),
                        "Temp_media": temp,
                        "n_solo_validos": ns, "outliers_solo": os,
                        "n_temperatura_validos": nt, "outliers_temperatura": ot})
    means = pd.DataFrame(records)
    return df.merge(means[["Coleta", "Soil_medio", "Soil_media_sem_filtro", "Temp_media"]],
                    on="Coleta", validate="many_to_one"), means


def split_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_groups = set()
    for _, block in df.groupby("Classe"):
        groups = pd.Series(block.Coleta.unique()).sample(frac=1, random_state=SEED).tolist()
        train_groups.update(groups[:int(len(groups) * 0.70)])
    return df[df.Coleta.isin(train_groups)].copy(), df[~df.Coleta.isin(train_groups)].copy()


def make_windows(df: pd.DataFrame, features: list[str], scaler: StandardScaler):
    xs, ys, groups = [], [], []
    for coleta, block in df.sort_values(["Coleta", "Tempo"]).groupby("Coleta", sort=False):
        values = scaler.transform(block[features].to_numpy(dtype=np.float32))
        starts = list(range(0, max(1, len(values) - WINDOW + 1), STRIDE))
        if len(values) >= WINDOW and starts[-1] != len(values) - WINDOW:
            starts.append(len(values) - WINDOW)
        for start in starts:
            window = values[start:start + WINDOW]
            if len(window) < WINDOW:
                window = np.pad(window, ((0, WINDOW - len(window)), (0, 0)), mode="edge")
            xs.append(window.T)
            ys.append(int(block.Classe.iloc[0]))
            groups.append(coleta)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys), np.asarray(groups)


class TemporalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int):
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(nn.Conv1d(c_in, c_out, 3, padding=padding, dilation=dilation), nn.ReLU(),
                                 nn.Conv1d(c_out, c_out, 3, padding=padding, dilation=dilation), nn.ReLU())
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return torch.relu(self.net(x)[..., :x.shape[-1]] + self.skip(x))


class XTCN(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.blocks = nn.Sequential(TemporalBlock(n_features, 24, 1),
                                    TemporalBlock(24, 24, 2), TemporalBlock(24, 32, 4))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(.20), nn.Linear(32, 2))

    def forward(self, x):
        return self.head(self.blocks(x))


def metrics(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}


def evaluate_rf(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    model = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1,
                                   class_weight="balanced")
    model.fit(train[features], train.Classe)
    probabilities = model.predict_proba(test[features])[:, 1]
    readings = metrics(test.Classe, probabilities >= .5)
    by_group = pd.DataFrame({"Coleta": test.Coleta, "Classe": test.Classe,
                             "prob": probabilities}).groupby("Coleta").agg(
                                 Classe=("Classe", "first"), prob=("prob", "mean"))
    return readings, metrics(by_group.Classe, by_group.prob >= .5)


def evaluate_xtcn(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    # Usa a configuracao de 60 epocas do script X-TCN original.
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
    windows = metrics(yte, probs >= .5)
    by_group = pd.DataFrame({"Coleta": gte, "Classe": yte, "prob": probs}).groupby("Coleta").agg(
        Classe=("Classe", "first"), prob=("prob", "mean"))
    return windows, metrics(by_group.Classe, by_group.prob >= .5), len(ytr), len(yte), str(device)


def main():
    df = pd.read_csv(DATASET)
    required = ["Coleta", "Classe", "Tempo", "Soil", "Temp.", *MQ]
    df = df.dropna(subset=required).copy()
    df["Classe"] = df.Classe.astype(int)
    if df.groupby("Coleta").Classe.nunique().max() != 1:
        raise ValueError("Ha coletas com mais de uma classe")
    df, means = add_group_means(df)
    train, test = split_groups(df)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    means.to_csv(OUTPUT / "medias_ambientais_por_coleta.csv", index=False)
    pd.DataFrame({"Coleta": means.Coleta,
                  "conjunto": np.where(means.Coleta.isin(train.Coleta), "treino", "teste")}).to_csv(
                      OUTPUT / "divisao_por_coleta.csv", index=False)
    rows = []
    for scenario, features in SCENARIOS.items():
        print(f"Treinando {scenario}...", flush=True)
        rf_readings, rf_groups = evaluate_rf(train, test, features)
        xt_windows, xt_groups, n_train_windows, n_test_windows, device = evaluate_xtcn(train, test, features)
        for model_name, level, result in [("RF", "leitura", rf_readings),
                                          ("RF", "coleta", rf_groups),
                                          ("X-TCN", "janela", xt_windows),
                                          ("X-TCN", "coleta", xt_groups)]:
            rows.append({"cenario": scenario, "modelo": model_name, "unidade": level,
                         "n_coletas_treino": train.Coleta.nunique(), "n_coletas_teste": test.Coleta.nunique(),
                         "n_leituras_treino": len(train), "n_leituras_teste": len(test),
                         "n_janelas_treino": n_train_windows if model_name == "X-TCN" else None,
                         "n_janelas_teste": n_test_windows if model_name == "X-TCN" else None,
                         **result})
    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT / "comparacao_rf_xtcn_medias_ambientais.csv", index=False)
    settings = {"dataset": str(DATASET.relative_to(ROOT)), "features_por_cenario": SCENARIOS,
                "outliers": "Cercas de Tukey (Q1 - 1,5*IQR; Q3 + 1,5*IQR) por coleta e variavel; remove apenas da media",
                "split": "70/30 por coleta em cada classe, seed 42", "epocas_xtcn": EPOCHS,
                "janela_xtcn": WINDOW, "stride_xtcn": STRIDE, "dispositivo": device}
    (OUTPUT / "metodo.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result_df[["cenario", "modelo", "unidade", "accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
