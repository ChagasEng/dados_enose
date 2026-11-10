"""LOSO de Random Forest e TCN apos retirar quatro coletas com maioria errada.

O filtro e exploratorio: as coletas foram escolhidas depois de observar erros
do Extra Trees original. Todos os dez vasos permanecem no LOSO.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rodar_combinacoes import DATASET, ENV
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_10_2026_xtcn_solo_medio" / "scripts"))
from comparar_rf_xtcn_medias_ambiente import XTCN, make_windows, WINDOW, STRIDE, EPOCHS, BATCH  # noqa: E402

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
OUT = Path(__file__).resolve().parent / "validacao_sem_quatro_coletas_erradas"
BAD = (
    "dia 12 - Soja Saudavel Vaso 2",
    "dia 19 - Soja Heterodera Vaso 9",
    "dia 19 - Soja Saudavel Vaso 2",
    "dia 19 - Soja Saudavel Vaso 1",
)
EXTRA_NOISY = "dia 19 - Soja Heterodera Vaso 1"
MQ = ["MQ2", "MQ3", "MQ8", "MQ135", "MQ138"]
SCENARIOS = {"mq": MQ, "mq_ambiente": [*MQ, *ENV]}
SEED = 42


def frame_summary(frame: pd.DataFrame, model_name: str, scenario: str) -> dict:
    correct = frame.Classe.eq(frame.predicao)
    return {
        "modelo": model_name,
        "cenario": scenario,
        "unidade": "leitura",
        "n_vasos": int(frame.Vaso_id.nunique()),
        "n_coletas": int(frame.Coleta.nunique()),
        "n_leituras": int(len(frame)),
        "acertos": int(correct.sum()),
        "erros": int((~correct).sum()),
        "acuracia": float(accuracy_score(frame.Classe, frame.predicao)),
        "acuracia_balanceada": float(balanced_accuracy_score(frame.Classe, frame.predicao)),
    }


def predict_rf(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    fitted = RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1, class_weight="balanced"
    ).fit(train[features], train.Classe)
    return fitted.predict_proba(test[features])[:, 1]


def predict_tcn(train: pd.DataFrame, test: pd.DataFrame,
                features: list[str]) -> tuple[np.ndarray, int]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    scaler = StandardScaler().fit(train[features].to_numpy(np.float32))
    xtr, ytr, _ = make_windows(train, features, scaler)
    xte, _, test_groups = make_windows(test, features, scaler)
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True)
    network = XTCN(len(features))
    counts = np.bincount(ytr, minlength=2)
    weights = torch.tensor(len(ytr) / (2 * counts), dtype=torch.float32)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    for _ in range(EPOCHS):
        network.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(network(xb), yb).backward()
            optimizer.step()
    network.eval()
    with torch.no_grad():
        window_prob = np.concatenate([
            torch.softmax(network(torch.from_numpy(xte[i:i + BATCH])), dim=1)[:, 1].numpy()
            for i in range(0, len(xte), BATCH)
        ])
    predictions = []
    offset = 0
    for name, block in test.sort_values(["Coleta", "Tempo"]).groupby("Coleta", sort=False):
        n = len(block)
        starts = list(range(0, max(1, n - WINDOW + 1), STRIDE))
        if n >= WINDOW and starts[-1] != n - WINDOW:
            starts.append(n - WINDOW)
        p = window_prob[offset:offset + len(starts)]
        groups = test_groups[offset:offset + len(starts)]
        if len(p) != len(starts) or not np.all(groups == name):
            raise AssertionError(f"Janelas nao correspondem a {name}")
        offset += len(starts)
        total, coverage = np.zeros(n), np.zeros(n, dtype=int)
        for start, probability in zip(starts, p):
            stop = min(n, start + WINDOW)
            total[start:stop] += probability
            coverage[start:stop] += 1
        if not np.all(coverage):
            raise AssertionError(f"Leituras sem janela em {name}")
        predictions.append(pd.Series(total / coverage, index=block.index))
    if offset != len(window_prob):
        raise AssertionError("Sobram previsoes por janela")
    return pd.concat(predictions).reindex(test.index).to_numpy(), len(window_prob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--somar-ruido", action="store_true",
                        help="Tambem retirar a coleta inoculada ruidosa do dia 19")
    args = parser.parse_args()
    excluded = (*BAD, EXTRA_NOISY) if args.somar_ruido else BAD
    out = (OUT.parent / "validacao_sem_cinco_coletas"
           if args.somar_ruido else OUT)
    out.mkdir(exist_ok=True)
    raw = pd.read_csv(DATASET)
    raw["Vaso_id"] = vessel_id(raw)
    if not set(excluded).issubset(set(raw.Coleta)):
        raise ValueError("Falta alguma coleta indicada para exclusao")
    manifest = (raw.loc[raw.Coleta.isin(excluded)]
                .groupby(["Coleta", "Classe", "Vaso_id"], as_index=False)
                .size().rename(columns={"size": "n_leituras"}))
    manifest["motivo"] = manifest.Coleta.map(
        lambda name: "ruido_multissensor" if name == EXTRA_NOISY
        else ("erro_concentrado_loso_original;ruido_multissensor"
              if name == "dia 19 - Soja Saudavel Vaso 1"
              else "erro_concentrado_loso_original")
    )
    manifest.to_csv(out / "coletas_excluidas.csv", index=False)
    df = raw.loc[~raw.Coleta.isin(excluded)].copy()
    expected_rows, expected_collections = (55946, 31) if args.somar_ruido else (57207, 32)
    if len(df) != expected_rows or df.Coleta.nunique() != expected_collections or df.Vaso_id.nunique() != 10:
        raise AssertionError("Conjunto filtrado inesperado")
    if df[["Coleta", "Vaso_id", "Classe", "Tempo", *SCENARIOS["mq_ambiente"]]].isna().any().any():
        raise ValueError("Valores ausentes")

    summary_rows = []
    for scenario, features in SCENARIOS.items():
        for model_name in ("RandomForest", "TCN"):
            path = out / f"predicoes_{model_name.lower()}_{scenario}.csv"
            if path.exists():
                all_pred = pd.read_csv(path)
                if len(all_pred) != len(df) or all_pred.Coleta.nunique() != expected_collections:
                    raise ValueError(f"Checkpoint incompleto: {path}")
            else:
                parts = []
                for fold, (fit_idx, test_idx) in enumerate(
                    LeaveOneGroupOut().split(df, df.Classe, df.Vaso_id), 1
                ):
                    train, test = df.iloc[fit_idx], df.iloc[test_idx]
                    if set(train.Vaso_id) & set(test.Vaso_id):
                        raise AssertionError("Vaso compartilhado")
                    if model_name == "RandomForest":
                        prob = predict_rf(train, test, features)
                        n_windows = None
                    else:
                        prob, n_windows = predict_tcn(train, test, features)
                    part = test[["Coleta", "Vaso_id", "Dia", "Tempo", "Classe"]].copy()
                    part["probabilidade_classe_1"] = prob
                    part["predicao"] = (prob >= .5).astype(int)
                    part.insert(0, "fold", fold)
                    part["n_janelas_teste"] = n_windows
                    parts.append(part)
                    print(f"{scenario} {model_name} fold {fold}/10: {test.Vaso_id.iloc[0]}", flush=True)
                all_pred = pd.concat(parts, ignore_index=True)
                all_pred.sort_values(["Coleta", "Tempo"]).to_csv(path, index=False)
            summary_rows.append(frame_summary(all_pred, model_name, scenario))
            pd.DataFrame(summary_rows).to_csv(out / "metricas_rf_tcn.csv", index=False)
    (out / "metodo_rf_tcn.json").write_text(json.dumps({
        "dataset": str(DATASET),
        "coletas_excluidas": excluded,
        "motivo_exploratorio": (
            "Quatro coletas com maioria de leituras incorretas no Extra Trees original; "
            "mais uma coleta ruidosa na auditoria"
            if args.somar_ruido else
            "Quatro coletas com maioria de leituras incorretas no Extra Trees original"
        ),
        "entradas": SCENARIOS,
        "validacao": f"LOSO por vaso, 10 folds, {expected_collections} coletas; uma predicao por leitura",
        "rf": "RandomForestClassifier 300 arvores, class_weight=balanced, seed=42",
        "tcn": f"Rede convolucional temporal do projeto, {EPOCHS} epocas, janela {WINDOW}, passo {STRIDE}, StandardScaler ajustado no treino de cada fold; media das probabilidades das janelas que cobrem cada leitura",
        "limite": "Selecao das coletas por erros do proprio modelo; resultados condicionais e exploratorios",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
