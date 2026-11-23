"""KNN, MLP e ensemble ANN/KNN/SVM/LDA na comparacao LOSO filtrada.

Uso: .venv/Scripts/python.exe analise_combinatoria_sensores/comparar_knn_mlp_ensemble_filtrada.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from rodar_combinacoes import DATASET, ENV
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "validacao_sem_cinco_coletas" / "coletas_excluidas.csv"
OUTPUT = ROOT / "comparacao_knn_mlp_ensemble_filtrada"
MQ = ["MQ2", "MQ3", "MQ8", "MQ135", "MQ138"]
SCENARIOS = {"mq": MQ, "mq_ambiente": [*MQ, *ENV]}
SEED = 42


def models(calibration_folds: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, object]:
    return {
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=2),
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001,
                batch_size=2048, early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=12, max_iter=250, random_state=SEED,
            ),
        ),
        "SVM_linear": CalibratedClassifierCV(
            estimator=make_pipeline(
                StandardScaler(),
                LinearSVC(C=1.0, class_weight="balanced", dual=False,
                          max_iter=10000, random_state=SEED),
            ),
            method="sigmoid", cv=calibration_folds,
        ),
        "LDA": make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        ),
    }


def summary(y: pd.Series, predicted: pd.Series, model: str, scenario: str,
            n_collections: int, n_vessels: int) -> dict:
    return {
        "modelo": model,
        "cenario": scenario,
        "n_vasos": n_vessels,
        "n_coletas": n_collections,
        "n_leituras": len(y),
        "acertos": int((y.to_numpy() == predicted.to_numpy()).sum()),
        "acuracia": float(accuracy_score(y, predicted)),
        "acuracia_balanceada": float(balanced_accuracy_score(y, predicted)),
        "f1_macro": float(f1_score(y, predicted, average="macro")),
    }


def main() -> None:
    excluded = set(pd.read_csv(SOURCE).Coleta)
    data = pd.read_csv(DATASET)
    data = data.loc[~data.Coleta.isin(excluded)].copy()
    data["Vaso_id"] = vessel_id(data)
    required = ["Coleta", "Vaso_id", "Classe", "Tempo", *SCENARIOS["mq_ambiente"]]
    if data[required].isna().any().any():
        raise ValueError("A base contem valores ausentes nas colunas necessarias")
    if (len(data), data.Coleta.nunique(), data.Vaso_id.nunique()) != (55946, 31, 10):
        raise ValueError("A base nao corresponde as 31 coletas e 55.946 leituras")
    if data.groupby("Coleta").Classe.nunique().max() != 1:
        raise ValueError("Uma coleta contem classes diferentes")
    if data.groupby("Vaso_id").Classe.nunique().max() != 1:
        raise ValueError("Um vaso contem classes diferentes")

    OUTPUT.mkdir(exist_ok=True)
    results = []
    for scenario, features in SCENARIOS.items():
        predictions = []
        for fold, (train_idx, test_idx) in enumerate(
            LeaveOneGroupOut().split(data, data.Classe, data.Vaso_id), 1
        ):
            train, test = data.iloc[train_idx], data.iloc[test_idx]
            if set(train.Vaso_id) & set(test.Vaso_id):
                raise AssertionError("Vaso compartilhado entre treino e teste")
            part = test[["Coleta", "Vaso_id", "Classe", "Tempo"]].copy()
            calibration_folds = list(StratifiedGroupKFold(
                n_splits=3, shuffle=True, random_state=SEED
            ).split(train, train.Classe, train.Vaso_id))
            for _, calibration_idx in calibration_folds:
                if train.iloc[calibration_idx].Classe.nunique() != 2:
                    raise ValueError("Fold de calibracao sem as duas classes")
            for name, estimator in models(calibration_folds).items():
                print(f"{scenario} fold {fold}/10 {name}", flush=True)
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always", ConvergenceWarning)
                    estimator.fit(train[features], train.Classe)
                for warning in seen:
                    if issubclass(warning.category, ConvergenceWarning):
                        print(f"AVISO {scenario} fold {fold} {name}: {warning.message}", flush=True)
                probability = estimator.predict_proba(test[features])[:, 1]
                part[f"p1_{name}"] = probability
                part[name] = (probability >= 0.5).astype(np.int8)

            members = ["MLP", "KNN", "SVM_linear", "LDA"]
            part["p1_Ensemble_ANN_KNN_SVM_LDA"] = part[
                [f"p1_{name}" for name in members]
            ].mean(axis=1)
            part["Ensemble_ANN_KNN_SVM_LDA"] = (
                part["p1_Ensemble_ANN_KNN_SVM_LDA"] >= 0.5
            ).astype(np.int8)
            part["empate_voto_simples"] = part[members].sum(axis=1).eq(2)
            part.insert(0, "fold", fold)
            predictions.append(part)

        all_predictions = pd.concat(predictions).sort_values(["Coleta", "Tempo"])
        if len(all_predictions) != len(data) or not all_predictions.index.is_unique:
            raise AssertionError("Predicoes incompletas ou duplicadas")
        all_predictions.to_csv(OUTPUT / f"predicoes_{scenario}.csv", index=False)
        for name in ["KNN", "MLP", "SVM_linear", "LDA", "Ensemble_ANN_KNN_SVM_LDA"]:
            results.append(summary(all_predictions.Classe, all_predictions[name],
                                   name, scenario, 31, 10))
        pd.DataFrame(results).to_csv(OUTPUT / "metricas.csv", index=False)
        print(f"{scenario}: {int(all_predictions.empate_voto_simples.sum())} empates evitados pelo voto probabilistico", flush=True)

    method = {
        "dataset": str(DATASET),
        "coletas_excluidas": str(SOURCE),
        "cenario_entradas": SCENARIOS,
        "protocolo": "LOSO por vaso: 10 folds, 31 coletas, 55946 leituras; previsoes por leitura",
        "preprocessamento": "StandardScaler ajustado somente no treino de cada fold",
        "knn": "5 vizinhos, distancia euclidiana, pesos inversamente proporcionais a distancia",
        "mlp_ann": "MLP 64/32 ReLU, alpha=0.001, batch=2048, ate 250 epocas, early stopping 15% do treino, seed=42",
        "svm": "LinearSVC C=1, class_weight=balanced, dual=False, max_iter=10000; probabilidades por CalibratedClassifierCV sigmoid, 3 folds internos estratificados por vaso somente no treino LOSO",
        "lda": "LinearDiscriminantAnalysis solver=lsqr, shrinkage=auto",
        "ensemble": "Media nao ponderada das probabilidades da classe 1 de MLP, KNN, SVM linear calibrada e LDA; limiar 0.5",
        "limite": "Exclusoes e escolha das entradas exploratorias nos mesmos dez vasos; sem estimativa independente de desempenho externo",
    }
    (OUTPUT / "metodo.json").write_text(
        json.dumps(method, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
