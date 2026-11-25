"""Combina previsoes LOSO salvas de ET, RF, TCN, MLP e k-NN."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut

from rodar_combinacoes import DATASET
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parent
EXCLUDED = ROOT / "validacao_sem_cinco_coletas" / "coletas_excluidas.csv"
ET_DIR = ROOT / "comparacao_auc_modelos_filtrada"
RF_TCN_DIR = ROOT / "validacao_sem_cinco_coletas"
MLP_KNN_DIR = ROOT / "comparacao_knn_mlp_ensemble_filtrada"
OUT = ROOT / "ensemble_cinco_modelos_filtrada"
KEY = ["fold", "Coleta", "Vaso_id", "Classe", "Tempo"]
MEMBERS = ["Extra Trees", "Random Forest", "TCN", "MLP", "k-NN"]


def filtered_data() -> pd.DataFrame:
    excluded = set(pd.read_csv(EXCLUDED).Coleta)
    data = pd.read_csv(DATASET)
    data = data.loc[~data.Coleta.isin(excluded)].copy()
    data["Vaso_id"] = vessel_id(data)
    if (len(data), data.Coleta.nunique(), data.Vaso_id.nunique()) != (55946, 31, 10):
        raise AssertionError("Base filtrada diferente de 55.946 leituras, 31 coletas e 10 vasos")
    if data.duplicated(["Coleta", "Tempo"]).any():
        raise AssertionError("Tempo duplicado dentro de uma coleta")
    return data


def et_probabilities(data: pd.DataFrame, scenario: str) -> pd.DataFrame:
    parts = []
    for fold, (_, test_idx) in enumerate(
        LeaveOneGroupOut().split(data, data.Classe, data.Vaso_id), 1
    ):
        part = data.iloc[test_idx][["Coleta", "Vaso_id", "Classe", "Tempo"]].copy()
        part.insert(0, "fold", fold)
        parts.append(part)
    ordered = pd.concat(parts, ignore_index=True)
    saved = pd.read_csv(ET_DIR / f"predicoes_et_{scenario}.csv")
    if len(saved) != len(ordered):
        raise AssertionError("Previsoes Extra Trees incompletas")
    for column in ["fold", "Coleta", "Vaso_id", "Classe"]:
        if not np.array_equal(saved[column].to_numpy(), ordered[column].to_numpy()):
            raise AssertionError(f"Ordem das previsoes Extra Trees difere em {column}")
    ordered["p1_Extra Trees"] = saved.probabilidade_classe_1.to_numpy()
    return ordered


def saved_probabilities(path: Path, column: str, name: str) -> pd.DataFrame:
    saved = pd.read_csv(path)
    if saved.duplicated(KEY).any():
        raise AssertionError(f"Chave duplicada em {name}")
    return saved[[*KEY, column]].rename(columns={column: f"p1_{name}"})


def predictions(data: pd.DataFrame, scenario: str) -> pd.DataFrame:
    result = et_probabilities(data, scenario)
    inputs = [
        saved_probabilities(RF_TCN_DIR / f"predicoes_randomforest_{scenario}.csv",
                            "probabilidade_classe_1", "Random Forest"),
        saved_probabilities(RF_TCN_DIR / f"predicoes_tcn_{scenario}.csv",
                            "probabilidade_classe_1", "TCN"),
        saved_probabilities(MLP_KNN_DIR / f"predicoes_{scenario}.csv", "p1_MLP", "MLP"),
        saved_probabilities(MLP_KNN_DIR / f"predicoes_{scenario}.csv", "p1_KNN", "k-NN"),
    ]
    for source in inputs:
        result = result.merge(source, on=KEY, how="left", validate="one_to_one")
    columns = [f"p1_{name}" for name in MEMBERS]
    if len(result) != 55946 or result[columns].isna().any().any():
        raise AssertionError("Probabilidades desalinhadas ou incompletas")
    if ((result[columns] < 0) | (result[columns] > 1)).any().any():
        raise AssertionError("Probabilidade fora do intervalo [0, 1]")
    result["p1_ensemble"] = result[columns].mean(axis=1)
    result["predicao_ensemble"] = (result.p1_ensemble >= 0.5).astype(np.int8)
    return result.sort_values(["Coleta", "Tempo"]).reset_index(drop=True)


def metrics(result: pd.DataFrame, scenario: str) -> dict:
    y = result.Classe
    predicted = result.predicao_ensemble
    return {
        "cenario": scenario,
        "modelo": "Ensemble (ET + RF + TCN + MLP + k-NN)",
        "n_vasos": result.Vaso_id.nunique(),
        "n_coletas": result.Coleta.nunique(),
        "n_leituras": len(result),
        "acertos": int((y == predicted).sum()),
        "acuracia": accuracy_score(y, predicted),
        "acuracia_balanceada": balanced_accuracy_score(y, predicted),
        "recall_classe_0": recall_score(y, predicted, pos_label=0),
        "f1_macro": f1_score(y, predicted, average="macro"),
    }


def main() -> None:
    data = filtered_data()
    OUT.mkdir(exist_ok=True)
    rows = []
    for scenario in ["mq", "mq_ambiente"]:
        result = predictions(data, scenario)
        result.to_csv(OUT / f"predicoes_{scenario}.csv", index=False)
        rows.append(metrics(result, scenario))
    pd.DataFrame(rows).to_csv(OUT / "metricas.csv", index=False)
    method = {
        "base": "55.946 leituras, 31 coletas, 10 vasos, apos exclusao de cinco coletas",
        "validacao": "LOSO por vaso; probabilidades fora do treino de cada vaso",
        "componentes": MEMBERS,
        "regra": "Media aritmetica nao ponderada das cinco probabilidades da classe 1; classe 1 se media >= 0,5",
        "cenarios": {"mq": "cinco MQ", "mq_ambiente": "cinco MQ + indice de solo, temperatura e pressao"},
        "fontes": {
            "Extra Trees": str(ET_DIR / "predicoes_et_{cenario}.csv"),
            "Random Forest e TCN": str(RF_TCN_DIR / "predicoes_{modelo}_{cenario}.csv"),
            "MLP e k-NN": str(MLP_KNN_DIR / "predicoes_{cenario}.csv"),
        },
        "limite": "Exclusoes, selecao das entradas e composicao do ensemble definidas apos inspecao de resultados nos mesmos dez vasos; resultado exploratorio",
    }
    (OUT / "metodo.json").write_text(json.dumps(method, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
