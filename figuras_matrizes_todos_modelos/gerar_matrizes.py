"""Gera uma figura em ingles com todas as matrizes comparaveis do projeto.

Uso: .venv/Scripts/python.exe figuras_matrizes_todos_modelos/gerar_matrizes.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent
LOSO = ROOT / "analise_combinatoria_sensores"
NEW = LOSO / "comparacao_knn_mlp_ensemble_filtrada"
HISTORICAL_ET = ROOT / "06_07_melhor_modelo" / "analise_seco_molhado"
HISTORICAL_TCN = (ROOT / "08_10_2026_xtcn_solo_medio"
                  / "resultados_tabela22_xtcn_mesmas_features"
                  / "predicoes_por_leitura.csv")
MODELS = [
    ("Extra Trees", "ET", None, 53015, 54783),
    ("Random Forest", "RF", None, 47172, 53385),
    ("TCN", "TCN", None, 49352, 49789),
    ("k-NN", "NEW", "KNN", 49767, 53183),
    ("MLP (ANN)", "NEW", "MLP", 50118, 53746),
    ("SVM", "NEW", "SVM_linear", 37015, 42679),
    ("LDA", "NEW", "LDA", 37465, 42968),
    ("ANN + k-NN + SVM + LDA", "NEW", "Ensemble_ANN_KNN_SVM_LDA", 49867, 51677),
]


def loso_matrix(scenario: str, model: tuple) -> np.ndarray:
    _, source, column, expected_mq, expected_env = model
    expected = expected_mq if scenario == "mq" else expected_env
    if source == "ET":
        path = LOSO / "comparacao_auc_modelos_filtrada" / f"predicoes_et_{scenario}.csv"
        frame = pd.read_csv(path, usecols=["Classe", "probabilidade_classe_1"])
        predicted = (frame.probabilidade_classe_1 >= 0.5).astype(int)
    elif source in {"RF", "TCN"}:
        name = "randomforest" if source == "RF" else "tcn"
        path = LOSO / "validacao_sem_cinco_coletas" / f"predicoes_{name}_{scenario}.csv"
        frame = pd.read_csv(path, usecols=["Classe", "predicao"])
        predicted = frame.predicao
    else:
        path = NEW / f"predicoes_{scenario}.csv"
        frame = pd.read_csv(path, usecols=["Classe", column])
        predicted = frame[column]
    matrix = confusion_matrix(frame.Classe, predicted, labels=[0, 1])
    if matrix.sum() != 55946 or np.trace(matrix) != expected:
        raise AssertionError(f"Contagens divergentes em {path}: {matrix.tolist()}")
    return matrix


def historical_matrices() -> list[tuple[str, np.ndarray]]:
    predictions = pd.read_csv(
        HISTORICAL_TCN,
        usecols=["Classe", "faixa", "classe_prevista_xtcn"],
    )
    result = []
    for condition, file_name, title, n in [
        ("seco", "seco", "TCN · Dry soil", 16692),
        ("umido", "molhado", "TCN · Wet soil", 5423),
    ]:
        subset = predictions.loc[predictions.faixa.eq(condition)]
        tcn = confusion_matrix(
            subset.Classe, subset.classe_prevista_xtcn, labels=[0, 1]
        )
        et = pd.read_csv(
            HISTORICAL_ET / f"matriz_confusao_{file_name}.csv", index_col=0
        ).to_numpy(dtype=int)
        if tcn.sum() != n or et.sum() != n or not np.array_equal(
            tcn.sum(axis=1), et.sum(axis=1)
        ):
            raise AssertionError(f"Matrizes historicas divergentes: {condition}")
        result.extend([(title, tcn), (f"Extra Trees · {'Dry' if condition == 'seco' else 'Wet'} soil", et)])
    return result


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    records = []
    for scenario in ("mq", "mq_ambiente"):
        for model in MODELS:
            title = model[0]
            matrix = loso_matrix(scenario, model)
            records.extend(matrix_records("LOSO by pot", scenario, title, matrix))

    for index, (title, matrix) in enumerate(historical_matrices()):
        records.extend(matrix_records("Historical split by collection", "dry" if index < 2 else "wet", title, matrix))
    pd.DataFrame(records).to_csv(OUTPUT / "confusion_matrix_counts.csv", index=False)
    print(f"Saved counts for {len(records) // 4} matrices to {OUTPUT}")


def matrix_records(protocol: str, scenario: str, model: str,
                   matrix: np.ndarray) -> list[dict]:
    rows = []
    for actual in (0, 1):
        for predicted in (0, 1):
            rows.append({
                "protocol": protocol,
                "scenario": scenario,
                "model": model,
                "actual_class": actual,
                "predicted_class": predicted,
                "count": int(matrix[actual, predicted]),
                "row_percent": float(100 * matrix[actual, predicted] / matrix[actual].sum()),
            })
    return rows


if __name__ == "__main__":
    main()
