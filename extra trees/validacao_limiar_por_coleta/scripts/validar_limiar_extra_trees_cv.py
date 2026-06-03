from pathlib import Path
import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


VALIDATION_DIR = Path(__file__).resolve().parent.parent
CLASSIFIER_DIR = VALIDATION_DIR.parent
ROOT_DIR = CLASSIFIER_DIR.parent
COMPARISONS_DIR = VALIDATION_DIR / "resultados" / "comparacoes"
METRICS_DIR = VALIDATION_DIR / "resultados" / "metricas"
MATRICES_DIR = VALIDATION_DIR / "resultados" / "matrizes"
DATASET_PATH = ROOT_DIR / "sem pressao" / "dataset_sem_pressao.csv"
GROUP_COLUMN = "Coleta"
RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3


def find_target_column(columns: list[str]) -> str:
    for column in columns:
        if column.lower() == "classe":
            return column
    raise ValueError("Coluna alvo 'classe' nao encontrada.")


def load_dataset() -> tuple[pd.DataFrame, list[str], str]:
    df = pd.read_csv(DATASET_PATH)
    target_column = find_target_column(df.columns.tolist())
    mq_columns = [column for column in df.columns if column.upper().startswith("MQ")]

    model_df = df[[GROUP_COLUMN] + mq_columns + [target_column]].copy()
    for column in mq_columns + [target_column]:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    model_df = model_df.dropna(subset=[GROUP_COLUMN] + mq_columns + [target_column])
    model_df[target_column] = model_df[target_column].astype(int)
    return model_df, mq_columns, target_column


def build_model(seed: int = RANDOM_STATE) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=900,
        random_state=seed,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=10,
        class_weight=None,
        bootstrap=False,
    )


def metrics_from_prediction(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1_macro": float(f1_score(y_true, prediction, average="macro")),
    }


def tune_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.30, 0.75, 46):
        prediction = (probabilities >= threshold).astype(int)
        metrics = metrics_from_prediction(y_true, prediction)
        rows.append(
            {
                "threshold": float(threshold),
                **metrics,
            }
        )

    threshold_df = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "accuracy", "f1_macro"], ascending=False
    )
    return float(threshold_df.iloc[0]["threshold"]), threshold_df


def tune_threshold_with_inner_cv(
    train_df: pd.DataFrame, mq_columns: list[str], target_column: str, fold_seed: int
) -> tuple[float, pd.DataFrame]:
    x = train_df[mq_columns]
    y = train_df[target_column]
    groups = train_df[GROUP_COLUMN]
    cv = StratifiedGroupKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=fold_seed
    )

    validation_probabilities = []
    validation_targets = []
    fold_rows = []

    for inner_fold, (fit_index, validation_index) in enumerate(cv.split(x, y, groups), 1):
        model = build_model(seed=fold_seed + inner_fold)
        model.fit(x.iloc[fit_index], y.iloc[fit_index])
        probabilities = model.predict_proba(x.iloc[validation_index])[:, 1]
        validation_probabilities.append(probabilities)
        validation_targets.append(y.iloc[validation_index].to_numpy())

        default_prediction = (probabilities >= 0.50).astype(int)
        default_metrics = metrics_from_prediction(
            y.iloc[validation_index], default_prediction
        )
        fold_rows.append(
            {
                "inner_fold": inner_fold,
                "threshold": 0.50,
                "tipo": "limiar_padrao",
                **default_metrics,
            }
        )

    y_validation = np.concatenate(validation_targets)
    p_validation = np.concatenate(validation_probabilities)
    threshold, threshold_df = tune_threshold(pd.Series(y_validation), p_validation)
    threshold_df["tipo"] = "busca_limiar"
    threshold_df["inner_fold"] = "todos"

    return threshold, pd.concat([pd.DataFrame(fold_rows), threshold_df], ignore_index=True)


def plot_thresholds(result_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result_df["outer_fold"], result_df["threshold_validado"], marker="o")
    ax.axhline(0.50, color="gray", linestyle="--", label="limiar padrao 0.50")
    ax.axhline(0.57, color="green", linestyle="--", label="limiar anterior 0.57")
    ax.set_xticks(result_df["outer_fold"])
    ax.set_ylim(0.30, 0.75)
    ax.set_title("Limiar validado por fold externo")
    ax.set_xlabel("Fold externo")
    ax.set_ylabel("Limiar escolhido")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MATRICES_DIR.mkdir(parents=True, exist_ok=True)

    df, mq_columns, target_column = load_dataset()
    x = df[mq_columns]
    y = df[target_column]
    groups = df[GROUP_COLUMN]

    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    fold_rows = []
    inner_rows = []
    matrix_default_total = np.zeros((2, 2), dtype=int)
    matrix_tuned_total = np.zeros((2, 2), dtype=int)
    matrix_fixed_057_total = np.zeros((2, 2), dtype=int)

    for outer_fold, (train_index, test_index) in enumerate(outer_cv.split(x, y, groups), 1):
        print(f"Fold externo {outer_fold}/{OUTER_FOLDS}: ajustando limiar...", flush=True)
        train_df = df.iloc[train_index].copy()
        test_df = df.iloc[test_index].copy()

        fold_seed = RANDOM_STATE + outer_fold * 100
        threshold, threshold_details = tune_threshold_with_inner_cv(
            train_df, mq_columns, target_column, fold_seed
        )
        threshold_details["outer_fold"] = outer_fold
        inner_rows.append(threshold_details)

        print(
            f"Fold externo {outer_fold}: treinando modelo final com limiar {threshold:.2f}...",
            flush=True,
        )
        model = build_model(seed=fold_seed)
        model.fit(train_df[mq_columns], train_df[target_column])
        probabilities = model.predict_proba(test_df[mq_columns])[:, 1]

        default_prediction = (probabilities >= 0.50).astype(int)
        tuned_prediction = (probabilities >= threshold).astype(int)
        fixed_057_prediction = (probabilities >= 0.57).astype(int)

        default_metrics = metrics_from_prediction(test_df[target_column], default_prediction)
        tuned_metrics = metrics_from_prediction(test_df[target_column], tuned_prediction)
        fixed_057_metrics = metrics_from_prediction(
            test_df[target_column], fixed_057_prediction
        )

        default_matrix = confusion_matrix(
            test_df[target_column], default_prediction, labels=[0, 1]
        )
        tuned_matrix = confusion_matrix(
            test_df[target_column], tuned_prediction, labels=[0, 1]
        )
        fixed_057_matrix = confusion_matrix(
            test_df[target_column], fixed_057_prediction, labels=[0, 1]
        )

        matrix_default_total += default_matrix
        matrix_tuned_total += tuned_matrix
        matrix_fixed_057_total += fixed_057_matrix

        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "threshold_validado": threshold,
                "treino_linhas": int(len(train_df)),
                "teste_linhas": int(len(test_df)),
                "treino_coletas": int(train_df[GROUP_COLUMN].nunique()),
                "teste_coletas": int(test_df[GROUP_COLUMN].nunique()),
                "accuracy_limiar_050": default_metrics["accuracy"],
                "balanced_accuracy_limiar_050": default_metrics["balanced_accuracy"],
                "f1_macro_limiar_050": default_metrics["f1_macro"],
                "accuracy_limiar_validado": tuned_metrics["accuracy"],
                "balanced_accuracy_limiar_validado": tuned_metrics["balanced_accuracy"],
                "f1_macro_limiar_validado": tuned_metrics["f1_macro"],
                "accuracy_limiar_057_fixo": fixed_057_metrics["accuracy"],
                "balanced_accuracy_limiar_057_fixo": fixed_057_metrics["balanced_accuracy"],
                "f1_macro_limiar_057_fixo": fixed_057_metrics["f1_macro"],
                "matriz_limiar_050": default_matrix.astype(int).tolist(),
                "matriz_limiar_validado": tuned_matrix.astype(int).tolist(),
                "matriz_limiar_057_fixo": fixed_057_matrix.astype(int).tolist(),
            }
        )

    result_df = pd.DataFrame(fold_rows)
    inner_df = pd.concat(inner_rows, ignore_index=True)

    result_df.to_csv(
        COMPARISONS_DIR / "validacao_cruzada_limiar_extra_trees.csv", index=False
    )
    inner_df.to_csv(
        COMPARISONS_DIR / "validacao_cruzada_limiar_extra_trees_detalhes_internos.csv",
        index=False,
    )

    pd.DataFrame(
        matrix_tuned_total,
        index=["real_0_doente", "real_1_saudavel"],
        columns=["previsto_0_doente", "previsto_1_saudavel"],
    ).to_csv(MATRICES_DIR / "matriz_confusao_extra_trees_cv_limiar_validado.csv")

    plot_thresholds(
        result_df, COMPARISONS_DIR / "limiares_extra_trees_cv.png"
    )

    metrics = {
        "modelo": "ExtraTreesClassifier",
        "dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
        "target": target_column,
        "target_mapping": {"0": "doente", "1": "saudavel"},
        "features_usadas": mq_columns,
        "coluna_usada_apenas_para_split": GROUP_COLUMN,
        "validacao": "nested StratifiedGroupKFold por Coleta",
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "thresholds_validados": result_df["threshold_validado"].tolist(),
        "threshold_medio": float(result_df["threshold_validado"].mean()),
        "threshold_mediano": float(result_df["threshold_validado"].median()),
        "threshold_desvio": float(result_df["threshold_validado"].std(ddof=0)),
        "metricas_medias": {
            "limiar_050": {
                "accuracy": float(result_df["accuracy_limiar_050"].mean()),
                "balanced_accuracy": float(
                    result_df["balanced_accuracy_limiar_050"].mean()
                ),
                "f1_macro": float(result_df["f1_macro_limiar_050"].mean()),
            },
            "limiar_validado_por_fold": {
                "accuracy": float(result_df["accuracy_limiar_validado"].mean()),
                "balanced_accuracy": float(
                    result_df["balanced_accuracy_limiar_validado"].mean()
                ),
                "f1_macro": float(result_df["f1_macro_limiar_validado"].mean()),
            },
            "limiar_057_fixo": {
                "accuracy": float(result_df["accuracy_limiar_057_fixo"].mean()),
                "balanced_accuracy": float(
                    result_df["balanced_accuracy_limiar_057_fixo"].mean()
                ),
                "f1_macro": float(result_df["f1_macro_limiar_057_fixo"].mean()),
            },
        },
        "matrizes_somadas": {
            "limiar_050": matrix_default_total.astype(int).tolist(),
            "limiar_validado_por_fold": matrix_tuned_total.astype(int).tolist(),
            "limiar_057_fixo": matrix_fixed_057_total.astype(int).tolist(),
        },
        "observacao": (
            "Em cada fold externo, o limiar foi escolhido apenas com folds internos "
            "do treino. As coletas do teste externo permaneceram isoladas."
        ),
    }
    (METRICS_DIR / "metricas_validacao_cruzada_limiar_extra_trees.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("Validacao cruzada do limiar concluida.")
    print("Limiares por fold:", ", ".join(f"{v:.2f}" for v in metrics["thresholds_validados"]))
    print(
        f"Limiar medio: {metrics['threshold_medio']:.3f} | "
        f"mediano: {metrics['threshold_mediano']:.3f} | "
        f"desvio: {metrics['threshold_desvio']:.3f}"
    )
    print("Metricas medias:")
    for name, values in metrics["metricas_medias"].items():
        print(
            f"  {name}: acc={values['accuracy']:.4f} "
            f"bal={values['balanced_accuracy']:.4f} "
            f"f1={values['f1_macro']:.4f}"
        )
    print("Matriz somada limiar validado:")
    print(matrix_tuned_total)


if __name__ == "__main__":
    main()
