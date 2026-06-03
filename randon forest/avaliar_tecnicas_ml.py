from pathlib import Path
import json

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATASET_PATH = (
    ROOT_DIR
    / "dataset_processado_por_dia_vaso_sem_vref0_sem_tempo_soil_temp_pres"
    / "dataset_unico_por_dia_vaso_sem_vref0_sem_tempo_soil_temp_pres.csv"
)
GROUP_COLUMN = "Coleta"
TRAIN_RATIO = 0.70
RANDOM_STATE = 42


def find_target_column(columns: list[str]) -> str:
    for column in columns:
        if column.lower() == "classe":
            return column
    raise ValueError("Coluna alvo 'classe' nao encontrada.")


def load_dataset() -> tuple[pd.DataFrame, list[str], str]:
    df = pd.read_csv(DATASET_PATH)
    target_column = find_target_column(df.columns.tolist())
    mq_columns = [column for column in df.columns if column.upper().startswith("MQ")]

    if GROUP_COLUMN not in df.columns:
        raise ValueError(f"Coluna de agrupamento '{GROUP_COLUMN}' nao encontrada.")
    if not mq_columns:
        raise ValueError("Nenhuma coluna MQ encontrada.")

    model_df = df[[GROUP_COLUMN] + mq_columns + [target_column]].copy()
    for column in mq_columns + [target_column]:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    model_df = model_df.dropna(subset=[GROUP_COLUMN] + mq_columns + [target_column])
    model_df[target_column] = model_df[target_column].astype(int)
    return model_df, mq_columns, target_column


def split_70_30_by_group_inside_class(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, int]]]:
    train_parts = []
    test_parts = []
    summary = []

    for class_value in sorted(df[target_column].unique()):
        class_block = df[df[target_column] == class_value]
        groups = (
            pd.Series(class_block[GROUP_COLUMN].unique())
            .sample(frac=1, random_state=RANDOM_STATE)
            .tolist()
        )
        train_group_count = int(len(groups) * TRAIN_RATIO)
        train_groups = set(groups[:train_group_count])

        train_part = class_block[class_block[GROUP_COLUMN].isin(train_groups)]
        test_part = class_block[~class_block[GROUP_COLUMN].isin(train_groups)]

        train_parts.append(train_part)
        test_parts.append(test_part)
        summary.append(
            {
                "classe": int(class_value),
                "grupos_total": int(len(groups)),
                "grupos_treino": int(len(train_groups)),
                "grupos_teste": int(len(groups) - len(train_groups)),
                "linhas_treino": int(len(train_part)),
                "linhas_teste": int(len(test_part)),
            }
        )

    train_df = pd.concat(train_parts).sample(frac=1, random_state=RANDOM_STATE)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=RANDOM_STATE)
    return train_df, test_df, summary


def build_models() -> dict[str, object]:
    return {
        "random_forest_baseline": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "random_forest_leaf10": RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=10,
        ),
        "extra_trees_leaf5": ExtraTreesClassifier(
            n_estimators=700,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight=None,
            bootstrap=False,
        ),
        "extra_trees_leaf10": ExtraTreesClassifier(
            n_estimators=700,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=10,
            class_weight=None,
            bootstrap=False,
        ),
        "extra_trees_leaf50": ExtraTreesClassifier(
            n_estimators=700,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features=None,
            min_samples_leaf=50,
            class_weight="balanced",
            bootstrap=True,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "logistic_regression_scaled": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def score_model(model: object, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    prediction = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "f1_macro": float(f1_score(y_test, prediction, average="macro")),
    }


def evaluate_holdout(
    df: pd.DataFrame, mq_columns: list[str], target_column: str
) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    train_df, test_df, split_summary = split_70_30_by_group_inside_class(
        df, target_column
    )

    rows = []
    for model_name, model in build_models().items():
        fitted_model = clone(model)
        fitted_model.fit(train_df[mq_columns], train_df[target_column])
        scores = score_model(fitted_model, test_df[mq_columns], test_df[target_column])
        rows.append(
            {
                "avaliacao": "holdout_70_30_por_coleta",
                "modelo": model_name,
                "features_usadas": ",".join(mq_columns),
                "treino_linhas": int(len(train_df)),
                "teste_linhas": int(len(test_df)),
                **scores,
            }
        )

    result_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    return result_df, split_summary


def evaluate_group_cv(
    df: pd.DataFrame, mq_columns: list[str], target_column: str
) -> pd.DataFrame:
    x = df[mq_columns]
    y = df[target_column]
    groups = df[GROUP_COLUMN]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for model_name, model in build_models().items():
        fold_scores = []
        for fold_index, (train_index, test_index) in enumerate(cv.split(x, y, groups), 1):
            fitted_model = clone(model)
            fitted_model.fit(x.iloc[train_index], y.iloc[train_index])
            scores = score_model(fitted_model, x.iloc[test_index], y.iloc[test_index])
            fold_scores.append(scores)
            rows.append(
                {
                    "avaliacao": "stratified_group_kfold",
                    "modelo": model_name,
                    "fold": fold_index,
                    **scores,
                }
            )

        rows.append(
            {
                "avaliacao": "stratified_group_kfold_media",
                "modelo": model_name,
                "fold": "media",
                "accuracy": float(pd.Series([s["accuracy"] for s in fold_scores]).mean()),
                "balanced_accuracy": float(
                    pd.Series([s["balanced_accuracy"] for s in fold_scores]).mean()
                ),
                "f1_macro": float(pd.Series([s["f1_macro"] for s in fold_scores]).mean()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df, mq_columns, target_column = load_dataset()
    holdout_df, split_summary = evaluate_holdout(df, mq_columns, target_column)
    cv_df = evaluate_group_cv(df, mq_columns, target_column)

    holdout_df.to_csv(BASE_DIR / "comparacao_tecnicas_ml_holdout.csv", index=False)
    cv_df.to_csv(BASE_DIR / "comparacao_tecnicas_ml_cv.csv", index=False)
    (BASE_DIR / "resumo_avaliacao_tecnicas_ml.json").write_text(
        json.dumps(
            {
                "dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
                "target": target_column,
                "features_usadas": mq_columns,
                "coluna_usada_apenas_para_split": GROUP_COLUMN,
                "split_holdout": split_summary,
                "observacao": (
                    "Coleta, Dia e Vaso nao entram como features do modelo. "
                    "Coleta e usada apenas para evitar vazamento entre treino e teste."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Comparacao holdout 70/30 por Coleta")
    print(holdout_df[["modelo", "accuracy", "balanced_accuracy", "f1_macro"]].to_string(index=False))
    print()
    print("Media StratifiedGroupKFold por Coleta")
    cv_means = cv_df[cv_df["fold"].astype(str) == "media"].sort_values(
        "accuracy", ascending=False
    )
    print(cv_means[["modelo", "accuracy", "balanced_accuracy", "f1_macro"]].to_string(index=False))


if __name__ == "__main__":
    main()
