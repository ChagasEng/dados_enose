from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
COMPARISONS_DIR = PROJECT_DIR / "resultados" / "comparacoes"
METRICS_DIR = PROJECT_DIR / "resultados" / "metricas"
DATASET_PATH = ROOT_DIR / "sem pressao" / "dataset_sem_pressao.csv"
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


def build_model(params: dict, y_train: pd.Series) -> XGBClassifier:
    model_params = params.copy()
    if model_params.pop("usar_scale_pos_weight", False):
        negative_count = int((y_train == 0).sum())
        positive_count = int((y_train == 1).sum())
        model_params["scale_pos_weight"] = negative_count / positive_count

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **model_params,
    )


def model_configs() -> dict[str, dict]:
    return {
        "xgboost_depth3_regularizado": {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 5,
            "reg_lambda": 5.0,
            "reg_alpha": 0.1,
        },
        "xgboost_depth4": {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 3,
            "reg_lambda": 2.0,
        },
        "xgboost_depth5": {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.04,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 3,
            "reg_lambda": 2.0,
        },
        "xgboost_depth4_balanceado": {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 3,
            "reg_lambda": 2.0,
            "usar_scale_pos_weight": True,
        },
        "xgboost_depth6_regularizado": {
            "n_estimators": 350,
            "max_depth": 6,
            "learning_rate": 0.035,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 8,
            "reg_lambda": 8.0,
            "reg_alpha": 0.2,
        },
    }


def score_model(model: XGBClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
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
    for model_name, params in model_configs().items():
        model = build_model(params, train_df[target_column])
        model.fit(train_df[mq_columns], train_df[target_column])
        scores = score_model(model, test_df[mq_columns], test_df[target_column])
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
    for model_name, params in model_configs().items():
        fold_scores = []
        for fold_index, (train_index, test_index) in enumerate(cv.split(x, y, groups), 1):
            model = build_model(params, y.iloc[train_index])
            model.fit(x.iloc[train_index], y.iloc[train_index])
            scores = score_model(model, x.iloc[test_index], y.iloc[test_index])
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
    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df, mq_columns, target_column = load_dataset()
    holdout_df, split_summary = evaluate_holdout(df, mq_columns, target_column)
    cv_df = evaluate_group_cv(df, mq_columns, target_column)

    holdout_df.to_csv(COMPARISONS_DIR / "comparacao_xgboost_holdout.csv", index=False)
    cv_df.to_csv(COMPARISONS_DIR / "comparacao_xgboost_cv.csv", index=False)
    (METRICS_DIR / "resumo_avaliacao_xgboost.json").write_text(
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

    print("Comparacao XGBoost holdout 70/30 por Coleta")
    print(holdout_df[["modelo", "accuracy", "balanced_accuracy", "f1_macro"]].to_string(index=False))
    print()
    print("Media StratifiedGroupKFold por Coleta")
    cv_means = cv_df[cv_df["fold"].astype(str) == "media"].sort_values(
        "accuracy", ascending=False
    )
    print(cv_means[["modelo", "accuracy", "balanced_accuracy", "f1_macro"]].to_string(index=False))


if __name__ == "__main__":
    main()
