from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


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


def model_configs() -> dict[str, dict]:
    return {
        "gaussian_nb_raw": {
            "model": GaussianNB(var_smoothing=1e-9),
            "use_weights": False,
        },
        "gaussian_nb_balanced": {
            "model": GaussianNB(var_smoothing=1e-9),
            "use_weights": True,
        },
        "gaussian_nb_standard": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", GaussianNB(var_smoothing=1e-9)),
                ]
            ),
            "use_weights": True,
        },
        "gaussian_nb_power": {
            "model": Pipeline(
                [
                    ("transformer", PowerTransformer(method="yeo-johnson", standardize=True)),
                    ("model", GaussianNB(var_smoothing=1e-9)),
                ]
            ),
            "use_weights": True,
        },
        "gaussian_nb_quantile": {
            "model": Pipeline(
                [
                    (
                        "transformer",
                        QuantileTransformer(
                            n_quantiles=1000,
                            output_distribution="normal",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                    ("model", GaussianNB(var_smoothing=1e-9)),
                ]
            ),
            "use_weights": True,
        },
        "multinomial_nb_minmax": {
            "model": Pipeline(
                [
                    ("scaler", MinMaxScaler()),
                    ("model", MultinomialNB(alpha=0.1)),
                ]
            ),
            "use_weights": True,
        },
        "complement_nb_minmax": {
            "model": Pipeline(
                [
                    ("scaler", MinMaxScaler()),
                    ("model", ComplementNB(alpha=0.1)),
                ]
            ),
            "use_weights": True,
        },
    }


def fit_model(model: object, x_train: pd.DataFrame, y_train: pd.Series, use_weights: bool) -> object:
    if not use_weights:
        model.fit(x_train, y_train)
        return model

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    if isinstance(model, Pipeline):
        model.fit(x_train, y_train, model__sample_weight=sample_weight)
    else:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


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
    for model_name, config in model_configs().items():
        print(f"Treinando holdout {model_name}...", flush=True)
        model = fit_model(
            config["model"], train_df[mq_columns], train_df[target_column], config["use_weights"]
        )
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
    df: pd.DataFrame, mq_columns: list[str], target_column: str, selected_models: list[str]
) -> pd.DataFrame:
    x = df[mq_columns]
    y = df[target_column]
    groups = df[GROUP_COLUMN]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    configs = model_configs()

    rows = []
    for model_name in selected_models:
        config = configs[model_name]
        fold_scores = []
        for fold_index, (train_index, test_index) in enumerate(cv.split(x, y, groups), 1):
            print(f"Treinando CV {model_name} fold {fold_index}...", flush=True)
            model = fit_model(
                config["model"],
                x.iloc[train_index],
                y.iloc[train_index],
                config["use_weights"],
            )
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
    selected_models = holdout_df.head(3)["modelo"].tolist()
    cv_df = evaluate_group_cv(df, mq_columns, target_column, selected_models)

    holdout_df.to_csv(COMPARISONS_DIR / "comparacao_naive_bayes_holdout.csv", index=False)
    cv_df.to_csv(COMPARISONS_DIR / "comparacao_naive_bayes_cv.csv", index=False)
    (METRICS_DIR / "resumo_avaliacao_naive_bayes.json").write_text(
        json.dumps(
            {
                "dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
                "target": target_column,
                "features_usadas": mq_columns,
                "coluna_usada_apenas_para_split": GROUP_COLUMN,
                "split_holdout": split_summary,
                "tecnicas": [
                    "GaussianNB",
                    "MultinomialNB",
                    "ComplementNB",
                    "StandardScaler",
                    "PowerTransformer",
                    "QuantileTransformer",
                    "pesos balanceados por classe",
                    "validacao por grupo de Coleta",
                ],
                "observacao": (
                    "Coleta, Dia e Vaso nao entram como features do modelo. "
                    "Coleta e usada apenas para evitar vazamento entre treino e teste."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Comparacao Naive Bayes holdout 70/30 por Coleta")
    print(holdout_df[["modelo", "accuracy", "balanced_accuracy", "f1_macro"]].to_string(index=False))
    print()
    print("Media StratifiedGroupKFold por Coleta")
    cv_means = cv_df[cv_df["fold"].astype(str) == "media"].sort_values(
        "accuracy", ascending=False
    )
    print(cv_means[["modelo", "accuracy", "balanced_accuracy", "f1_macro"]].to_string(index=False))


if __name__ == "__main__":
    main()
