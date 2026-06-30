from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(__file__).resolve().parents[1]

PRIMARY_DATASET_PATH = ROOT_DIR / "sem pressao" / "dataset_sem_pressao.csv"
ENV_DATASET_PATH = (
    ROOT_DIR
    / "dataset_processado_por_dia_vaso_sem_vref0"
    / "dataset_unico_por_dia_vaso_sem_vref0.csv"
)

DATA_DIR = PROJECT_DIR / "dados_preprocessados"
RESULTS_DIR = PROJECT_DIR / "resultados"
METRICS_DIR = RESULTS_DIR / "metricas"
REPORTS_DIR = RESULTS_DIR / "relatorios"
MATRICES_DIR = RESULTS_DIR / "matrizes"
COMPARISONS_DIR = RESULTS_DIR / "comparacoes"
FEATURE_SELECTION_DIR = RESULTS_DIR / "selecao_atributos"
GRAPHICS_DIR = PROJECT_DIR / "graficos"
MODELS_DIR = PROJECT_DIR / "modelos"

GROUP_COLUMN = "Coleta"
TARGET_COLUMN = "Classe"
RANDOM_STATE = 42
N_SPLITS = 10
MOVING_AVERAGE_WINDOW = 11


def ensure_dirs() -> None:
    for directory in [
        DATA_DIR,
        METRICS_DIR,
        REPORTS_DIR,
        MATRICES_DIR,
        COMPARISONS_DIR,
        FEATURE_SELECTION_DIR,
        GRAPHICS_DIR,
        MODELS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def find_mq_columns(columns: list[str]) -> list[str]:
    mq_columns = [column for column in columns if column.upper().startswith("MQ")]
    if not mq_columns:
        raise ValueError("Nenhuma coluna MQ encontrada.")
    return mq_columns


def load_author_base() -> tuple[pd.DataFrame, list[str], list[str], dict[str, object]]:
    primary_df = pd.read_csv(PRIMARY_DATASET_PATH)
    primary_mq_columns = find_mq_columns(primary_df.columns.tolist())

    if ENV_DATASET_PATH.exists():
        df = pd.read_csv(ENV_DATASET_PATH)
        df = df[pd.to_numeric(df["Dia"], errors="coerce") < 20].copy()
        env_columns = [column for column in ["Temp.", "Pres."] if column in df.columns]
        source = str(ENV_DATASET_PATH.relative_to(ROOT_DIR))
        source_note = (
            "Usada a base anterior filtrada por Dia < 20 para manter as mesmas "
            "coletas da pasta sem pressao e recuperar Temp./Pres."
        )
    else:
        df = primary_df.copy()
        env_columns = []
        source = str(PRIMARY_DATASET_PATH.relative_to(ROOT_DIR))
        source_note = "Base sem pressao usada diretamente; sem colunas ambientais."

    mq_columns = find_mq_columns(df.columns.tolist())
    required_columns = [GROUP_COLUMN, TARGET_COLUMN, *mq_columns]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")

    for column in [*mq_columns, TARGET_COLUMN, "Dia", "Vaso", *env_columns]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[GROUP_COLUMN, TARGET_COLUMN, *mq_columns]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    df = df.reset_index(drop=True)

    primary_target = pd.to_numeric(primary_df[TARGET_COLUMN], errors="coerce")
    primary_counts = (
        primary_target.dropna().astype(int).value_counts().sort_index().to_dict()
    )
    base_counts = df[TARGET_COLUMN].value_counts().sort_index().to_dict()
    metadata = {
        "dataset_origem": source,
        "observacao_origem": source_note,
        "linhas_base_sem_pressao": int(len(primary_df)),
        "linhas_base_sem_pressao_com_classe_vazia": int(primary_target.isna().sum()),
        "linhas_base_usada": int(len(df)),
        "classes_base_sem_pressao": {str(k): int(v) for k, v in primary_counts.items()},
        "classes_base_usada": {str(k): int(v) for k, v in base_counts.items()},
        "colunas_ambientais_disponiveis": env_columns,
        "colunas_mq": mq_columns,
        "colunas_mq_base_sem_pressao": primary_mq_columns,
    }
    return df, mq_columns, env_columns, metadata


def modified_moving_average(
    df: pd.DataFrame,
    sensor_columns: list[str],
    group_column: str,
    window: int = MOVING_AVERAGE_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = df.copy()
    output[sensor_columns] = output[sensor_columns].astype(float)
    half_window = window // 2
    outlier_counts = {column: 0 for column in sensor_columns}

    for _, group in df.groupby(group_column, sort=False):
        group_index = group.index.to_numpy()
        for column in sensor_columns:
            values = group[column].astype(float).to_numpy()
            smoothed = np.empty_like(values, dtype=float)

            for position in range(len(values)):
                start = max(0, position - half_window)
                end = min(len(values), position + half_window + 1)
                local_values = values[start:end]
                local_values = local_values[~np.isnan(local_values)]

                if len(local_values) == 0:
                    smoothed[position] = values[position]
                    continue

                local_mean = float(np.mean(local_values))
                local_std = float(np.std(local_values))
                if local_std == 0:
                    accepted_values = local_values
                    center_is_outlier = False
                else:
                    accepted_values = local_values[
                        np.abs(local_values - local_mean) <= 3 * local_std
                    ]
                    center_is_outlier = (
                        abs(float(values[position]) - local_mean) > 3 * local_std
                    )

                if center_is_outlier:
                    outlier_counts[column] += 1

                if len(accepted_values) == 0:
                    smoothed[position] = local_mean
                else:
                    smoothed[position] = float(np.mean(accepted_values))

            output.loc[group_index, column] = smoothed

    summary = pd.DataFrame(
        [
            {
                "sensor": column,
                "pontos_marcados_como_outlier_local": int(count),
                "janela_media_movel": int(window),
            }
            for column, count in outlier_counts.items()
        ]
    )
    return output, summary


def compensate_environment(
    df: pd.DataFrame,
    sensor_columns: list[str],
    env_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = df.copy()
    if not env_columns:
        return output, pd.DataFrame()

    env_values = output[env_columns].astype(float)
    env_means = env_values.mean(axis=0)
    params = []

    for sensor in sensor_columns:
        y = output[sensor].astype(float)
        valid_mask = y.notna()
        for env_column in env_columns:
            valid_mask &= env_values[env_column].notna()

        x_valid = env_values.loc[valid_mask, env_columns].to_numpy(dtype=float)
        y_valid = y.loc[valid_mask].to_numpy(dtype=float)

        design = np.column_stack([np.ones(len(x_valid)), x_valid])
        coefficients, *_ = np.linalg.lstsq(design, y_valid, rcond=None)
        intercept = float(coefficients[0])
        slopes = coefficients[1:]

        centered_env = env_values[env_columns] - env_means
        correction = centered_env.to_numpy(dtype=float) @ slopes
        output[sensor] = y - correction

        row = {
            "sensor": sensor,
            "intercepto": intercept,
            "media_sensor_antes": float(y.mean()),
            "media_sensor_depois": float(output[sensor].mean()),
        }
        for env_column, slope in zip(env_columns, slopes):
            row[f"coef_{env_column}"] = float(slope)
            row[f"media_{env_column}"] = float(env_means[env_column])
        params.append(row)

    return output, pd.DataFrame(params)


def cramers_v(feature: pd.Series, target: pd.Series, bins: int = 10) -> float:
    binned = pd.qcut(feature, q=bins, duplicates="drop")
    table = pd.crosstab(binned, target)
    observed = table.to_numpy(dtype=float)
    if observed.size == 0:
        return 0.0

    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row_sum @ col_sum / total
    valid = expected > 0
    chi_square = float(((observed - expected) ** 2 / np.where(valid, expected, 1)).sum())
    denominator = total * max(1, min(observed.shape[0] - 1, observed.shape[1] - 1))
    return float(math.sqrt(chi_square / denominator)) if denominator else 0.0


def run_feature_selection(
    df: pd.DataFrame,
    sensor_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df[sensor_columns]
    y = df[TARGET_COLUMN]
    scaled = MinMaxScaler().fit_transform(x)

    chi2_scores, chi2_p_values = chi2(scaled, y)
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=10,
        class_weight="balanced",
    )
    rf.fit(scaled, y)

    score_rows = []
    for index, sensor in enumerate(sensor_columns):
        score_rows.append(
            {
                "sensor": sensor,
                "chi2_score": float(chi2_scores[index]),
                "chi2_p_value": float(chi2_p_values[index]),
                "cramers_v": cramers_v(df[sensor], y),
                "random_forest_importance": float(rf.feature_importances_[index]),
            }
        )

    scores_df = pd.DataFrame(score_rows).sort_values(
        ["chi2_score", "random_forest_importance"], ascending=False
    )

    exhaustive_df = exhaustive_subset_search(df, sensor_columns)
    return scores_df, exhaustive_df


def split_70_30_by_group_inside_class(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    rng = np.random.default_rng(RANDOM_STATE)

    for class_value in sorted(df[TARGET_COLUMN].unique()):
        class_block = df[df[TARGET_COLUMN] == class_value]
        groups = np.array(sorted(class_block[GROUP_COLUMN].unique()))
        rng.shuffle(groups)
        train_count = int(len(groups) * 0.70)
        train_groups = set(groups[:train_count])
        train_parts.append(class_block[class_block[GROUP_COLUMN].isin(train_groups)])
        test_parts.append(class_block[~class_block[GROUP_COLUMN].isin(train_groups)])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=RANDOM_STATE)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=RANDOM_STATE)
    return train_df, test_df


def exhaustive_subset_search(
    df: pd.DataFrame,
    sensor_columns: list[str],
) -> pd.DataFrame:
    train_df, test_df = split_70_30_by_group_inside_class(df)
    rows = []

    for subset_size in range(1, len(sensor_columns) + 1):
        for subset in itertools.combinations(sensor_columns, subset_size):
            pipeline = Pipeline(
                [
                    ("minmax", MinMaxScaler()),
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=180,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            max_features="sqrt",
                            min_samples_leaf=10,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )
            pipeline.fit(train_df[list(subset)], train_df[TARGET_COLUMN])
            prediction = pipeline.predict(test_df[list(subset)])
            rows.append(
                {
                    "sensores": ",".join(subset),
                    "qtd_sensores": int(len(subset)),
                    "accuracy": float(accuracy_score(test_df[TARGET_COLUMN], prediction)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(test_df[TARGET_COLUMN], prediction)
                    ),
                    "f1_macro": float(
                        f1_score(test_df[TARGET_COLUMN], prediction, average="macro")
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "f1_macro", "accuracy"], ascending=False
    )


def build_pipelines() -> dict[str, Pipeline]:
    classifiers = {
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
        "svm_linear": LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=10,
            class_weight="balanced",
        ),
    }
    transforms = {
        "minmax": [("minmax", MinMaxScaler())],
        "pca3": [("minmax", MinMaxScaler()), ("pca", PCA(n_components=3))],
        "lda1": [
            ("minmax", MinMaxScaler()),
            ("lda", LinearDiscriminantAnalysis(n_components=1)),
        ],
    }

    pipelines = {}
    for transform_name, transform_steps in transforms.items():
        for classifier_name, classifier in classifiers.items():
            pipelines[f"{transform_name}_{classifier_name}"] = Pipeline(
                [*transform_steps, ("classifier", classifier)]
            )
    return pipelines


def evaluate_pipelines(
    df: pd.DataFrame,
    sensor_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str]]:
    x = df[sensor_columns]
    y = df[TARGET_COLUMN].astype(int)
    groups = df[GROUP_COLUMN]
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    predictions_by_model = {}
    reports_by_model = {}

    for model_name, pipeline in build_pipelines().items():
        fold_predictions = np.full(len(df), fill_value=-1, dtype=int)
        fold_scores = []

        for fold_index, (train_index, test_index) in enumerate(cv.split(x, y, groups), 1):
            model = clone(pipeline)
            model.fit(x.iloc[train_index], y.iloc[train_index])
            prediction = model.predict(x.iloc[test_index]).astype(int)
            fold_predictions[test_index] = prediction
            fold_scores.append(
                {
                    "modelo": model_name,
                    "fold": fold_index,
                    "accuracy": float(accuracy_score(y.iloc[test_index], prediction)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y.iloc[test_index], prediction)
                    ),
                    "f1_macro": float(
                        f1_score(y.iloc[test_index], prediction, average="macro")
                    ),
                }
            )

        valid_mask = fold_predictions >= 0
        y_true = y.to_numpy()[valid_mask]
        y_pred = fold_predictions[valid_mask]
        predictions_by_model[model_name] = fold_predictions
        reports_by_model[model_name] = classification_report(
            y_true,
            y_pred,
            target_names=["doente", "saudavel"],
            digits=4,
            zero_division=0,
        )

        rows.extend(fold_scores)
        rows.append(
            {
                "modelo": model_name,
                "fold": "media",
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            }
        )

    return pd.DataFrame(rows), predictions_by_model, reports_by_model


def save_confusion_matrix(
    matrix: np.ndarray,
    model_name: str,
    output_png: Path,
    output_csv: Path,
) -> None:
    pd.DataFrame(
        matrix,
        index=["real_doente", "real_saudavel"],
        columns=["pred_doente", "pred_saudavel"],
    ).to_csv(output_csv)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Matriz de confusao - {model_name}")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1], ["doente", "saudavel"])
    ax.set_yticks([0, 1], ["doente", "saudavel"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def save_plots(
    df: pd.DataFrame,
    sensor_columns: list[str],
    cv_summary: pd.DataFrame,
    feature_scores: pd.DataFrame,
) -> None:
    x_scaled = MinMaxScaler().fit_transform(df[sensor_columns])
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    sample_size = min(6000, len(df))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_index = rng.choice(len(df), size=sample_size, replace=False)

    pca = PCA(n_components=3)
    pca_values = pca.fit_transform(x_scaled)
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda_values = lda.fit_transform(x_scaled, y).reshape(-1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["PC1", "PC2", "PC3"], pca.explained_variance_ratio_)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Variancia explicada")
    ax.set_title("PCA - variancia explicada")
    fig.tight_layout()
    fig.savefig(GRAPHICS_DIR / "pca_variancia_explicada.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for class_value, label, color in [(0, "doente", "#c0392b"), (1, "saudavel", "#1f618d")]:
        mask = y[sample_index] == class_value
        ax.scatter(
            pca_values[sample_index][mask, 0],
            pca_values[sample_index][mask, 1],
            s=8,
            alpha=0.45,
            label=label,
            color=color,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA 2D por classe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHICS_DIR / "pca_2d_classes.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    for class_value, label, color in [(0, "doente", "#c0392b"), (1, "saudavel", "#1f618d")]:
        mask = y[sample_index] == class_value
        ax.scatter(
            pca_values[sample_index][mask, 0],
            pca_values[sample_index][mask, 1],
            pca_values[sample_index][mask, 2],
            s=7,
            alpha=0.35,
            label=label,
            color=color,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA 3D por classe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHICS_DIR / "pca_3d_classes.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(lda_values[y == 0], bins=45, alpha=0.65, label="doente", color="#c0392b")
    ax.hist(lda_values[y == 1], bins=45, alpha=0.65, label="saudavel", color="#1f618d")
    ax.set_title("LDA 1D por classe")
    ax.set_xlabel("LD1")
    ax.set_ylabel("Frequencia")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHICS_DIR / "lda_1d_classes.png", dpi=160)
    plt.close(fig)

    media_df = cv_summary[cv_summary["fold"].astype(str) == "media"].sort_values(
        "balanced_accuracy", ascending=False
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(media_df["modelo"], media_df["balanced_accuracy"], color="#2874a6")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Comparacao dos modelos - metodo do autor")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(GRAPHICS_DIR / "comparacao_modelos_balanced_accuracy.png", dpi=170)
    plt.close(fig)

    feature_plot_df = feature_scores.sort_values("chi2_score", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(feature_plot_df["sensor"], feature_plot_df["chi2_score"], color="#117864")
    ax.set_xlabel("Chi-quadrado")
    ax.set_title("Selecao de atributos - chi-quadrado")
    fig.tight_layout()
    fig.savefig(GRAPHICS_DIR / "selecao_atributos_chi2.png", dpi=160)
    plt.close(fig)


def write_report(
    metadata: dict[str, object],
    smoothing_summary: pd.DataFrame,
    compensation_params: pd.DataFrame,
    feature_scores: pd.DataFrame,
    exhaustive_df: pd.DataFrame,
    cv_summary: pd.DataFrame,
    best_model_name: str,
    best_report: str,
) -> None:
    media_df = cv_summary[cv_summary["fold"].astype(str) == "media"].sort_values(
        "balanced_accuracy", ascending=False
    )
    best_row = media_df.iloc[0]
    best_subset = exhaustive_df.iloc[0]
    env_columns = metadata["colunas_ambientais_disponiveis"]

    lines = [
        "Relatorio - Metodo inspirado no autor",
        "=====================================",
        "",
        "Base usada:",
        f"- Origem: {metadata['dataset_origem']}",
        f"- Observacao: {metadata['observacao_origem']}",
        f"- Linhas usadas: {metadata['linhas_base_usada']}",
        "- Linhas com Classe vazia na base sem pressao original: "
        f"{metadata['linhas_base_sem_pressao_com_classe_vazia']}",
        f"- Classes: {metadata['classes_base_usada']}",
        "",
        "Pre-processamento aplicado:",
        f"- Filtro de media movel modificado com janela {MOVING_AVERAGE_WINDOW}.",
        "- Normalizacao maximo-minimo dentro dos pipelines de modelagem.",
        "- PCA com 3 componentes.",
        "- LDA com 1 componente, porque a base tem apenas 2 classes.",
        "- Validacao cruzada StratifiedGroupKFold por Coleta com 10 folds.",
        "",
        "Compensacao ambiental:",
    ]

    if env_columns:
        lines.extend(
            [
                f"- Colunas ambientais disponiveis: {', '.join(env_columns)}.",
                "- Nao existe coluna explicita de umidade relativa nesta base.",
                "- Foi aplicada compensacao linear usando as variaveis ambientais disponiveis.",
            ]
        )
    else:
        lines.append("- Nao aplicada, porque a base nao possui colunas ambientais.")

    lines.extend(
        [
            "",
            "Sensores usados:",
            "- " + ", ".join(metadata["colunas_mq"]),
            "",
            "Melhor resultado geral:",
            f"- Modelo: {best_model_name}",
            f"- Accuracy: {best_row['accuracy']:.6f}",
            f"- Balanced accuracy: {best_row['balanced_accuracy']:.6f}",
            f"- F1 macro: {best_row['f1_macro']:.6f}",
            "",
            "Melhor subconjunto na busca exaustiva RF holdout:",
            f"- Sensores: {best_subset['sensores']}",
            f"- Qtd sensores: {int(best_subset['qtd_sensores'])}",
            f"- Accuracy: {best_subset['accuracy']:.6f}",
            f"- Balanced accuracy: {best_subset['balanced_accuracy']:.6f}",
            f"- F1 macro: {best_subset['f1_macro']:.6f}",
            "",
            "Ranking medio dos modelos:",
        ]
    )

    for _, row in media_df.iterrows():
        lines.append(
            "- {modelo}: accuracy={accuracy:.6f}, balanced_accuracy={balanced_accuracy:.6f}, "
            "f1_macro={f1_macro:.6f}".format(**row.to_dict())
        )

    lines.extend(
        [
            "",
            "Top sensores por chi-quadrado:",
        ]
    )
    for _, row in feature_scores.head(6).iterrows():
        lines.append(
            f"- {row['sensor']}: chi2={row['chi2_score']:.4f}, "
            f"cramers_v={row['cramers_v']:.4f}, "
            f"rf_importance={row['random_forest_importance']:.4f}"
        )

    lines.extend(
        [
            "",
            "Resumo do filtro de media movel:",
        ]
    )
    for _, row in smoothing_summary.iterrows():
        lines.append(
            f"- {row['sensor']}: {int(row['pontos_marcados_como_outlier_local'])} "
            "pontos marcados como outlier local"
        )

    if not compensation_params.empty:
        lines.extend(["", "Parametros da compensacao ambiental:"])
        for _, row in compensation_params.iterrows():
            details = [
                f"{column}={row[column]:.6f}"
                for column in compensation_params.columns
                if column.startswith("coef_")
            ]
            lines.append(f"- {row['sensor']}: " + ", ".join(details))

    lines.extend(
        [
            "",
            "Relatorio de classificacao do melhor modelo:",
            best_report,
            "",
            "Arquivos principais:",
            "- resultados/metricas/resumo_modelos.csv",
            "- resultados/metricas/resumo_modelos.json",
            "- resultados/relatorios/relatorio_metodo_autor.txt",
            "- resultados/matrizes/matriz_confusao_melhor_modelo.png",
            "- resultados/selecao_atributos/feature_selection_scores.csv",
            "- resultados/selecao_atributos/busca_exaustiva_subconjuntos_rf.csv",
            "- graficos/comparacao_modelos_balanced_accuracy.png",
            "- graficos/pca_2d_classes.png",
            "- graficos/pca_3d_classes.png",
            "- graficos/lda_1d_classes.png",
            "- modelos/melhor_modelo_metodo_autor.joblib",
        ]
    )

    (REPORTS_DIR / "relatorio_metodo_autor.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_readme(best_model_name: str, best_scores: dict[str, float]) -> None:
    content = f"""# Metodo do autor aplicado a base sem pressao

Esta pasta contem uma reproducao adaptada das tecnicas descritas no trabalho do autor, usando a base que vinha dando melhor resultado no projeto.

## Melhor modelo encontrado nesta rodada

- Modelo: `{best_model_name}`
- Accuracy: `{best_scores["accuracy"]:.6f}`
- Balanced accuracy: `{best_scores["balanced_accuracy"]:.6f}`
- F1 macro: `{best_scores["f1_macro"]:.6f}`

## O que foi aplicado

- Filtro de media movel modificado nos sensores MQ.
- Compensacao linear com as colunas ambientais disponiveis (`Temp.` e `Pres.`), quando presentes.
- Normalizacao maximo-minimo.
- Selecao de atributos com chi-quadrado, V de Cramer, importancia por Random Forest e busca exaustiva por subconjuntos.
- PCA com 3 componentes.
- LDA com 1 componente, pois esta base possui apenas 2 classes.
- KNN, SVM linear e Random Forest.
- Validacao cruzada `StratifiedGroupKFold` por `Coleta` com 10 folds.

## Arquivos principais

- `scripts/rodar_metodo_autor.py`: script que gera toda a pasta.
- `resultados/relatorios/relatorio_metodo_autor.txt`: leitura principal dos resultados.
- `resultados/metricas/resumo_modelos.csv`: comparacao dos modelos.
- `resultados/matrizes/matriz_confusao_melhor_modelo.png`: matriz de confusao do melhor modelo.
- `resultados/selecao_atributos/feature_selection_scores.csv`: pontuacao dos sensores.
- `resultados/selecao_atributos/busca_exaustiva_subconjuntos_rf.csv`: busca de subconjuntos.
- `graficos/`: PCA, LDA, comparacao de modelos e selecao de atributos.
- `modelos/melhor_modelo_metodo_autor.joblib`: melhor pipeline treinado na base completa.

## Limitacoes

A base possui apenas os 6 sensores MQ atuais, entao nao foi possivel repetir exatamente a reducao de 13 sensores para 6. Tambem nao existe coluna explicita de umidade relativa; a compensacao ambiental usa apenas `Temp.` e `Pres.`.
"""
    (PROJECT_DIR / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    raw_df, mq_columns, env_columns, metadata = load_author_base()
    raw_df.to_csv(DATA_DIR / "dataset_base_metodo_autor.csv", index=False)

    smoothed_df, smoothing_summary = modified_moving_average(
        raw_df, mq_columns, GROUP_COLUMN
    )
    smoothing_summary.to_csv(DATA_DIR / "resumo_media_movel_modificada.csv", index=False)
    smoothed_df.to_csv(DATA_DIR / "dataset_media_movel_modificada.csv", index=False)

    compensated_df, compensation_params = compensate_environment(
        smoothed_df, mq_columns, env_columns
    )
    compensation_params.to_csv(
        DATA_DIR / "parametros_compensacao_ambiental.csv", index=False
    )
    compensated_df.to_csv(DATA_DIR / "dataset_media_movel_compensado.csv", index=False)

    feature_scores, exhaustive_df = run_feature_selection(compensated_df, mq_columns)
    feature_scores.to_csv(
        FEATURE_SELECTION_DIR / "feature_selection_scores.csv", index=False
    )
    exhaustive_df.to_csv(
        FEATURE_SELECTION_DIR / "busca_exaustiva_subconjuntos_rf.csv", index=False
    )

    cv_summary, predictions_by_model, reports_by_model = evaluate_pipelines(
        compensated_df, mq_columns
    )
    cv_summary.to_csv(METRICS_DIR / "resumo_modelos.csv", index=False)

    media_df = cv_summary[cv_summary["fold"].astype(str) == "media"].sort_values(
        "balanced_accuracy", ascending=False
    )
    best_model_name = str(media_df.iloc[0]["modelo"])
    best_scores = {
        "accuracy": float(media_df.iloc[0]["accuracy"]),
        "balanced_accuracy": float(media_df.iloc[0]["balanced_accuracy"]),
        "f1_macro": float(media_df.iloc[0]["f1_macro"]),
    }

    summary_json = {
        "metadata": metadata,
        "preprocessamento": {
            "media_movel_modificada": {
                "janela": MOVING_AVERAGE_WINDOW,
                "arquivo": str((DATA_DIR / "dataset_media_movel_modificada.csv").relative_to(PROJECT_DIR)),
            },
            "compensacao_ambiental": {
                "colunas_usadas": env_columns,
                "arquivo": str((DATA_DIR / "dataset_media_movel_compensado.csv").relative_to(PROJECT_DIR)),
                "observacao": (
                    "Nao ha coluna explicita de umidade relativa; foi usada somente "
                    "a informacao ambiental disponivel."
                ),
            },
        },
        "melhor_modelo": {
            "nome": best_model_name,
            **best_scores,
        },
        "ranking_modelos": media_df.to_dict(orient="records"),
    }
    (METRICS_DIR / "resumo_modelos.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    y_true = compensated_df[TARGET_COLUMN].astype(int).to_numpy()
    best_prediction = predictions_by_model[best_model_name]
    best_matrix = confusion_matrix(y_true, best_prediction, labels=[0, 1])
    save_confusion_matrix(
        best_matrix,
        best_model_name,
        MATRICES_DIR / "matriz_confusao_melhor_modelo.png",
        MATRICES_DIR / "matriz_confusao_melhor_modelo.csv",
    )

    for model_name, prediction in predictions_by_model.items():
        matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
        safe_name = model_name.replace("/", "_")
        save_confusion_matrix(
            matrix,
            model_name,
            MATRICES_DIR / f"matriz_confusao_{safe_name}.png",
            MATRICES_DIR / f"matriz_confusao_{safe_name}.csv",
        )

    save_plots(compensated_df, mq_columns, cv_summary, feature_scores)

    best_pipeline = clone(build_pipelines()[best_model_name])
    best_pipeline.fit(compensated_df[mq_columns], compensated_df[TARGET_COLUMN].astype(int))
    joblib.dump(
        {
            "modelo": best_pipeline,
            "nome_modelo": best_model_name,
            "features": mq_columns,
            "target": TARGET_COLUMN,
            "preprocessamento_externo": [
                "media_movel_modificada",
                "compensacao_ambiental_quando_disponivel",
            ],
            "scores_validacao_cruzada": best_scores,
        },
        MODELS_DIR / "melhor_modelo_metodo_autor.joblib",
    )

    write_report(
        metadata,
        smoothing_summary,
        compensation_params,
        feature_scores,
        exhaustive_df,
        cv_summary,
        best_model_name,
        reports_by_model[best_model_name],
    )
    write_readme(best_model_name, best_scores)

    print("Metodo do autor concluido.")
    print(f"Melhor modelo: {best_model_name}")
    print(json.dumps(best_scores, indent=2))
    print(f"Relatorio: {REPORTS_DIR / 'relatorio_metodo_autor.txt'}")


if __name__ == "__main__":
    main()
