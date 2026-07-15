from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[1]
ROOT_06 = BASE.parent
INPUT_DATASET = (
    ROOT_06
    / "3_compensacao_umidade_temperatura"
    / "dados_base"
    / "antes_dia_20_pressao_filtrada_estrito_com_ambiente.csv"
)
SENSOR_DOC = (
    ROOT_06
    / "3_compensacao_umidade_temperatura"
    / "datasheets_calibracao"
    / "sensores_ambiente_confirmados.md"
)

TARGET = "Classe"
GROUP = "Coleta"
MQ_FEATURES = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
ENV_FEATURES_ORIG = ["Soil", "Temp.", "Pres."]
ENV_FEATURES_CONFIRMADAS = ["Soil_indice_0_1", "Temp_C", "Pres_kPa"]
TRAIN_RATIO = 0.70
RANDOM_STATE = 42
PERMUTATION_SAMPLE_SIZE = 8000


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "base": BASE,
        "dados": BASE / "dados_processados",
        "docs": BASE / "documentos",
        "scripts": BASE / "scripts",
        "metricas": BASE / "modelagem" / "metricas",
        "matrizes": BASE / "modelagem" / "matrizes",
        "importancias": BASE / "modelagem" / "importancias",
        "modelos": BASE / "modelagem" / "modelos",
        "relatorios": BASE / "modelagem" / "relatorios",
        "graficos": BASE / "graficos",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(INPUT_DATASET)
    required = [GROUP, TARGET, *MQ_FEATURES, *ENV_FEATURES_ORIG]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")

    keep = [
        column
        for column in ["Coleta", "Dia", "Vaso", "Tempo", *ENV_FEATURES_ORIG, *MQ_FEATURES, TARGET]
        if column in df.columns
    ]
    df = df[keep].copy()
    for column in [*ENV_FEATURES_ORIG, *MQ_FEATURES, TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[GROUP, TARGET, *ENV_FEATURES_ORIG, *MQ_FEATURES]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    df["Indice_original"] = np.arange(len(df))
    df["Temp_C"] = df["Temp."]
    df["Pres_kPa"] = df["Pres."]
    df["Pres_hPa"] = df["Pres."] * 10.0

    soil = df["Soil"].astype(float)
    soil_min = float(soil.min())
    soil_max = float(soil.max())
    if soil_max == soil_min:
        df["Soil_indice_0_1"] = 0.0
    else:
        df["Soil_indice_0_1"] = (soil - soil_min) / (soil_max - soil_min)

    df["Nematoide"] = np.where(df[TARGET] == 0, "Com nematoide", "Sem nematoide")
    return df


def split_indices_by_group(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    train_indices: list[int] = []
    test_indices: list[int] = []
    rows: list[dict[str, Any]] = []

    for class_value, class_block in df.groupby(TARGET):
        groups = (
            pd.Series(class_block[GROUP].dropna().unique())
            .sample(frac=1, random_state=RANDOM_STATE)
            .tolist()
        )
        train_count = max(1, int(len(groups) * TRAIN_RATIO))
        if train_count >= len(groups):
            train_count = len(groups) - 1

        train_groups = set(groups[:train_count])
        test_groups = set(groups[train_count:])

        train_mask = (df[TARGET] == class_value) & df[GROUP].isin(train_groups)
        test_mask = (df[TARGET] == class_value) & df[GROUP].isin(test_groups)
        train_indices.extend(df.index[train_mask].tolist())
        test_indices.extend(df.index[test_mask].tolist())

        train_block = df.loc[train_mask]
        test_block = df.loc[test_mask]
        rows.append(
            {
                "classe": int(class_value),
                "rotulo": "Com nematoide" if int(class_value) == 0 else "Sem nematoide",
                "linhas_total": int(len(class_block)),
                "linhas_treino": int(len(train_block)),
                "linhas_teste": int(len(test_block)),
                "coletas_total": int(len(groups)),
                "coletas_treino": int(len(train_groups)),
                "coletas_teste": int(len(test_groups)),
                "coletas_treino_lista": " | ".join(map(str, sorted(train_groups))),
                "coletas_teste_lista": " | ".join(map(str, sorted(test_groups))),
            }
        )

    train_indices_array = np.array(train_indices, dtype=int)
    test_indices_array = np.array(test_indices, dtype=int)
    return train_indices_array, test_indices_array, pd.DataFrame(rows)


def corrected_feature_names() -> list[str]:
    return [f"{sensor}_corrigido_env" for sensor in MQ_FEATURES]


def add_environment_correction(
    df: pd.DataFrame, train_indices: np.ndarray, output_dir: Path
) -> pd.DataFrame:
    corrected = df.copy()
    coef_rows: list[dict[str, Any]] = []
    env_features = ENV_FEATURES_CONFIRMADAS

    for sensor in MQ_FEATURES:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "huber",
                    HuberRegressor(
                        epsilon=1.35,
                        alpha=0.0001,
                        max_iter=400,
                    ),
                ),
            ]
        )
        model.fit(corrected.loc[train_indices, env_features], corrected.loc[train_indices, sensor])
        predicted_all = model.predict(corrected[env_features])
        predicted_train = model.predict(corrected.loc[train_indices, env_features])
        train_level = float(np.mean(predicted_train))

        corrected[f"{sensor}_efeito_ambiente"] = predicted_all - train_level
        corrected[f"{sensor}_corrigido_env"] = corrected[sensor] - corrected[f"{sensor}_efeito_ambiente"]

        scaler = model.named_steps["scaler"]
        huber = model.named_steps["huber"]
        coef_original = huber.coef_ / scaler.scale_
        intercept_original = huber.intercept_ - float(np.sum(huber.coef_ * scaler.mean_ / scaler.scale_))
        coef_rows.append(
            {
                "sensor": sensor,
                "intercept_original_units": intercept_original,
                "coef_soil_indice_0_1": coef_original[0],
                "coef_temp_c": coef_original[1],
                "coef_pres_kpa": coef_original[2],
                "nivel_medio_treino_remantido": train_level,
                "metodo": "HuberRegressor ajustado somente no treino",
            }
        )

    pd.DataFrame(coef_rows).to_csv(
        output_dir / "coeficientes_compensacao_ambiente_por_sensor.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return corrected


def build_extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=900,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=10,
        class_weight=None,
        bootstrap=False,
    )


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def classify_feature(feature: str) -> str:
    if feature in MQ_FEATURES:
        return "mq_cru"
    if feature in corrected_feature_names():
        return "mq_corrigido_ambiente"
    if feature in ENV_FEATURES_CONFIRMADAS or feature in ENV_FEATURES_ORIG:
        return "ambiente_confirmado"
    return "outro"


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    png_path: Path,
    csv_path: Path,
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["real_com_nematoide", "real_sem_nematoide"],
        columns=["previsto_com_nematoide", "previsto_sem_nematoide"],
    ).to_csv(csv_path, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Com nematoide", "Sem nematoide"],
    )
    display.plot(ax=ax, cmap="Greens", values_format="d", colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def save_importance_plot(df: pd.DataFrame, output: Path, title: str, value_column: str) -> None:
    plot_df = df.sort_values(value_column, ascending=True)
    color_map = {
        "mq_cru": "#2d6cdf",
        "mq_corrigido_ambiente": "#1f7a5b",
        "ambiente_confirmado": "#d9822b",
        "outro": "#777777",
    }
    colors = [color_map.get(kind, "#777777") for kind in plot_df["tipo"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(plot_df["feature"], plot_df[value_column], color=colors)
    ax.set_xlabel("Importancia")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_permutation(model: Any, test_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if len(test_df) > PERMUTATION_SAMPLE_SIZE:
        sample_df = test_df.sample(n=PERMUTATION_SAMPLE_SIZE, random_state=RANDOM_STATE)
    else:
        sample_df = test_df

    result = permutation_importance(
        model,
        sample_df[features],
        sample_df[TARGET],
        scoring="balanced_accuracy",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": features,
                "tipo": [classify_feature(feature) for feature in features],
                "importancia_media": result.importances_mean,
                "importancia_desvio": result.importances_std,
                "amostra_teste_usada": len(sample_df),
            }
        )
        .sort_values("importancia_media", ascending=False)
        .reset_index(drop=True)
    )


def run_scenario(
    scenario: dict[str, Any],
    df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    dirs: dict[str, Path],
) -> dict[str, Any]:
    features = scenario["features"]
    train_df = df.loc[train_indices].sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df = df.loc[test_indices].sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    model = build_extra_trees()
    model.fit(train_df[features], train_df[TARGET])
    pred = model.predict(test_df[features])
    proba = model.predict_proba(test_df[features])[:, 1]
    metrics = compute_metrics(test_df[TARGET], pred)

    joblib.dump(model, dirs["modelos"] / f"{scenario['id']}_extra_trees.joblib")
    save_confusion_matrix(
        test_df[TARGET],
        pred,
        dirs["matrizes"] / f"{scenario['id']}_matriz_confusao.png",
        dirs["matrizes"] / f"{scenario['id']}_matriz_confusao.csv",
        f"{scenario['titulo']} - ExtraTrees",
    )

    native_importance = (
        pd.DataFrame(
            {
                "feature": features,
                "tipo": [classify_feature(feature) for feature in features],
                "importancia": model.feature_importances_,
            }
        )
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    native_importance.to_csv(
        dirs["importancias"] / f"{scenario['id']}_importancia_nativa.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_importance_plot(
        native_importance,
        dirs["graficos"] / f"{scenario['id']}_importancia_nativa.png",
        f"{scenario['titulo']} - importancia nativa",
        "importancia",
    )

    permutation = run_permutation(model, test_df, features)
    permutation.to_csv(
        dirs["importancias"] / f"{scenario['id']}_importancia_permutacao.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_importance_plot(
        permutation,
        dirs["graficos"] / f"{scenario['id']}_importancia_permutacao.png",
        f"{scenario['titulo']} - importancia por permutacao",
        "importancia_media",
    )

    predictions = test_df[[column for column in ["Coleta", "Dia", "Vaso", "Tempo", TARGET] if column in test_df.columns]].copy()
    predictions["predito"] = pred
    predictions["prob_sem_nematoide"] = proba
    predictions["acertou"] = predictions[TARGET].to_numpy() == pred
    predictions.to_csv(
        dirs["metricas"] / f"{scenario['id']}_predicoes_teste.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_payload = {
        "cenario": scenario["id"],
        "titulo": scenario["titulo"],
        "descricao": scenario["descricao"],
        "dataset": str(INPUT_DATASET.relative_to(ROOT_06)),
        "features": features,
        "linhas_dataset": int(len(df)),
        "linhas_treino": int(len(train_df)),
        "linhas_teste": int(len(test_df)),
        "coletas_total": int(df[GROUP].nunique()),
        "split": "70/30 por grupos de Coleta dentro de cada classe",
        "extra_trees": metrics,
        "top5_importancia_nativa": native_importance.head(5).to_dict(orient="records"),
        "top5_importancia_permutacao": permutation.head(5).to_dict(orient="records"),
    }
    with (dirs["metricas"] / f"{scenario['id']}_metricas.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2, ensure_ascii=False)

    return {
        "cenario": scenario["id"],
        "titulo": scenario["titulo"],
        "features": ", ".join(features),
        "linhas_treino": int(len(train_df)),
        "linhas_teste": int(len(test_df)),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1_macro": metrics["f1_macro"],
        "top1_nativo": native_importance.iloc[0]["feature"],
        "top1_permutacao": permutation.iloc[0]["feature"],
    }


def smooth(series: pd.Series, window: int = 35) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).median()


def ordered_for_collection_plot(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    blocks: list[dict[str, Any]] = []
    cursor = 0
    count = 1

    for class_value, class_block in df.groupby(TARGET, sort=True):
        for coleta, coleta_df in class_block.groupby(GROUP, sort=False):
            subset = coleta_df.copy()
            label = f"C{count}"
            subset["Coleta_plot"] = label
            start = cursor
            end = cursor + len(subset) - 1
            blocks.append(
                {
                    "Coleta_plot": label,
                    "Coleta": coleta,
                    "Classe": int(class_value),
                    "Nematoide": "Com nematoide" if int(class_value) == 0 else "Sem nematoide",
                    "inicio": start,
                    "fim": end,
                    "linhas": int(len(subset)),
                }
            )
            cursor = end + 1
            count += 1
            parts.append(subset)

    ordered = pd.concat(parts, ignore_index=True)
    return ordered, pd.DataFrame(blocks)


def shade_collections(axes: list[plt.Axes], blocks: pd.DataFrame) -> None:
    class_colors = {0: "#e57373", 1: "#6ab187"}
    class_fill = {0: "#f7d8d5", 1: "#dcefe3"}
    for _, row in blocks.iterrows():
        cls = int(row["Classe"])
        for ax in axes:
            ax.axvspan(row["inicio"], row["fim"], color=class_fill[cls], alpha=0.62, linewidth=0)
            ax.axvline(row["inicio"], color=class_colors[cls], linewidth=0.8, alpha=0.72)
        axes[0].text(
            (row["inicio"] + row["fim"]) / 2,
            1.012,
            row["Coleta_plot"],
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#b33a3a" if cls == 0 else "#2f7d51",
            fontweight="bold",
        )


def save_collection_map(blocks: pd.DataFrame, output: Path) -> None:
    blocks.to_csv(output, index=False, encoding="utf-8-sig")


def plot_collection_raw(df: pd.DataFrame, output: Path) -> None:
    ordered, blocks = ordered_for_collection_plot(df)
    panels = ["Soil", "Temp_C", "Pres_kPa", *MQ_FEATURES]
    fig, axes = plt.subplots(len(panels), 1, figsize=(18, 20), sharex=True)
    axes_list = list(axes)
    shade_collections(axes_list, blocks)

    x = np.arange(len(ordered))
    for ax, feature in zip(axes_list, panels):
        ax.plot(x, smooth(ordered[feature]), color="#34495e", linewidth=0.85)
        ax.set_ylabel(feature, fontsize=9)
        ax.grid(axis="y", alpha=0.16)
        ax.set_xlim(0, len(ordered))

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#f7d8d5", alpha=0.8, label="Com nematoide"),
        plt.Rectangle((0, 0), 1, 1, color="#dcefe3", alpha=0.8, label="Sem nematoide"),
    ]
    axes_list[0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Sinais crus com sensores ambientais confirmados e coletas demarcadas", y=0.996)
    axes_list[-1].set_xlabel("Indice da linha no dataset")
    fig.tight_layout(rect=[0, 0, 1, 0.982])
    fig.savefig(output, dpi=180)
    plt.close(fig)
    save_collection_map(blocks, output.with_name("mapa_coletas_nematoide_sensores_confirmados.csv"))


def plot_collection_corrected(df: pd.DataFrame, output: Path) -> None:
    ordered, blocks = ordered_for_collection_plot(df)
    panels = ["Soil_indice_0_1", "Temp_C", "Pres_kPa", *MQ_FEATURES]
    fig, axes = plt.subplots(len(panels), 1, figsize=(18, 20), sharex=True)
    axes_list = list(axes)
    shade_collections(axes_list, blocks)

    x = np.arange(len(ordered))
    for ax, feature in zip(axes_list[:3], panels[:3]):
        ax.plot(x, smooth(ordered[feature]), color="#34495e", linewidth=0.85)
        ax.set_ylabel(feature, fontsize=9)
        ax.grid(axis="y", alpha=0.16)
        ax.set_xlim(0, len(ordered))

    for ax, sensor in zip(axes_list[3:], MQ_FEATURES):
        corrected = f"{sensor}_corrigido_env"
        ax.plot(x, smooth(ordered[sensor]), color="#9aa0a6", linewidth=0.65, alpha=0.62, label="cru")
        ax.plot(x, smooth(ordered[corrected]), color="#1f5f8b", linewidth=0.9, label="corrigido")
        ax.set_ylabel(sensor, fontsize=9)
        ax.grid(axis="y", alpha=0.16)
        ax.set_xlim(0, len(ordered))

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#f7d8d5", alpha=0.8, label="Com nematoide"),
        plt.Rectangle((0, 0), 1, 1, color="#dcefe3", alpha=0.8, label="Sem nematoide"),
        plt.Line2D([0], [0], color="#9aa0a6", linewidth=1.4, label="MQ cru"),
        plt.Line2D([0], [0], color="#1f5f8b", linewidth=1.4, label="MQ corrigido"),
    ]
    axes_list[0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Coletas por nematoide com compensacao estatistica dos MQ por ambiente", y=0.996)
    axes_list[-1].set_xlabel("Indice da linha no dataset")
    fig.tight_layout(rect=[0, 0, 1, 0.982])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def robust_z(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or np.isnan(mad):
        std = series.std()
        return (series - series.mean()) / std if std else series * 0
    return (series - median) / (1.4826 * mad)


def plot_collection_corrected_zscore(df: pd.DataFrame, output: Path) -> None:
    ordered, blocks = ordered_for_collection_plot(df)
    fig, ax = plt.subplots(figsize=(18, 9))
    shade_collections([ax], blocks)
    x = np.arange(len(ordered))

    offsets = np.arange(len(MQ_FEATURES)) * 3.0
    for offset, sensor in zip(offsets, MQ_FEATURES):
        corrected = f"{sensor}_corrigido_env"
        y = smooth(robust_z(ordered[corrected]).clip(-2.5, 2.5), window=45) + offset
        ax.plot(x, y, linewidth=0.9, label=sensor)
        ax.axhline(offset, color="#d9d9d9", linewidth=0.55)

    ax.set_yticks(offsets, MQ_FEATURES)
    ax.set_xlabel("Indice da linha no dataset")
    ax.set_ylabel("MQ corrigido por ambiente em z-score robusto + deslocamento")
    fig.suptitle("MQ corrigidos e normalizados para enxergar mudancas por coleta", y=0.995)
    ax.grid(axis="x", alpha=0.15)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#f7d8d5", alpha=0.8, label="Com nematoide"),
        plt.Rectangle((0, 0), 1, 1, color="#dcefe3", alpha=0.8, label="Sem nematoide"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_metric_comparison(summary_df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["accuracy"] * 100, color="#2d6cdf", label="Accuracy")
    ax.plot(x, summary_df["balanced_accuracy"] * 100, marker="o", color="#d9822b", label="Balanced accuracy")
    ax.set_xticks(x, summary_df["titulo"], rotation=14, ha="right")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("ExtraTrees - comparacao dos cenarios com sensores confirmados")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_correlation_diagnostic(df: pd.DataFrame, output: Path, csv_output: Path) -> None:
    env = ENV_FEATURES_CONFIRMADAS
    corr_rows = []
    for sensor in MQ_FEATURES:
        for env_feature in env:
            corr_rows.append(
                {
                    "sensor": sensor,
                    "versao": "cru",
                    "ambiente": env_feature,
                    "correlacao": float(df[sensor].corr(df[env_feature])),
                }
            )
            corr_rows.append(
                {
                    "sensor": sensor,
                    "versao": "corrigido",
                    "ambiente": env_feature,
                    "correlacao": float(df[f"{sensor}_corrigido_env"].corr(df[env_feature])),
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(csv_output, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, constrained_layout=True)
    for ax, version in zip(axes, ["cru", "corrigido"]):
        pivot = (
            corr_df[corr_df["versao"] == version]
            .pivot(index="ambiente", columns="sensor", values="correlacao")
            .reindex(index=env, columns=MQ_FEATURES)
        )
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(MQ_FEATURES)), MQ_FEATURES, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(env)), env)
        ax.set_title(f"Correlacao ambiente x MQ - {version}")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.025, label="correlacao de Pearson")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_readme(summary_df: pd.DataFrame, dirs: dict[str, Path]) -> None:
    best = summary_df.sort_values(["accuracy", "balanced_accuracy"], ascending=False).iloc[0]
    table_rows = [
        "| cenario | accuracy | balanced_accuracy | f1_macro | top1_nativo | top1_permutacao |",
        "|---|---:|---:|---:|---|---|",
    ]
    for _, row in summary_df.iterrows():
        table_rows.append(
            "| "
            f"{row['cenario']} | "
            f"{row['accuracy']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | "
            f"{row['f1_macro']:.4f} | "
            f"{row['top1_nativo']} | "
            f"{row['top1_permutacao']} |"
        )
    lines = [
        "# ExtraTrees com sensores ambientais confirmados",
        "",
        "Rodada criada depois de confirmar o hardware ambiental:",
        "",
        "- `Temp.` e `Pres.`: BMP280.",
        "- `Soil`: Capacitive Soil Moisture Sensor V2.0.",
        "",
        "## Base usada",
        "",
        f"`{INPUT_DATASET.relative_to(ROOT_06)}`",
        "",
        "A base ja estava com corte estrito por pressao. Nesta rodada foram criadas colunas interpretadas:",
        "",
        "- `Temp_C`: temperatura do BMP280.",
        "- `Pres_kPa`: pressao em kPa, mantendo a escala 93.x do dataset.",
        "- `Pres_hPa`: pressao convertida para hPa.",
        "- `Soil_indice_0_1`: normalizacao operacional do sensor capacitivo de solo.",
        "",
        "## Correcao aplicada",
        "",
        "Como o BMP280 nao mede umidade relativa do ar, nao foi aplicada uma correcao completa de datasheet por RH. A correcao feita aqui e estatistica: para cada MQ, ajustei no treino um HuberRegressor usando `Soil_indice_0_1`, `Temp_C` e `Pres_kPa`, removendo do MQ a componente linear associada ao ambiente. Isso reduz efeito ambiental sem fingir que `Soil` e RH.",
        "",
        "## Resultados ExtraTrees",
        "",
        *table_rows,
        "",
        f"Melhor cenario por accuracy: `{best['cenario']}` com accuracy `{best['accuracy']:.4f}`.",
        "",
        "## Arquivos principais",
        "",
        "- `dados_processados/dataset_sensores_confirmados_com_correcoes.csv`",
        "- `modelagem/metricas/resumo_extratrees_sensores_confirmados.csv`",
        "- `modelagem/importancias/`",
        "- `modelagem/matrizes/`",
        "- `graficos/01_coletas_por_nematoide_sinais_crus_confirmados.png`",
        "- `graficos/02_coletas_por_nematoide_sinais_corrigidos_overlay.png`",
        "- `graficos/03_coletas_por_nematoide_mq_corrigidos_zscore.png`",
        "- `graficos/04_correlacao_ambiente_antes_depois_correcao.png`",
        "- `graficos/05_comparacao_metricas_extratrees.png`",
        "",
    ]
    (dirs["base"] / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    dirs = ensure_dirs()
    if SENSOR_DOC.exists():
        shutil.copy2(SENSOR_DOC, dirs["docs"] / SENSOR_DOC.name)

    df = load_dataset()
    train_indices, test_indices, split_df = split_indices_by_group(df)
    df["Conjunto"] = "fora_split"
    df.loc[train_indices, "Conjunto"] = "Treino"
    df.loc[test_indices, "Conjunto"] = "Teste"

    df = add_environment_correction(df, train_indices, dirs["dados"])
    df.to_csv(
        dirs["dados"] / "dataset_sensores_confirmados_com_correcoes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    split_df.to_csv(
        dirs["metricas"] / "split_70_30_por_coleta.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scenarios = [
        {
            "id": "01_mq_cru",
            "titulo": "MQ cru",
            "features": MQ_FEATURES,
            "descricao": "Somente MQ, depois do corte de pressao.",
        },
        {
            "id": "02_mq_ambiente_confirmado",
            "titulo": "MQ + BMP280 + solo capacitivo",
            "features": [*MQ_FEATURES, *ENV_FEATURES_CONFIRMADAS],
            "descricao": "MQ cru com ambiente confirmado: Soil capacitivo, Temp BMP280 e Pres BMP280.",
        },
        {
            "id": "03_mq_corrigido_ambiente",
            "titulo": "MQ corrigido por ambiente",
            "features": corrected_feature_names(),
            "descricao": "MQ com componente linear de Soil/Temp/Pres removida a partir do treino.",
        },
        {
            "id": "04_mq_corrigido_ambiente_com_contexto",
            "titulo": "MQ corrigido + ambiente",
            "features": [*corrected_feature_names(), *ENV_FEATURES_CONFIRMADAS],
            "descricao": "MQ corrigido mantendo variaveis ambientais confirmadas como contexto.",
        },
    ]

    summary_rows = []
    for scenario in scenarios:
        print(f"Rodando {scenario['id']}")
        summary_rows.append(run_scenario(scenario, df, train_indices, test_indices, dirs))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        dirs["metricas"] / "resumo_extratrees_sensores_confirmados.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_collection_raw(
        df,
        dirs["graficos"] / "01_coletas_por_nematoide_sinais_crus_confirmados.png",
    )
    plot_collection_corrected(
        df,
        dirs["graficos"] / "02_coletas_por_nematoide_sinais_corrigidos_overlay.png",
    )
    plot_collection_corrected_zscore(
        df,
        dirs["graficos"] / "03_coletas_por_nematoide_mq_corrigidos_zscore.png",
    )
    save_correlation_diagnostic(
        df,
        dirs["graficos"] / "04_correlacao_ambiente_antes_depois_correcao.png",
        dirs["metricas"] / "correlacao_ambiente_antes_depois_correcao.csv",
    )
    save_metric_comparison(
        summary_df,
        dirs["graficos"] / "05_comparacao_metricas_extratrees.png",
    )
    save_readme(summary_df, dirs)

    print(summary_df.to_string(index=False))
    print(f"Arquivos em: {BASE}")


if __name__ == "__main__":
    main()
