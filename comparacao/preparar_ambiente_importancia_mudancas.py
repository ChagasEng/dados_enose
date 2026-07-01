from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
FULL_DATASET = (
    ROOT
    / "dataset_processado_por_dia_vaso_sem_vref0"
    / "dataset_unico_por_dia_vaso_sem_vref0.csv"
)
REDUCED_DIA20 = ROOT / "dia_20_mais" / "dia_20_mais" / "dataset_dia_20_mais.csv"
REDUCED_ANTES = ROOT / "sem pressao" / "dataset_sem_pressao.csv"

OUT_DATASETS = ROOT / "comparacao" / "datasets_com_ambiente"
OUT_IMPORTANCE = ROOT / "comparacao" / "importancia_sensores"
OUT_CURVES = ROOT / "comparacao" / "curvas_cruas_com_ambiente"
OUT_EVENTS = ROOT / "comparacao" / "instantes_mudanca"

MQ_FEATURES = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
ENV_FEATURES = ["Soil", "Temp.", "Pres."]
ALL_FEATURES = MQ_FEATURES + ENV_FEATURES
GROUP_COLUMN = "Coleta"
TARGET = "Classe"
TRAIN_RATIO = 0.70
RANDOM_STATE = 42


def ensure_dirs() -> None:
    for path in [OUT_DATASETS, OUT_IMPORTANCE, OUT_CURVES, OUT_EVENTS]:
        path.mkdir(parents=True, exist_ok=True)


def read_full_dataset() -> pd.DataFrame:
    df = pd.read_csv(FULL_DATASET)
    df["Dia"] = pd.to_numeric(df["Dia"], errors="coerce")
    page_day_map = {
        "Página3": 20,
        "Página4": 20,
        "Página5": 20,
        "Página6": 20,
        "Página7": 20,
        "Página8": 20,
        "Página9": 20,
        "Página10": 20,
        "Página11": 20,
        "Página12": 20,
        "Página13": 21,
        "Página14": 21,
        "Página15": 21,
    }
    coleta = df["Coleta"].astype(str).str.strip()
    for page, day in page_day_map.items():
        df.loc[coleta.eq(page) & df["Dia"].isna(), "Dia"] = day
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    for column in ["Vaso", "Tempo", *ALL_FEATURES]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def write_environment_datasets(full_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["Coleta", "Dia", "Vaso", "Tempo", *ENV_FEATURES, *MQ_FEATURES, TARGET]
    antes = full_df[full_df["Dia"] < 20][columns].copy()
    dia20 = full_df[full_df["Dia"] >= 20][columns].copy()

    antes.to_csv(OUT_DATASETS / "antes_dia_20_com_ambiente.csv", index=False)
    dia20.to_csv(OUT_DATASETS / "dia_20_mais_com_ambiente.csv", index=False)
    antes[antes[TARGET] == 0].to_csv(
        OUT_DATASETS / "antes_dia_20_com_ambiente_classe_0.csv", index=False
    )
    antes[antes[TARGET] == 1].to_csv(
        OUT_DATASETS / "antes_dia_20_com_ambiente_classe_1.csv", index=False
    )
    dia20[dia20[TARGET] == 0].to_csv(
        OUT_DATASETS / "dia_20_mais_com_ambiente_classe_0.csv", index=False
    )
    dia20[dia20[TARGET] == 1].to_csv(
        OUT_DATASETS / "dia_20_mais_com_ambiente_classe_1.csv", index=False
    )
    return antes, dia20


def validate_alignment(antes: pd.DataFrame, dia20: pd.DataFrame) -> pd.DataFrame:
    checks = []
    reduced_antes = pd.read_csv(REDUCED_ANTES)
    reduced_dia20 = pd.read_csv(REDUCED_DIA20)
    compare_columns = ["Coleta", "Dia", "Vaso", *MQ_FEATURES, TARGET]

    for name, full_part, reduced_part in [
        ("antes_dia_20", antes, reduced_antes),
        ("dia_20_mais", dia20, reduced_dia20),
    ]:
        full_cmp = full_part[compare_columns].reset_index(drop=True).copy()
        reduced_cmp = reduced_part[compare_columns].reset_index(drop=True).copy()
        same_rows = len(full_cmp) == len(reduced_cmp)
        same_values = same_rows
        differences = 0
        if same_rows:
            for column in compare_columns:
                if column == "Coleta":
                    left = full_cmp[column].astype(str).fillna("__NA__")
                    right = reduced_cmp[column].astype(str).fillna("__NA__")
                    unequal = left.ne(right)
                else:
                    left = pd.to_numeric(full_cmp[column], errors="coerce")
                    right = pd.to_numeric(reduced_cmp[column], errors="coerce")
                    unequal = ~((left.eq(right)) | (left.isna() & right.isna()))
                differences += int(unequal.sum())
            same_values = differences == 0
        checks.append(
            {
                "dataset": name,
                "linhas_com_ambiente": len(full_cmp),
                "linhas_reduzido": len(reduced_cmp),
                "mesma_quantidade_linhas": same_rows,
                "mesmos_valores_colunas_originais": same_values,
                "diferencas_colunas_originais": differences if same_rows else None,
            }
        )

    result = pd.DataFrame(checks)
    result.to_csv(OUT_DATASETS / "validacao_alinhamento.csv", index=False)
    return result


def split_by_group_inside_class(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    for class_value in sorted(df[TARGET].dropna().unique()):
        class_block = df[df[TARGET] == class_value]
        groups = (
            pd.Series(class_block[GROUP_COLUMN].dropna().unique())
            .sample(frac=1, random_state=RANDOM_STATE)
            .tolist()
        )
        train_groups = set(groups[: int(len(groups) * TRAIN_RATIO)])
        train_parts.append(class_block[class_block[GROUP_COLUMN].isin(train_groups)])
        test_parts.append(class_block[~class_block[GROUP_COLUMN].isin(train_groups)])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=RANDOM_STATE)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=RANDOM_STATE)
    return train_df, test_df


def cohen_d_abs(class_0: pd.Series, class_1: pd.Series) -> float:
    std_0 = class_0.std(ddof=1)
    std_1 = class_1.std(ddof=1)
    pooled = np.sqrt(
        ((len(class_0) - 1) * std_0**2 + (len(class_1) - 1) * std_1**2)
        / (len(class_0) + len(class_1) - 2)
    )
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float(abs((class_1.mean() - class_0.mean()) / pooled))


def evaluate_importance(df: pd.DataFrame, dataset_name: str, features: list[str]) -> pd.DataFrame:
    model_df = df[[GROUP_COLUMN, *features, TARGET]].dropna().copy()
    model_df[TARGET] = model_df[TARGET].astype(int)
    train_df, test_df = split_by_group_inside_class(model_df)

    model = ExtraTreesClassifier(
        n_estimators=700,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=10,
        bootstrap=False,
    )
    model.fit(train_df[features], train_df[TARGET])
    prediction = model.predict(test_df[features])

    permutation = permutation_importance(
        model,
        test_df[features],
        test_df[TARGET],
        n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    mutual_info = mutual_info_classif(
        train_df[features], train_df[TARGET], random_state=RANDOM_STATE
    )

    rows = []
    for index, feature in enumerate(features):
        class_0 = train_df.loc[train_df[TARGET] == 0, feature].astype(float)
        class_1 = train_df.loc[train_df[TARGET] == 1, feature].astype(float)
        rows.append(
            {
                "dataset": dataset_name,
                "features_set": "mq_ambiente" if any(f in ENV_FEATURES for f in features) else "mq",
                "sensor": feature,
                "extra_trees_importance": float(model.feature_importances_[index]),
                "queda_balanced_accuracy_por_permutacao": float(
                    permutation.importances_mean[index]
                ),
                "desvio_permutacao": float(permutation.importances_std[index]),
                "mutual_info": float(mutual_info[index]),
                "media_doente_classe_0": float(class_0.mean()),
                "media_saudavel_classe_1": float(class_1.mean()),
                "diferenca_media_1_menos_0": float(class_1.mean() - class_0.mean()),
                "cohen_d_abs": cohen_d_abs(class_0, class_1),
                "accuracy_holdout": float(accuracy_score(test_df[TARGET], prediction)),
                "balanced_accuracy_holdout": float(
                    balanced_accuracy_score(test_df[TARGET], prediction)
                ),
                "f1_macro_holdout": float(f1_score(test_df[TARGET], prediction, average="macro")),
                "treino_linhas": int(len(train_df)),
                "teste_linhas": int(len(test_df)),
            }
        )
    return pd.DataFrame(rows).sort_values("extra_trees_importance", ascending=False)


def plot_importance(result_df: pd.DataFrame, output_name: str, title: str) -> None:
    plot_df = result_df.sort_values("extra_trees_importance", ascending=True)
    y = np.arange(len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharey=True)
    fig.suptitle(title, fontsize=14)

    axes[0].barh(y, plot_df["extra_trees_importance"], color="#2f6f73")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_df["sensor"])
    axes[0].set_xlabel("Importancia interna ExtraTrees")
    axes[0].grid(axis="x", alpha=0.22)

    axes[1].barh(
        y,
        plot_df["queda_balanced_accuracy_por_permutacao"],
        xerr=plot_df["desvio_permutacao"],
        color="#c7503d",
    )
    axes[1].set_xlabel("Queda na balanced accuracy por permutacao")
    axes[1].grid(axis="x", alpha=0.22)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_IMPORTANCE / output_name, dpi=180)
    plt.close(fig)


def plot_raw_curves(df: pd.DataFrame, dataset_name: str) -> None:
    features = ALL_FEATURES
    fig, axes = plt.subplots(len(features), 1, figsize=(20, 18), sharex=True)
    ordered = df.dropna(subset=[TARGET]).reset_index(drop=True).copy()
    x = np.arange(len(ordered))

    for ax, feature in zip(axes, features):
        y = ordered[feature].rolling(21, center=True, min_periods=1).mean()
        ax.plot(x, y, linewidth=0.8, color="#253858")
        ax.set_ylabel(feature)
        ax.grid(axis="x", alpha=0.14)

    boundaries = (
        ordered.groupby(["Coleta"], sort=False)
        .size()
        .cumsum()
        .shift(fill_value=0)
        .astype(int)
        .tolist()
    )
    for boundary in boundaries:
        for ax in axes:
            ax.axvline(boundary, color="#aaaaaa", linewidth=0.35, alpha=0.55)

    axes[0].set_title(f"{dataset_name} - dados crus com Soil, Temp. e Pres.")
    axes[-1].set_xlabel("Indice da linha no dataset")
    fig.tight_layout()
    fig.savefig(OUT_CURVES / f"{dataset_name}_curvas_cruas_mq_ambiente.png", dpi=180)
    plt.close(fig)


def detect_change_events(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = []
    saturation_rows = []
    features = ALL_FEATURES
    thresholds_high = {feature: df[feature].quantile(0.99) for feature in features}

    for coleta, block in df.groupby("Coleta", sort=False):
        block = block.reset_index().rename(columns={"index": "linha_global"})
        for feature in features:
            values = block[feature].astype(float)
            diff_abs = values.diff().abs()
            mad = (diff_abs - diff_abs.median()).abs().median()
            threshold = max(diff_abs.quantile(0.995), diff_abs.median() + 8 * mad)
            candidates = block.loc[diff_abs >= threshold].copy()
            candidates["delta_abs"] = diff_abs.loc[candidates.index].values
            for _, row in candidates.nlargest(10, "delta_abs").iterrows():
                event_rows.append(
                    {
                        "dataset": dataset_name,
                        "tipo": "mudanca_abrupta",
                        "Coleta": coleta,
                        "Dia": row.get("Dia"),
                        "Vaso": row.get("Vaso"),
                        "Classe": row.get(TARGET),
                        "Tempo": row.get("Tempo"),
                        "linha_global": int(row["linha_global"]),
                        "sensor": feature,
                        "valor": row[feature],
                        "delta_abs": row["delta_abs"],
                    }
                )

            high_mask = values >= thresholds_high[feature]
            if high_mask.any():
                high_indices = np.flatnonzero(high_mask.to_numpy())
                splits = np.split(high_indices, np.where(np.diff(high_indices) != 1)[0] + 1)
                for segment in splits:
                    if len(segment) < 5:
                        continue
                    start = int(segment[0])
                    end = int(segment[-1])
                    start_row = block.iloc[start]
                    end_row = block.iloc[end]
                    saturation_rows.append(
                        {
                            "dataset": dataset_name,
                            "tipo": "valor_alto_p99_intervalo",
                            "Coleta": coleta,
                            "Dia": start_row.get("Dia"),
                            "Vaso": start_row.get("Vaso"),
                            "Classe": start_row.get(TARGET),
                            "sensor": feature,
                            "linha_global_inicio": int(start_row["linha_global"]),
                            "linha_global_fim": int(end_row["linha_global"]),
                            "Tempo_inicio": start_row.get("Tempo"),
                            "Tempo_fim": end_row.get("Tempo"),
                            "duracao_linhas": int(len(segment)),
                            "valor_medio_intervalo": float(values.iloc[segment].mean()),
                            "valor_max_intervalo": float(values.iloc[segment].max()),
                            "limiar_p99": float(thresholds_high[feature]),
                        }
                    )

    events = pd.DataFrame(event_rows).sort_values(
        ["dataset", "sensor", "delta_abs"], ascending=[True, True, False]
    )
    saturations = pd.DataFrame(saturation_rows).sort_values(
        ["dataset", "sensor", "duracao_linhas"], ascending=[True, True, False]
    )
    return events, saturations


def main() -> None:
    ensure_dirs()
    full = read_full_dataset()
    antes, dia20 = write_environment_datasets(full)
    validation = validate_alignment(antes, dia20)

    importance_frames = []
    scenarios = [
        ("antes_dia_20", antes, MQ_FEATURES),
        ("antes_dia_20", antes, ALL_FEATURES),
        ("dia_20_mais", dia20, MQ_FEATURES),
        ("dia_20_mais", dia20, ALL_FEATURES),
    ]
    for dataset_name, dataset_df, features in scenarios:
        result = evaluate_importance(dataset_df, dataset_name, features)
        suffix = "mq_ambiente" if any(feature in ENV_FEATURES for feature in features) else "mq"
        result.to_csv(OUT_IMPORTANCE / f"importancia_{dataset_name}_{suffix}.csv", index=False)
        plot_importance(
            result,
            f"importancia_{dataset_name}_{suffix}.png",
            f"Importancia - {dataset_name} - {suffix}",
        )
        importance_frames.append(result)
    all_importance = pd.concat(importance_frames, ignore_index=True)
    all_importance.to_csv(OUT_IMPORTANCE / "importancia_todos_cenarios.csv", index=False)

    plot_raw_curves(antes, "antes_dia_20")
    plot_raw_curves(dia20, "dia_20_mais")

    events_antes, saturations_antes = detect_change_events(antes, "antes_dia_20")
    events_dia20, saturations_dia20 = detect_change_events(dia20, "dia_20_mais")
    pd.concat([events_antes, events_dia20], ignore_index=True).to_csv(
        OUT_EVENTS / "instantes_mudanca_top.csv", index=False
    )
    pd.concat([saturations_antes, saturations_dia20], ignore_index=True).to_csv(
        OUT_EVENTS / "intervalos_valor_alto_p99.csv", index=False
    )

    report = [
        "Resumo do processamento",
        "",
        "Datasets com ambiente criados a partir do dataset upstream com V_ref_0 removido, antes da remocao de Soil/Temp./Pres.",
        "",
        validation.to_string(index=False),
        "",
        "Arquivos principais:",
        f"- {OUT_DATASETS / 'antes_dia_20_com_ambiente.csv'}",
        f"- {OUT_DATASETS / 'dia_20_mais_com_ambiente.csv'}",
        f"- {OUT_IMPORTANCE / 'importancia_todos_cenarios.csv'}",
        f"- {OUT_EVENTS / 'instantes_mudanca_top.csv'}",
        f"- {OUT_EVENTS / 'intervalos_valor_alto_p99.csv'}",
    ]
    (ROOT / "comparacao" / "README_ambiente_importancia_mudancas.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("Validacao de alinhamento:")
    print(validation.to_string(index=False))
    print()
    print("Top importancia dia_20_mais com ambiente:")
    top = all_importance[
        (all_importance["dataset"] == "dia_20_mais")
        & (all_importance["features_set"] == "mq_ambiente")
    ].sort_values("extra_trees_importance", ascending=False)
    print(
        top[
            [
                "sensor",
                "extra_trees_importance",
                "queda_balanced_accuracy_por_permutacao",
                "balanced_accuracy_holdout",
            ]
        ]
        .head(9)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
