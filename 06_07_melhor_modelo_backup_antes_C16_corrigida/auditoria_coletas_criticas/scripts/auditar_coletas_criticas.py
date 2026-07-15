from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[2]
AUDIT = BASE / "auditoria_coletas_criticas"
DATASET = BASE / "dados" / "dataset_melhor_modelo_sensores_corrigidos.csv"
OUT_DATA = AUDIT / "dados_auditoria"
OUT_MODEL = AUDIT / "auditoria_modelo"
OUT_PLOTS = AUDIT / "graficos"

RAW_MQ = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
CORR_MQ = [f"{sensor}_corrigido_env" for sensor in RAW_MQ]
ENV = ["Soil_indice_0_1", "Temp_C", "Pres_kPa"]
TARGET = "Classe"
GROUP = "Coleta"
CRITICAL = [15, 16, 17, 24, 28, 29, 30, 31, 32]
RANDOM_STATE = 42


def ensure_dirs() -> None:
    for path in [AUDIT, OUT_DATA, OUT_MODEL, OUT_PLOTS]:
        path.mkdir(parents=True, exist_ok=True)


def collection_map(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    rows = []
    lookup: dict[int, str] = {}
    for number, ((classe, coleta), block) in enumerate(
        df.groupby([TARGET, GROUP], sort=False), start=1
    ):
        label = f"C{number}"
        lookup[number] = coleta
        rows.append(
            {
                "C": label,
                "numero": number,
                "Classe": int(classe),
                "Nematoide": "Com nematoide" if int(classe) == 0 else "Sem nematoide",
                "Coleta": coleta,
                "linhas": len(block),
                "Conjunto": ",".join(sorted(block["Conjunto"].unique())),
                "Indice_original_min": int(block["Indice_original"].min()),
                "Indice_original_max": int(block["Indice_original"].max()),
            }
        )
    return pd.DataFrame(rows), lookup


def get_collection(df: pd.DataFrame, number: int) -> pd.DataFrame:
    groups = list(df.groupby([TARGET, GROUP], sort=False))
    return groups[number - 1][1].reset_index(drop=True)


def sha256_frame(df: pd.DataFrame, columns: list[str]) -> str:
    values = pd.util.hash_pandas_object(df[columns].reset_index(drop=True), index=False).values
    return hashlib.sha256(values.tobytes()).hexdigest()


def audit_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    a = get_collection(df, 15)
    b = get_collection(df, 16)
    columns = ["Tempo", *ENV, *RAW_MQ, *CORR_MQ, TARGET]
    rows = []
    for column in columns:
        av = pd.to_numeric(a[column], errors="coerce").to_numpy()
        bv = pd.to_numeric(b[column], errors="coerce").to_numpy()
        equal = bool(np.array_equal(av, bv, equal_nan=True))
        max_diff = float(np.nanmax(np.abs(av - bv))) if len(av) == len(bv) else np.nan
        rows.append(
            {
                "coluna": column,
                "linhas_C15": len(a),
                "linhas_C16": len(b),
                "identica": equal,
                "diferenca_max_abs": max_diff,
                "hash_C15": sha256_frame(a, [column]),
                "hash_C16": sha256_frame(b, [column]),
            }
        )
    return pd.DataFrame(rows)


def global_jump_thresholds(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    thresholds = {}
    for feature in features:
        diffs = [
            block[feature].astype(float).diff().abs().dropna()
            for _, block in df.groupby([TARGET, GROUP], sort=False)
        ]
        thresholds[feature] = float(pd.concat(diffs).quantile(0.995))
    return thresholds


def audit_events(
    df: pd.DataFrame, raw_thresholds: dict[str, float], corr_thresholds: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = []
    summary_rows = []
    for number in CRITICAL:
        block = get_collection(df, number)
        raw_flags = pd.DataFrame(
            {
                sensor: block[sensor].diff().abs() >= raw_thresholds[sensor]
                for sensor in RAW_MQ
            }
        )
        corr_flags = pd.DataFrame(
            {
                sensor: block[f"{sensor}_corrigido_env"].diff().abs()
                >= corr_thresholds[f"{sensor}_corrigido_env"]
                for sensor in RAW_MQ
            }
        )

        for kind, features, thresholds, flags in [
            ("cru", RAW_MQ, raw_thresholds, raw_flags),
            ("corrigido", CORR_MQ, corr_thresholds, corr_flags),
        ]:
            for feature in features:
                sensor = feature.replace("_corrigido_env", "")
                delta = block[feature].astype(float).diff()
                threshold = thresholds[feature]
                positions = np.flatnonzero((delta.abs() >= threshold).fillna(False).to_numpy())
                for position in positions:
                    event_rows.append(
                        {
                            "C": f"C{number}",
                            "Coleta": block[GROUP].iloc[0],
                            "Classe": int(block[TARGET].iloc[0]),
                            "tipo_sinal": kind,
                            "sensor": sensor,
                            "posicao_local": int(position),
                            "posicao_percentual": float(position / len(block) * 100),
                            "Indice_original": int(block["Indice_original"].iloc[position]),
                            "Tempo": float(block["Tempo"].iloc[position]),
                            "valor": float(block[feature].iloc[position]),
                            "salto": float(delta.iloc[position]),
                            "salto_abs": float(abs(delta.iloc[position])),
                            "limiar_global_p995": threshold,
                            "mqs_simultaneos": int(flags.iloc[position].sum()),
                            "delta_soil": float(block["Soil_indice_0_1"].diff().iloc[position]),
                            "delta_temp_c": float(block["Temp_C"].diff().iloc[position]),
                            "delta_pres_kpa": float(block["Pres_kPa"].diff().iloc[position]),
                        }
                    )

        raw_count = int(raw_flags.sum().sum())
        corr_count = int(corr_flags.sum().sum())
        summary_rows.append(
            {
                "C": f"C{number}",
                "Coleta": block[GROUP].iloc[0],
                "Classe": int(block[TARGET].iloc[0]),
                "Conjunto": block["Conjunto"].iloc[0],
                "linhas": len(block),
                "eventos_mq_cru": raw_count,
                "eventos_mq_corrigido": corr_count,
                "linhas_multissensor_cru": int((raw_flags.sum(axis=1) >= 2).sum()),
                "linhas_multissensor_corrigido": int((corr_flags.sum(axis=1) >= 2).sum()),
                "maior_salto_cru": float(
                    max(block[sensor].diff().abs().max() for sensor in RAW_MQ)
                ),
                "maior_salto_corrigido": float(
                    max(block[feature].diff().abs().max() for feature in CORR_MQ)
                ),
                "amplitude_soil": float(block["Soil_indice_0_1"].max() - block["Soil_indice_0_1"].min()),
                "amplitude_temp_c": float(block["Temp_C"].max() - block["Temp_C"].min()),
                "amplitude_pres_kpa": float(block["Pres_kPa"].max() - block["Pres_kPa"].min()),
            }
        )
    return pd.DataFrame(event_rows), pd.DataFrame(summary_rows)


def partial_spearman_time(y: pd.Series, x: pd.Series) -> float:
    n = len(y)
    t = np.linspace(0, 1, n)
    control = np.column_stack([np.ones(n), t, t * t])
    yr = y.rank().to_numpy(float)
    xr = x.rank().to_numpy(float)
    y_res = yr - control @ np.linalg.lstsq(control, yr, rcond=None)[0]
    x_res = xr - control @ np.linalg.lstsq(control, xr, rcond=None)[0]
    if np.std(y_res) == 0 or np.std(x_res) == 0:
        return np.nan
    return float(np.corrcoef(y_res, x_res)[0, 1])


def audit_environment(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for number in CRITICAL:
        block = get_collection(df, number)
        for sensor in RAW_MQ:
            for env in ENV:
                for kind, feature in [
                    ("cru", sensor),
                    ("corrigido", f"{sensor}_corrigido_env"),
                ]:
                    corr = spearmanr(block[feature], block[env], nan_policy="omit").statistic
                    rows.append(
                        {
                            "C": f"C{number}",
                            "Coleta": block[GROUP].iloc[0],
                            "sensor": sensor,
                            "tipo_sinal": kind,
                            "ambiente": env,
                            "spearman": float(corr),
                            "spearman_parcial_controlando_tempo": partial_spearman_time(
                                block[feature], block[env]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def add_environment_correction(
    df: pd.DataFrame, train_mask: pd.Series, env_features: list[str] | None = None
) -> pd.DataFrame:
    corrected = df.copy()
    correction_features = ENV if env_features is None else env_features
    for sensor in RAW_MQ:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("huber", HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=400)),
            ]
        )
        model.fit(
            corrected.loc[train_mask, correction_features],
            corrected.loc[train_mask, sensor],
        )
        predicted_all = model.predict(corrected[correction_features])
        predicted_train = model.predict(corrected.loc[train_mask, correction_features])
        corrected[f"{sensor}_corrigido_env"] = corrected[sensor] - (
            predicted_all - float(np.mean(predicted_train))
        )
    return corrected


def build_model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=900,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=10,
        class_weight=None,
        bootstrap=False,
    )


def metric_row(name: str, y_true: pd.Series, prediction: np.ndarray, **extra) -> dict:
    return {
        "cenario": name,
        "linhas_avaliadas": len(y_true),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1_macro": float(f1_score(y_true, prediction, average="macro")),
        **extra,
    }


def audit_model(df: pd.DataFrame, lookup: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_train = df["Conjunto"].eq("Treino")
    test_mask = df["Conjunto"].eq("Teste")
    features = [*CORR_MQ, *ENV]
    drop_variants = {
        "baseline_recalculada": [],
        "sem_C16_duplicada": [16],
        "sem_C17_ruidosa": [17],
        "sem_C29_C31_soil_treino": [29, 31],
        "sem_C15_C16_duplicatas": [15, 16],
        "sem_criticas_do_treino": [16, 17, 29, 31],
    }
    sensitivity_rows = []
    baseline_corrected = None
    baseline_model = None
    for name, numbers in drop_variants.items():
        dropped_names = [lookup[number] for number in numbers]
        train_mask = base_train & ~df[GROUP].isin(dropped_names)
        corrected = add_environment_correction(df, train_mask)
        model = build_model()
        model.fit(corrected.loc[train_mask, features], corrected.loc[train_mask, TARGET])
        prediction = model.predict(corrected.loc[test_mask, features])
        sensitivity_rows.append(
            metric_row(
                name,
                corrected.loc[test_mask, TARGET],
                prediction,
                linhas_treino=int(train_mask.sum()),
                coletas_removidas=" | ".join(f"C{n}" for n in numbers),
            )
        )
        if name == "baseline_recalculada":
            baseline_corrected = corrected
            baseline_model = model

    assert baseline_corrected is not None and baseline_model is not None
    y_test = baseline_corrected.loc[test_mask, TARGET]
    pred = baseline_model.predict(baseline_corrected.loc[test_mask, features])
    probs = baseline_model.predict_proba(baseline_corrected.loc[test_mask, features])[:, 1]
    test_predictions = baseline_corrected.loc[test_mask, [GROUP, TARGET]].copy()
    test_predictions["predito"] = pred
    test_predictions["prob_sem_nematoide"] = probs
    test_predictions["acertou"] = test_predictions[TARGET].to_numpy() == pred
    per_group = (
        test_predictions.groupby([TARGET, GROUP], sort=False)
        .agg(
            linhas=("acertou", "size"),
            accuracy=("acertou", "mean"),
            prob_sem_nematoide_media=("prob_sem_nematoide", "mean"),
            prob_sem_nematoide_desvio=("prob_sem_nematoide", "std"),
        )
        .reset_index()
    )
    reverse = {name: f"C{number}" for number, name in lookup.items()}
    per_group.insert(0, "C", per_group[GROUP].map(reverse))

    exclusion_rows = [metric_row("teste_completo", y_test, pred, coletas_excluidas="")]
    for numbers in [[24], [28], [30], [32], [24, 28, 30], [24, 28, 30, 32]]:
        excluded = [lookup[number] for number in numbers]
        keep = ~test_predictions[GROUP].isin(excluded)
        exclusion_rows.append(
            metric_row(
                "teste_sem_" + "_".join(f"C{n}" for n in numbers),
                test_predictions.loc[keep, TARGET],
                test_predictions.loc[keep, "predito"].to_numpy(),
                coletas_excluidas=" | ".join(f"C{n}" for n in numbers),
            )
        )

    ablations = {
        "corrigido_mais_ambiente": ENV,
        "recompensado_sem_soil": ["Temp_C", "Pres_kPa"],
        "recompensado_sem_temp": ["Soil_indice_0_1", "Pres_kPa"],
        "recompensado_sem_pressao": ["Soil_indice_0_1", "Temp_C"],
    }
    ablation_rows = []
    for name, environment_columns in ablations.items():
        ablation_df = add_environment_correction(
            df, base_train, env_features=environment_columns
        )
        columns = [*CORR_MQ, *environment_columns]
        model = build_model()
        model.fit(
            ablation_df.loc[base_train, columns],
            ablation_df.loc[base_train, TARGET],
        )
        prediction = model.predict(ablation_df.loc[test_mask, columns])
        ablation_rows.append(
            metric_row(
                name,
                y_test,
                prediction,
                variaveis_usadas_na_compensacao=" | ".join(environment_columns),
                features=" | ".join(columns),
            )
        )
    for name, columns in [
        ("somente_mq_corrigido", CORR_MQ),
        ("somente_mq_cru", RAW_MQ),
    ]:
        model = build_model()
        model.fit(
            baseline_corrected.loc[base_train, columns],
            baseline_corrected.loc[base_train, TARGET],
        )
        prediction = model.predict(baseline_corrected.loc[test_mask, columns])
        ablation_rows.append(
            metric_row(
                name,
                y_test,
                prediction,
                variaveis_usadas_na_compensacao=(" | ".join(ENV) if "corrigido" in name else "nenhuma"),
                features=" | ".join(columns),
            )
        )
    return (
        pd.DataFrame(sensitivity_rows),
        per_group,
        pd.DataFrame(exclusion_rows),
        pd.DataFrame(ablation_rows),
    )


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = values.std()
    return (values - values.mean()) / std if std > 0 else values * 0


def plot_collection(df: pd.DataFrame, number: int) -> None:
    block = get_collection(df, number)
    x = np.arange(len(block))
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    palettes = ["tab:green", "tab:orange", "tab:blue"]
    for feature, color in zip(ENV, palettes):
        axes[0].plot(x, zscore(block[feature]).rolling(9, center=True, min_periods=1).median(), label=feature, linewidth=0.9, color=color)
    for feature in RAW_MQ:
        axes[1].plot(x, zscore(block[feature]).rolling(9, center=True, min_periods=1).median(), label=feature, linewidth=0.75)
    for feature in CORR_MQ:
        axes[2].plot(x, zscore(block[feature]).rolling(9, center=True, min_periods=1).median(), label=feature.replace("_corrigido_env", ""), linewidth=0.75)
    axes[0].set_title(f"C{number} - {block[GROUP].iloc[0]} - ambiente normalizado")
    axes[1].set_title("MQ crus normalizados")
    axes[2].set_title("MQ corrigidos normalizados")
    for ax in axes:
        ax.axhline(0, color="#444444", linewidth=0.5, alpha=0.4)
        ax.grid(alpha=0.15)
        ax.legend(loc="upper right", ncol=3, fontsize=7)
        ax.set_ylabel("z-score")
    axes[-1].set_xlabel("Indice local da coleta")
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / f"C{number:02d}_auditoria.png", dpi=170)
    plt.close(fig)


def plot_model_sensitivity(sensitivity: pd.DataFrame, exclusions: pd.DataFrame, ablations: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 14))
    for ax, data, label in [
        (axes[0], sensitivity, "Retreinamento removendo coletas criticas do treino"),
        (axes[1], exclusions, "Metricas do teste ao excluir coletas criticas do calculo"),
        (axes[2], ablations, "Ablacao das variaveis ambientais"),
    ]:
        ax.barh(data["cenario"], data["balanced_accuracy"] * 100, color="#356a8a")
        ax.set_xlabel("Balanced accuracy (%)")
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.2)
        for pos, value in enumerate(data["balanced_accuracy"] * 100):
            ax.text(value + 0.2, pos, f"{value:.2f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "sensibilidade_modelo_coletas_criticas.png", dpi=180)
    plt.close(fig)


def write_report(
    duplicate: pd.DataFrame,
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    sensitivity: pd.DataFrame,
    per_group: pd.DataFrame,
    exclusions: pd.DataFrame,
    ablations: pd.DataFrame,
) -> None:
    duplicate_ok = bool(duplicate["identica"].all())
    base = sensitivity.iloc[0]
    full_test = exclusions.iloc[0]
    critical_test = per_group[per_group["C"].isin(["C24", "C28", "C30", "C32"])]
    corr_critical = correlations[correlations["tipo_sinal"].eq("corrigido")]
    strongest = corr_critical.loc[
        corr_critical["spearman_parcial_controlando_tempo"].abs().idxmax()
    ]
    lines = [
        "# Auditoria das coletas criticas",
        "",
        "## Escopo",
        "",
        "Coletas auditadas: C15, C16, C17, C24, C28, C29, C30, C31 e C32.",
        "Foram verificados sinais crus, sinais corrigidos, variaveis ambientais, eventos abruptos e impacto no ExtraTrees.",
        "",
        "## Resultado executivo",
        "",
        f"- C15 e C16 identicas em todas as colunas verificadas: `{duplicate_ok}`.",
        f"- Baseline recalculada no mesmo split: accuracy `{base['accuracy']:.4f}`; balanced accuracy `{base['balanced_accuracy']:.4f}`.",
        f"- Baseline do teste completo na auditoria: balanced accuracy `{full_test['balanced_accuracy']:.4f}`.",
        f"- Maior associacao ambiental residual nas coletas criticas: `{strongest['C']}` / `{strongest['sensor']}` x `{strongest['ambiente']}` = `{strongest['spearman_parcial_controlando_tempo']:.3f}`.",
        "",
        "## Decisao por coleta",
        "",
        "- `C15/C16`: duplicacao confirmada. Manter somente uma delas em qualquer treino/validacao.",
        "- `C17`: quarentena. Ruido multissensor extremo sem variacao ambiental proporcional.",
        "- `C24`: revisar canal MQ138 e log operacional; nao remover automaticamente antes de conferir a origem.",
        "- `C28`: quarentena provisoria. Ha degrau multissensor e forte instabilidade em Soil.",
        "- `C29-C31`: manter para auditoria biologica, mas repetir modelagem sem Soil e com validacao por coleta/dia.",
        "- `C32`: quarentena. Ruido multissensor extremo sem correspondencia ambiental.",
        "",
        "## Eventos e ambiente",
        "",
    ]
    for row in summary.itertuples():
        lines.append(
            f"- `{row.C}`: {row.eventos_mq_cru} eventos nos MQ crus; "
            f"{row.eventos_mq_corrigido} nos corrigidos; {row.linhas_multissensor_cru} linhas multissensor; "
            f"amplitudes ambiente Soil={row.amplitude_soil:.3f}, Temp={row.amplitude_temp_c:.2f} C, Pres={row.amplitude_pres_kpa:.3f} kPa."
        )
    lines.extend(
        [
            "",
            "## Coletas criticas presentes no teste original",
            "",
        ]
    )
    for row in critical_test.itertuples():
        lines.append(f"- `{row.C}`: accuracy por linha `{row.accuracy:.4f}` em `{row.linhas}` linhas.")
    lines.extend(
        [
            "",
            "C24, C28 e C30 tiveram 100% de acerto por linha no teste original. Como possuem artefatos ou forte assinatura ambiental, elas podem tornar o teste artificialmente facil. C32 teve desempenho inferior e adiciona ruido ao teste.",
            "",
            "## Sensibilidade ao remover coletas do treino",
            "",
        ]
    )
    for row in sensitivity.itertuples():
        lines.append(
            f"- `{row.cenario}`: accuracy `{row.accuracy:.4f}`; balanced accuracy `{row.balanced_accuracy:.4f}`; treino `{row.linhas_treino}` linhas."
        )
    lines.extend(
        [
            "",
            "Retirar apenas C16 quase nao altera o resultado, portanto uma das duplicatas pode ser removida com seguranca. Retirar C17 melhora a accuracy, confirmando que seu ruido prejudica o treino. Retirar C29 e C31 derruba fortemente o resultado porque elas sao as coletas saudaveis do dia 13 presentes no treino, enquanto C28 e C30, tambem do dia 13, estao no teste. Isso mostra dependencia de representacao por dia/condicao e exige validacao deixando um dia inteiro de fora.",
            "",
            "## Ablacao ambiental com compensacao recalculada",
            "",
        ]
    )
    for row in ablations.itertuples():
        lines.append(
            f"- `{row.cenario}`: accuracy `{row.accuracy:.4f}`; balanced accuracy `{row.balanced_accuracy:.4f}`."
        )
    without_perfect = exclusions.loc[
        exclusions["cenario"].eq("teste_sem_C24_C28_C30")
    ].iloc[0]
    lines.extend(
        [
            "",
            "A variante recompensada sem pressao chegou a 97,60% neste split fixo, acima dos 93,20% originais. Isso nao deve ser anunciado como novo resultado final antes de validacao por coleta e por dia: o ganho pode refletir a assinatura de Soil/temperatura dos dias presentes no treino e teste. A queda para cerca de 89% sem Soil ou sem temperatura confirma que o classificador depende bastante do contexto ambiental.",
            "",
            "## Efeito das coletas perfeitas no teste",
            "",
            f"Ao retirar C24, C28 e C30 apenas do calculo do teste, a accuracy cai de `{full_test['accuracy']:.4f}` para `{without_perfect['accuracy']:.4f}`. Essas tres coletas acrescentam aproximadamente `{(full_test['accuracy'] - without_perfect['accuracy']) * 100:.2f}` pontos percentuais a accuracy por linha.",
            "",
            "## Arquivos",
            "",
            "- `dados_auditoria/resumo_eventos_coletas_criticas.csv`",
            "- `dados_auditoria/eventos_detalhados_coletas_criticas.csv`",
            "- `dados_auditoria/correlacoes_ambiente_antes_depois.csv`",
            "- `dados_auditoria/verificacao_duplicata_C15_C16.csv`",
            "- `dados_auditoria/decisao_recomendada_por_coleta.csv`",
            "- `auditoria_modelo/sensibilidade_remocao_treino.csv`",
            "- `auditoria_modelo/desempenho_teste_por_coleta.csv`",
            "- `auditoria_modelo/sensibilidade_exclusao_teste.csv`",
            "- `auditoria_modelo/ablacao_variaveis_ambientais.csv`",
            "- `graficos/C*_auditoria.png`",
            "- `graficos/sensibilidade_modelo_coletas_criticas.png`",
            "",
            "## Limite cientifico",
            "",
            "A auditoria identifica anomalias matematicas e associacoes, mas a causa fisica final exige conferir logs de bomba, vedacao, alimentacao, ADC, troca de vaso e planilha bruta.",
            "",
        ]
    )
    (AUDIT / "RELATORIO_AUDITORIA.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(DATASET)
    mapping, lookup = collection_map(df)
    mapping.to_csv(OUT_DATA / "mapa_todas_coletas.csv", index=False, encoding="utf-8-sig")
    mapping[mapping["numero"].isin(CRITICAL)].to_csv(
        OUT_DATA / "mapa_coletas_criticas.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {"C": "C15", "decisao": "manter uma unica representante", "motivo": "duplicada exata de C16"},
            {"C": "C16", "decisao": "remover como duplicata", "motivo": "duplicada exata de C15"},
            {"C": "C17", "decisao": "quarentena", "motivo": "ruido multissensor extremo"},
            {"C": "C24", "decisao": "manter marcada e conferir origem", "motivo": "57 eventos isolados em MQ138"},
            {"C": "C28", "decisao": "quarentena provisoria", "motivo": "degrau multissensor e Soil instavel"},
            {"C": "C29", "decisao": "manter marcada", "motivo": "forte assinatura Soil/dia 13 no treino"},
            {"C": "C30", "decisao": "manter marcada", "motivo": "assinatura ambiental dia 13 e 100% no teste"},
            {"C": "C31", "decisao": "manter marcada", "motivo": "forte assinatura Soil/dia 13 no treino"},
            {"C": "C32", "decisao": "quarentena", "motivo": "ruido multissensor extremo"},
        ]
    ).to_csv(
        OUT_DATA / "decisao_recomendada_por_coleta.csv",
        index=False,
        encoding="utf-8-sig",
    )

    duplicate = audit_duplicate(df)
    duplicate.to_csv(OUT_DATA / "verificacao_duplicata_C15_C16.csv", index=False, encoding="utf-8-sig")

    raw_thresholds = global_jump_thresholds(df, RAW_MQ)
    corr_thresholds = global_jump_thresholds(df, CORR_MQ)
    events, summary = audit_events(df, raw_thresholds, corr_thresholds)
    events.to_csv(OUT_DATA / "eventos_detalhados_coletas_criticas.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DATA / "resumo_eventos_coletas_criticas.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"feature": key, "limiar_abs_diff_p995": value} for key, value in {**raw_thresholds, **corr_thresholds}.items()]
    ).to_csv(OUT_DATA / "limiares_eventos_p995.csv", index=False, encoding="utf-8-sig")

    correlations = audit_environment(df)
    correlations.to_csv(OUT_DATA / "correlacoes_ambiente_antes_depois.csv", index=False, encoding="utf-8-sig")

    sensitivity, per_group, exclusions, ablations = audit_model(df, lookup)
    sensitivity.to_csv(OUT_MODEL / "sensibilidade_remocao_treino.csv", index=False, encoding="utf-8-sig")
    per_group.to_csv(OUT_MODEL / "desempenho_teste_por_coleta.csv", index=False, encoding="utf-8-sig")
    exclusions.to_csv(OUT_MODEL / "sensibilidade_exclusao_teste.csv", index=False, encoding="utf-8-sig")
    ablations.to_csv(OUT_MODEL / "ablacao_variaveis_ambientais.csv", index=False, encoding="utf-8-sig")

    for number in CRITICAL:
        plot_collection(df, number)
    plot_model_sensitivity(sensitivity, exclusions, ablations)
    write_report(duplicate, summary, correlations, sensitivity, per_group, exclusions, ablations)

    print(summary.to_string(index=False))
    print("\nSensibilidade treino:")
    print(sensitivity.to_string(index=False))
    print("\nAblacao ambiente:")
    print(ablations.to_string(index=False))
    print(f"\nRelatorio: {AUDIT / 'RELATORIO_AUDITORIA.md'}")


if __name__ == "__main__":
    main()
