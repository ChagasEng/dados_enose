"""Auditoria de dia e periodo com as 36 coletas, sem excluir coletas.

Executar da raiz: python analise_combinatoria_sensores/avaliar_confundimento_dia.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

from rodar_combinacoes import DATASET, ENV, MQ
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "confundimento_dia"
FEATURES = {"nove_entradas": [*MQ, *ENV], "seis_mq": list(MQ), "ambiente": list(ENV)}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                normalize_mq_by_day: bool = False) -> np.ndarray:
    x_train = train[features].copy()
    x_test = test[features].copy()
    if normalize_mq_by_day:
        for day in sorted(train.Dia.unique()):
            train_day = train.Dia.eq(day)
            test_day = test.Dia.eq(day)
            if not test_day.any():
                continue
            values = x_train.loc[train_day, features]
            mean = values.mean()
            scale = values.std(ddof=0).replace(0, 1)
            x_test.loc[test_day, features] = (x_test.loc[test_day, features] - mean) / scale
            x_train.loc[train_day, features] = (values - mean) / scale
    model = ExtraTreesClassifier(n_estimators=900, random_state=42, n_jobs=6,
                                 max_features="sqrt", min_samples_leaf=10,
                                 bootstrap=False)
    model.fit(x_train, train.Classe)
    return model.predict_proba(x_test)[:, 1]


def summarize(predictions: pd.DataFrame, protocol: str, model: str) -> dict:
    readings = predictions.copy()
    readings["predicao"] = readings.probabilidade_classe_1.ge(.5).astype(int)
    collections = (readings.groupby(["Coleta", "Vaso_id", "Classe", "Dia"], as_index=False)
                   .probabilidade_classe_1.mean())
    collections["predicao"] = collections.probabilidade_classe_1.ge(.5).astype(int)
    out = {"protocolo": protocol, "modelo": model, "n_leituras": len(readings),
           "n_coletas": len(collections), "n_vasos": collections.Vaso_id.nunique()}
    for unit, frame in [("leitura", readings), ("coleta", collections)]:
        out[f"acertos_{unit}"] = int(frame.Classe.eq(frame.predicao).sum())
        out[f"acuracia_{unit}"] = float(accuracy_score(frame.Classe, frame.predicao))
        out[f"balanceada_{unit}"] = float(balanced_accuracy_score(frame.Classe, frame.predicao))
        out[f"auc_{unit}"] = float(roc_auc_score(frame.Classe, frame.probabilidade_classe_1))
    return out


def run_loso(frame: pd.DataFrame, protocol: str, features: list[str],
             normalize_mq_by_day: bool = False) -> pd.DataFrame:
    parts = []
    for fold, (fit_idx, test_idx) in enumerate(
            LeaveOneGroupOut().split(frame, frame.Classe, frame.Vaso_id), 1):
        train, test = frame.iloc[fit_idx], frame.iloc[test_idx]
        if normalize_mq_by_day and not set(test.Dia).issubset(set(train.Dia)):
            raise AssertionError("Dia do teste sem controles para normalizacao")
        pred = test[["Coleta", "Vaso_id", "Classe", "Dia", "Tempo"]].copy()
        pred["probabilidade_classe_1"] = fit_predict(
            train, test, features, normalize_mq_by_day)
        pred["fold"] = fold
        parts.append(pred)
        print(f"{protocol}: fold {fold}/10", flush=True)
    return pd.concat(parts, ignore_index=True)


def day_baseline(df: pd.DataFrame) -> pd.DataFrame:
    collections = df[["Coleta", "Vaso_id", "Classe", "Dia"]].drop_duplicates().copy()
    # 24/36: usa a classe majoritaria do proprio conjunto inteiro; descritivo,
    # nao predicao fora de treino.
    counts = collections.groupby(["Dia", "Classe"]).size().unstack(fill_value=0)
    counts["majoritaria"] = counts.idxmax(axis=1)
    collections["predicao_mesmo_conjunto"] = collections.Dia.map(counts.majoritaria)
    # Referencia genuinamente fora de treino: maioria do mesmo dia nos outros
    # nove vasos, desempate pela classe mais frequente nos demais vasos.
    for index, row in collections.iterrows():
        other = collections.loc[collections.Vaso_id.ne(row.Vaso_id)]
        same_day = other.loc[other.Dia.eq(row.Dia)]
        day_counts = same_day.Classe.value_counts()
        if len(day_counts) == 2 and day_counts.iloc[0] == day_counts.iloc[1]:
            global_counts = other.Classe.value_counts()
            label = int(global_counts.idxmax())
        else:
            label = int(day_counts.idxmax())
        collections.loc[index, "predicao_loso_dia"] = label
    collections.predicao_loso_dia = collections.predicao_loso_dia.astype(int)
    return collections


def paired_sign_flip() -> pd.DataFrame:
    base = ROOT / "baselines_fixos_base_completa"
    results = []
    for unit in ["predicoes", "coletas"]:
        name = "predicoes" if unit == "predicoes" else "coletas"
        full = pd.read_csv(base / f"{name}_nove_entradas.csv")
        env = pd.read_csv(base / f"{name}_somente_ambientais.csv")
        for frame in [full, env]:
            frame["acerto"] = frame.Classe.eq(frame.predicao).astype(int)
        a = full.groupby("Vaso_id").acerto.mean().sort_index()
        b = env.groupby("Vaso_id").acerto.mean().sort_index()
        delta = a - b
        observed = abs(delta.mean())
        null = np.array([abs(np.mean(delta.to_numpy() * signs))
                         for signs in product((-1, 1), repeat=len(delta))])
        # Tolerancia para empates de representacao em ponto flutuante.
        p = float(np.mean(null >= observed - 1e-12))
        results.append({"unidade": "leitura" if unit == "predicoes" else "coleta",
                        "media_diferencas_nove_menos_ambiente": float(delta.mean()),
                        "vasos_melhora": int(delta.gt(0).sum()),
                        "vasos_piora": int(delta.lt(0).sum()),
                        "vasos_empate": int(delta.eq(0).sum()),
                        "p_bicaudal_inversao_exploratorio": p,
                        "n_inversoes": len(null)})
        delta.rename("diferenca_nove_menos_ambiente").to_csv(
            OUT / f"diferencas_por_vaso_{results[-1]['unidade']}.csv")
    return pd.DataFrame(results)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATASET)
    df["Vaso_id"] = vessel_id(df)
    if len(df) != 63492 or df.Coleta.nunique() != 36 or df.Vaso_id.nunique() != 10:
        raise AssertionError("Base completa inesperada")

    baseline = day_baseline(df)
    baseline.to_csv(OUT / "baseline_por_dia_coletas.csv", index=False)
    print("Dia, maioria no mesmo conjunto:",
          int(baseline.Classe.eq(baseline.predicao_mesmo_conjunto).sum()), "/36")
    print("Dia, maioria LOSO:",
          int(baseline.Classe.eq(baseline.predicao_loso_dia).sum()), "/36")

    excluded = pd.read_csv(ROOT / "validacao_sem_cinco_coletas" / "coletas_excluidas.csv")
    excluded["Dia"] = excluded.Coleta.str.extract(r"dia (\d+)").astype(int)
    excluded["minoritario_no_dia"] = excluded.apply(
        lambda row: int(baseline.loc[baseline.Dia.eq(row.Dia), "Classe"]
                        .value_counts().get(row.Classe, 0)) <
                    int(baseline.loc[baseline.Dia.eq(row.Dia), "Classe"]
                        .value_counts().max()), axis=1)
    excluded.to_csv(OUT / "exclusoes_por_dia_e_classe.csv", index=False)

    rank = pd.read_csv(ROOT / "busca_mq_ambiente_loso_leituras" / "ranking_63_mq_cru.csv")
    rank.acuracia_normal.describe(percentiles=[.25, .5, .75]).to_csv(
        OUT / "dispersao_63_combinacoes.csv", header=["valor"])

    summary = []
    wet = df.loc[df.Dia.isin([18, 19])].copy()
    for name, features in FEATURES.items():
        pred = run_loso(wet, f"loso_dias_18_19_{name}", features)
        pred.to_csv(OUT / f"predicoes_loso_18_19_{name}.csv", index=False)
        summary.append(summarize(pred, "loso_dias_18_19", name))
        pd.DataFrame(summary).to_csv(OUT / "metricas_modelos.csv", index=False)

    for train_days, test_days, label in [([11, 12, 13], [18, 19], "seco_para_umido"),
                                        ([18, 19], [11, 12, 13], "umido_para_seco")]:
        train = df.loc[df.Dia.isin(train_days)]
        test = df.loc[df.Dia.isin(test_days)]
        for name, features in FEATURES.items():
            pred = test[["Coleta", "Vaso_id", "Classe", "Dia", "Tempo"]].copy()
            pred["probabilidade_classe_1"] = fit_predict(train, test, features)
            pred.to_csv(OUT / f"predicoes_{label}_{name}.csv", index=False)
            summary.append(summarize(pred, label, name))
            print(label, name, summary[-1]["acuracia_coleta"], flush=True)
            pd.DataFrame(summary).to_csv(OUT / "metricas_modelos.csv", index=False)

    pred = run_loso(df, "loso_mq_padronizado_por_dia", list(MQ), True)
    pred.to_csv(OUT / "predicoes_mq_padronizado_por_dia.csv", index=False)
    summary.append(summarize(pred, "loso_base_completa", "mq_padronizado_por_dia"))
    pd.DataFrame(summary).to_csv(OUT / "metricas_modelos.csv", index=False)

    timing = (df.groupby(["Coleta", "Classe", "Dia"], as_index=False)
              .agg(tempo_min=("Tempo", "min"), tempo_max=("Tempo", "max"),
                   n_leituras=("Tempo", "size")))
    timing["duracao_retida_s"] = timing.tempo_max - timing.tempo_min
    timing.to_csv(OUT / "duracao_por_coleta.csv", index=False)
    df.assign(fase=np.select([df.Tempo.lt(200), df.Tempo.gt(800)],
                              ["antes_200", "apos_800"], default="200_a_800"))\
      .groupby(["Classe", "fase"]).size().rename("n_leituras").reset_index().to_csv(
          OUT / "leituras_por_fase_nominal.csv", index=False)

    pressure = pd.read_csv(ROOT.parent / "comparacao" / "pressao_filtrada" /
                           "antes_dia_20_pressao_filtrada.csv")
    pressure.groupby("Classe")["Pres."].describe(percentiles=[.01, .05, .5, .95, .99])\
        .to_csv(OUT / "pressao_apos_janela_por_classe.csv")
    strict = pd.read_csv(ROOT.parent / "comparacao" / "pressao_filtrada" /
                         "antes_dia_20_pressao_filtrada_estrito.csv")
    strict.groupby("Classe")["Pres."].describe(percentiles=[.01, .05, .5, .95, .99])\
        .to_csv(OUT / "pressao_apos_faixa_por_classe.csv")
    pd.DataFrame([{"mediana_apos_janela": pressure["Pres."].median(),
                   "limite_inferior": pressure["Pres."].median() - .5,
                   "limite_superior": pressure["Pres."].median() + .5}]).to_csv(
                       OUT / "faixa_pressao.csv", index=False)
    paired_sign_flip().to_csv(OUT / "inversao_sinais_pareada.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
