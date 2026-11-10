"""LOSO por vaso na base completa para conjuntos fixos, sem seleção de entradas."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rodar_combinacoes import DATASET, ENV, MQ
from validacao_por_vaso import vessel_id


OUT = Path(__file__).resolve().parent / "baselines_fixos_base_completa"
PROJECT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "nove_entradas": [*MQ, *ENV],
    "somente_mq": list(MQ),
    "somente_ambientais": list(ENV),
    "logistica_nove_entradas": [*MQ, *ENV],
}


def metricas(df: pd.DataFrame, modelo: str, unidade: str) -> dict:
    true = df.Classe.to_numpy()
    pred = df.predicao.to_numpy()
    probability = df.probabilidade_classe_1.to_numpy()
    return {
        "modelo": modelo,
        "unidade": unidade,
        "n": len(df),
        "n_acertos": int((true == pred).sum()),
        "acuracia": float(accuracy_score(true, pred)),
        "acuracia_balanceada": float(balanced_accuracy_score(true, pred)),
        "f1_macro": float(f1_score(true, pred, average="macro")),
        "auc": float(roc_auc_score(true, probability)),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw_pressure = pd.read_csv(PROJECT / "comparacao" / "datasets_com_ambiente" /
                               "antes_dia_20_com_ambiente.csv")
    after_window = pd.read_csv(PROJECT / "comparacao" / "pressao_filtrada" /
                               "antes_dia_20_pressao_filtrada.csv")
    after_strict = pd.read_csv(PROJECT / "comparacao" / "pressao_filtrada" /
                               "antes_dia_20_pressao_filtrada_estrito.csv")
    pressure = pd.DataFrame({
        "antes": raw_pressure.groupby("Classe").size(),
        "apos_janela_30": after_window.groupby("Classe").size(),
        "apos_faixa_estrita": after_strict.groupby("Classe").size(),
    })
    pressure["removidas_janela_30"] = pressure.antes - pressure.apos_janela_30
    pressure["percentual_janela_30"] = 100 * pressure.removidas_janela_30 / pressure.antes
    pressure["removidas_faixa_adicional"] = pressure.apos_janela_30 - pressure.apos_faixa_estrita
    pressure["percentual_total"] = 100 * (pressure.antes - pressure.apos_faixa_estrita) / pressure.antes
    pressure.reset_index().to_csv(OUT / "remocao_pressao_por_classe.csv", index=False)
    df = pd.read_csv(DATASET)
    df["Vaso_id"] = vessel_id(df)
    if len(df) != 63492 or df.Coleta.nunique() != 36 or df.Vaso_id.nunique() != 10:
        raise AssertionError("Base completa inesperada")
    soil = (df.groupby(["Coleta", "Vaso_id", "Classe", "Dia"], as_index=False)
            .agg(solo_mediano=("Soil_indice_0_1", "median"),
                 solo_minimo=("Soil_indice_0_1", "min"),
                 solo_maximo=("Soil_indice_0_1", "max"),
                 n_leituras=("Classe", "size")))
    soil["faixa_solo"] = soil.solo_mediano.gt(.4).map({True: "umido", False: "seco"})
    soil["cruza_limiar"] = soil.solo_minimo.le(.4) & soil.solo_maximo.gt(.4)
    soil.to_csv(OUT / "solo_por_coleta.csv", index=False)
    table = (soil.groupby(["Dia", "faixa_solo", "Classe"], as_index=False)
             .agg(n_coletas=("Coleta", "size"), n_leituras=("n_leituras", "sum")))
    table.to_csv(OUT / "tabela_solo_classe_dia.csv", index=False)
    folds = list(LeaveOneGroupOut().split(df, df.Classe, df.Vaso_id))
    summaries = []
    for name, features in CONFIGS.items():
        path = OUT / f"predicoes_{name}.csv"
        if path.exists():
            readings = pd.read_csv(path)
        else:
            parts = []
            for fold, (fit_idx, test_idx) in enumerate(folds, 1):
                train, test = df.iloc[fit_idx], df.iloc[test_idx]
                model = (make_pipeline(StandardScaler(),
                                       LogisticRegression(max_iter=2000, random_state=42))
                         if name == "logistica_nove_entradas" else
                         ExtraTreesClassifier(
                             n_estimators=900, random_state=42, n_jobs=6,
                             max_features="sqrt", min_samples_leaf=10,
                             bootstrap=False,
                         ))
                model.fit(train[features], train.Classe)
                part = test[["Coleta", "Vaso_id", "Classe", "Dia", "Tempo"]].copy()
                part["probabilidade_classe_1"] = model.predict_proba(test[features])[:, 1]
                part["predicao"] = (part.probabilidade_classe_1 >= .5).astype(int)
                part["fold"] = fold
                parts.append(part)
                print(f"{name}: fold {fold}/10", flush=True)
            readings = pd.concat(parts, ignore_index=True)
            readings.to_csv(path, index=False)
        if len(readings) != len(df) or readings.Coleta.nunique() != 36:
            raise AssertionError(f"Predições incompletas: {name}")

        collections = (readings.groupby(["Coleta", "Vaso_id", "Classe"], as_index=False)
                       .agg(probabilidade_classe_1=("probabilidade_classe_1", "mean"),
                            n_leituras=("Classe", "size")))
        collections["predicao"] = (collections.probabilidade_classe_1 >= .5).astype(int)
        vessels = (collections.groupby(["Vaso_id", "Classe"], as_index=False)
                   .agg(probabilidade_classe_1=("probabilidade_classe_1", "mean"),
                        n_coletas=("Coleta", "size")))
        vessels["predicao"] = (vessels.probabilidade_classe_1 >= .5).astype(int)
        collections.to_csv(OUT / f"coletas_{name}.csv", index=False)
        vessels.to_csv(OUT / f"vasos_{name}.csv", index=False)
        summaries.extend([
            metricas(readings, name, "leitura"),
            metricas(collections, name, "coleta"),
            metricas(vessels, name, "vaso"),
        ])
        pd.DataFrame(summaries).to_csv(OUT / "metricas.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
