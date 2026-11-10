"""AUC por coleta de ET, RF e TCN nas mesmas 31 coletas e entradas fixas."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

from rodar_combinacoes import DATASET, ENV, model
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "validacao_sem_cinco_coletas"
OUT = ROOT / "comparacao_auc_modelos_filtrada"
MQ = ["MQ2", "MQ3", "MQ8", "MQ135", "MQ138"]
SCENARIOS = {"mq": MQ, "mq_ambiente": [*MQ, *ENV]}


def predictions_et(df: pd.DataFrame, features: list[str], path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    pieces = []
    for fold, (fit_idx, test_idx) in enumerate(
        LeaveOneGroupOut().split(df, df.Classe, df.Vaso_id), 1
    ):
        train, test = df.iloc[fit_idx], df.iloc[test_idx]
        fitted = model().set_params(n_jobs=6).fit(train[features], train.Classe)
        part = test[["Coleta", "Vaso_id", "Classe"]].copy()
        part["probabilidade_classe_1"] = fitted.predict_proba(test[features])[:, 1]
        part["fold"] = fold
        pieces.append(part)
        print(f"ET {'+'.join(features)}: {fold}/10", flush=True)
    result = pd.concat(pieces, ignore_index=True)
    result.to_csv(path, index=False)
    return result


def auc_summary(readings: pd.DataFrame, name: str, scenario: str) -> dict:
    collections = (readings.groupby(["Coleta", "Vaso_id", "Classe"], as_index=False)
                   .probabilidade_classe_1.mean())
    vessels = (collections.groupby(["Vaso_id", "Classe"], as_index=False)
               .probabilidade_classe_1.mean())
    return {
        "modelo": name,
        "cenario": scenario,
        "n_leituras": len(readings),
        "n_coletas": len(collections),
        "n_vasos": len(vessels),
        "auc_leitura": float(roc_auc_score(readings.Classe,
                                          readings.probabilidade_classe_1)),
        "auc_coleta": float(roc_auc_score(collections.Classe,
                                         collections.probabilidade_classe_1)),
        "auc_vaso": float(roc_auc_score(vessels.Classe,
                                       vessels.probabilidade_classe_1)),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    excluded = set(pd.read_csv(SOURCE / "coletas_excluidas.csv").Coleta)
    df = pd.read_csv(DATASET)
    df = df.loc[~df.Coleta.isin(excluded)].copy()
    df["Vaso_id"] = vessel_id(df)
    if len(df) != 55946 or df.Coleta.nunique() != 31 or df.Vaso_id.nunique() != 10:
        raise AssertionError("Base filtrada inesperada")
    summaries = []
    for scenario, features in SCENARIOS.items():
        for name in ("ExtraTrees", "RandomForest", "TCN"):
            if name == "ExtraTrees":
                readings = predictions_et(df, features, OUT / f"predicoes_et_{scenario}.csv")
            else:
                readings = pd.read_csv(SOURCE / f"predicoes_{name.lower()}_{scenario}.csv")
            if len(readings) != len(df) or readings.Coleta.nunique() != 31:
                raise AssertionError(f"Predições incompletas: {name}/{scenario}")
            summaries.append(auc_summary(readings, name, scenario))
            pd.DataFrame(summaries).to_csv(OUT / "metricas_auc.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
