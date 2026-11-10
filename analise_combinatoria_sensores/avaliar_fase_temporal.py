"""Sensibilidade LOSO usando apenas leituras da fase nominal de 200 a 800 s."""

from pathlib import Path

import pandas as pd

from avaliar_confundimento_dia import FEATURES, run_loso, summarize
from rodar_combinacoes import DATASET
from validacao_por_vaso import vessel_id


OUT = Path(__file__).resolve().parent / "confundimento_dia"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATASET)
    df = df.loc[df.Tempo.between(200, 800)].copy()
    df["Vaso_id"] = vessel_id(df)
    if df.Coleta.nunique() != 36 or df.Vaso_id.nunique() != 10:
        raise AssertionError("A fase nominal eliminou coletas ou vasos")
    summary = []
    for name, features in FEATURES.items():
        pred = run_loso(df, f"fase_200_800_{name}", features)
        pred.to_csv(OUT / f"predicoes_fase_200_800_{name}.csv", index=False)
        summary.append(summarize(pred, "loso_fase_200_800", name))
        pd.DataFrame(summary).to_csv(OUT / "metricas_fase_200_800.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
