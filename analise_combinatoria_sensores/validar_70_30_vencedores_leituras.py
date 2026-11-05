"""Avalia em 70/30, por amostra, os vencedores das buscas LOSO."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from rodar_combinacoes import DATASET, ENV
from validacao_por_leitura import make_mean4, predict_samples
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validacao_por_leitura"


def main() -> None:
    raw = pd.read_csv(DATASET)
    raw["Vaso_id"] = vessel_id(raw)
    mean4 = make_mean4(raw)
    mean4["Conjunto"] = mean4.Coleta.map(raw.groupby("Coleta").Conjunto.first())
    vessels = raw.groupby("Vaso_id", sort=True).Classe.first()
    _, test_vessels = train_test_split(
        vessels.index.to_numpy(), test_size=0.3,
        stratify=vessels.to_numpy(), random_state=42,
    )
    test_vessels = set(test_vessels)
    rows = []
    for representation, df in (("cru", raw), ("media_4", mean4)):
        for scenario in ("mq", "mq_ambiente"):
            folder = ROOT / ("busca_mq_loso_leituras" if scenario == "mq"
                             else "busca_mq_ambiente_loso_leituras")
            winner = pd.read_csv(folder / f"ranking_63_mq_{representation}.csv").iloc[0]
            features = winner.sensores_mq.split(" | ")
            if scenario == "mq_ambiente":
                features += list(ENV)
            for protocol, train, test in (
                ("70_30_coleta_historico", df[df.Conjunto.eq("Treino")],
                 df[df.Conjunto.eq("Teste")]),
                ("70_30_vaso", df[~df.Vaso_id.isin(test_vessels)],
                 df[df.Vaso_id.isin(test_vessels)]),
            ):
                if protocol == "70_30_vaso":
                    assert not (set(train.Vaso_id) & set(test.Vaso_id))
                predictions = predict_samples(train, test, features, protocol,
                                              representation, scenario)
                correct = int(predictions.Classe.eq(predictions.predicao).sum())
                rows.append({
                    "representacao": representation,
                    "cenario": scenario,
                    "protocolo": protocol,
                    "sensores": " | ".join(features),
                    "n_vasos_treino": train.Vaso_id.nunique(),
                    "n_vasos_teste": test.Vaso_id.nunique(),
                    "n_coletas_teste": test.Coleta.nunique(),
                    "n_amostras_teste": len(predictions),
                    "n_acertos": correct,
                    "acuracia_normal": correct / len(predictions),
                })
                print(representation, scenario, protocol,
                      f"{correct}/{len(predictions)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "metricas_vencedores_70_30.csv", index=False)


if __name__ == "__main__":
    main()
