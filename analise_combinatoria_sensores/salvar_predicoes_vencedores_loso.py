"""Salva previsoes por amostra dos vencedores das quatro buscas LOSO."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from rodar_combinacoes import DATASET, ENV
from validacao_por_leitura import make_mean4, predict_samples
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parent


def main() -> None:
    raw = pd.read_csv(DATASET)
    raw["Vaso_id"] = vessel_id(raw)
    representations = {"cru": raw, "media_4": make_mean4(raw)}
    for scenario in ("mq", "mq_ambiente"):
        output = ROOT / ("busca_mq_loso_leituras" if scenario == "mq"
                         else "busca_mq_ambiente_loso_leituras")
        for representation, df in representations.items():
            ranking = pd.read_csv(output / f"ranking_63_mq_{representation}.csv")
            winner = ranking.iloc[0]
            mq = winner.sensores_mq.split(" | ")
            features = mq + (list(ENV) if scenario == "mq_ambiente" else [])
            parts = []
            for fold, (fit_idx, test_idx) in enumerate(
                LeaveOneGroupOut().split(df, df.Classe, df.Vaso_id), 1
            ):
                train, test = df.iloc[fit_idx], df.iloc[test_idx]
                assert not (set(train.Vaso_id) & set(test.Vaso_id))
                predicted = predict_samples(train, test, features, "loso_vaso",
                                            representation, scenario)
                predicted.insert(3, "fold", fold)
                parts.append(predicted)
            all_predictions = pd.concat(parts, ignore_index=True)
            correct = int(all_predictions.Classe.eq(all_predictions.predicao).sum())
            if len(all_predictions) != winner.n_amostras or correct != winner.n_acertos:
                raise AssertionError("Previsoes do vencedor nao reproduzem o ranking")
            all_predictions.sort_values(["Coleta", "Tempo"]).to_csv(
                output / f"predicoes_vencedor_{representation}.csv", index=False
            )
            print(scenario, representation, winner.sensores_mq,
                  f"{correct}/{len(all_predictions)}", flush=True)


if __name__ == "__main__":
    main()
