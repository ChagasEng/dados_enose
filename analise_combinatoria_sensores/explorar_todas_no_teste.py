"""Avalia todas as 126 configuracoes no teste para ranking exploratorio.

O mesmo teste passa a participar da escolha. Resultados daqui nao estimam de
forma imparcial o desempenho futuro da configuracao vencedora.
"""

from __future__ import annotations

import pandas as pd

from rodar_combinacoes import DATASET, ENV, OUT, model, score


def main() -> None:
    cv_path = OUT / "validacao_cruzada_126_combinacoes.csv"
    if not cv_path.exists():
        raise FileNotFoundError("Execute rodar_combinacoes.py primeiro")
    cv = pd.read_csv(cv_path)
    df = pd.read_csv(DATASET)
    train = df.loc[df.Conjunto.eq("Treino")].reset_index(drop=True)
    test = df.loc[df.Conjunto.eq("Teste")].reset_index(drop=True)
    rows = []
    for number, row in enumerate(cv.itertuples(index=False), 1):
        mq = row.sensores_mq.split(" | ")
        features = mq + (list(ENV) if row.cenario == "mq_ambiente" else [])
        fitted = model().fit(train[features], train.Classe)
        probability = fitted.predict_proba(test[features])[:, 1]
        result, _ = score(test, probability)
        rows.append({
            "cenario": row.cenario,
            "sensores_mq": row.sensores_mq,
            "n_mq": len(mq),
            "n_sensores_total": len(features),
            "cv_coleta_balanced_accuracy": row.coleta_balanced_accuracy,
            **result,
        })
        print(f"{number}/{len(cv)} {row.cenario}: {row.sensores_mq}", flush=True)
    # Todos os criterios de desempenho usam coletas inteiras.
    # Entre empates no teste, privilegia suporte na CV do treino;
    # depois, o menor numero total de sensores.
    results = pd.DataFrame(rows).sort_values(
        ["coleta_balanced_accuracy", "cv_coleta_balanced_accuracy",
         "n_sensores_total", "sensores_mq"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    results.insert(0, "ranking_exploratorio", range(1, len(results) + 1))
    results.to_csv(OUT / "exploracao_teste_126_combinacoes.csv", index=False)
    candidate = results.iloc[0]
    candidate_mq = candidate.sensores_mq.split(" | ")
    candidate_features = candidate_mq + (list(ENV) if candidate.cenario == "mq_ambiente" else [])
    fitted = model().fit(train[candidate_features], train.Classe)
    _, by_collection = score(test, fitted.predict_proba(test[candidate_features])[:, 1])
    by_collection.to_csv(OUT / "predicoes_candidato_exploratorio.csv", index=False)
    print(results.head(15).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
