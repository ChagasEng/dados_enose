"""Acrescenta acurácia balanceada ao ranking das 511 combinações já avaliadas."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent / "busca_nove_variaveis_loso"
KEY = ["sensores_mq", "sensores_ambiente"]


def main() -> None:
    ranking = pd.read_csv(ROOT / "ranking_511.csv")
    collections = pd.read_csv(ROOT / "acertos_por_coleta_511.csv")
    ranking[KEY] = ranking[KEY].fillna("")
    collections[KEY] = collections[KEY].fillna("")
    grouped = (collections.groupby([*KEY, "Classe"], as_index=False)
               [["n_leituras", "n_acertos"]].sum())
    grouped["recall"] = grouped.n_acertos / grouped.n_leituras
    if grouped.groupby(KEY).Classe.nunique().ne(2).any():
        raise AssertionError("Há combinação sem as duas classes")
    balanced = (grouped.groupby(KEY, as_index=False)
                .recall.mean().rename(columns={"recall": "acuracia_balanceada"}))
    result = ranking.merge(balanced, on=KEY, validate="one_to_one")
    if len(result) != 511:
        raise AssertionError("Esperadas 511 combinações")
    result = result.sort_values(
        ["acuracia_balanceada", "n_variaveis", "n_mq", *KEY],
        ascending=[False, True, False, True, True],
    ).reset_index(drop=True)
    result.insert(0, "ranking_balanceado", range(1, len(result) + 1))
    result.to_csv(ROOT / "ranking_511_acuracia_balanceada.csv", index=False)
    print(result.head(5)[["ranking_balanceado", *KEY, "acuracia_balanceada"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
