"""Resume erros por vaso e por coleta para o vencedor com ambiente, dados crus."""

from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent / "busca_mq_ambiente_loso_leituras"


def main() -> None:
    predictions = pd.read_csv(OUT / "predicoes_vencedor_cru.csv")
    predictions["correta"] = predictions.Classe.eq(predictions.predicao).astype(int)
    if len(predictions) != 63492 or predictions.Vaso_id.nunique() != 10:
        raise ValueError("Predicoes LOSO inesperadas")
    by_vessel = (
        predictions.groupby(["Vaso_id", "Classe"], as_index=False)
        .agg(leituras=("correta", "size"), acertos=("correta", "sum"))
    )
    by_collection = (
        predictions.groupby(["Vaso_id", "Coleta", "Classe"], as_index=False)
        .agg(leituras=("correta", "size"), acertos=("correta", "sum"))
    )
    for frame in (by_vessel, by_collection):
        frame["erros"] = frame.leituras - frame.acertos
        frame["acuracia_normal"] = frame.acertos / frame.leituras
        frame["maioria_errada"] = frame.erros > frame.acertos
    by_vessel.sort_values("erros", ascending=False).to_csv(
        OUT / "erros_por_vaso_cru.csv", index=False,
    )
    by_collection.sort_values("erros", ascending=False).to_csv(
        OUT / "erros_por_coleta_cru.csv", index=False,
    )
    print("Vasos com maioria errada:", int(by_vessel.maioria_errada.sum()))
    print("Coletas com maioria errada:", int(by_collection.maioria_errada.sum()))
    print(by_collection.loc[by_collection.maioria_errada].sort_values(
        "erros", ascending=False
    ).to_string(index=False))


if __name__ == "__main__":
    main()
