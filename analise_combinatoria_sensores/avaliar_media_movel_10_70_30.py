"""Compara as mesmas coletas suavizadas no split 70/30 por leitura existente."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from avaliar_media_movel_10_arquivo import DATASET, FEATURES, smooth_one_collection
from avaliar_media_movel_10_coletas_destacadas import HIGHLIGHTED
from rodar_combinacoes import model
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "media_movel_10_coletas_destacadas"
SPLIT = ROOT / "validacao_70_30_por_leitura" / "split_por_leitura.csv"


def main() -> None:
    df = pd.read_csv(DATASET)
    df["Vaso_id"] = vessel_id(df)
    split = pd.read_csv(SPLIT)
    if len(split) != len(df) or not np.array_equal(split.linha_original, np.arange(len(df))):
        raise ValueError("Split 70/30 nao corresponde ao dataset")
    train_idx = split.loc[split.conjunto.eq("Treino"), "linha_original"].to_numpy()
    test_idx = split.loc[split.conjunto.eq("Teste"), "linha_original"].to_numpy()

    scenarios = [("original", df)]
    for number, collection in enumerate(HIGHLIGHTED, 1):
        modified, _ = smooth_one_collection(df, collection)
        scenarios.append((f"individual_{number}", modified))
    all_four = df.copy()
    for collection in HIGHLIGHTED:
        all_four, _ = smooth_one_collection(all_four, collection)
    scenarios.append(("quatro_juntas", all_four))

    rows = []
    for name, frame in scenarios:
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        fitted = model().fit(train[FEATURES], train.Classe)
        predicted = fitted.predict(test[FEATURES])
        correct = test.Classe.to_numpy() == predicted
        rows.append({
            "cenario": name,
            "n_leituras_treino": len(train),
            "n_leituras_teste": len(test),
            "n_acertos_teste": int(correct.sum()),
            "acuracia_normal": float(correct.mean()),
        })
        print(name, f"{int(correct.sum())}/{len(test)}", flush=True)
    result = pd.DataFrame(rows)
    result["delta_acertos_vs_original"] = result.n_acertos_teste - result.n_acertos_teste.iloc[0]
    result.to_csv(OUT / "comparacao_70_30_por_leitura.csv", index=False)


if __name__ == "__main__":
    main()
