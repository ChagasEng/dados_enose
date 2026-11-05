"""Testa media movel causal de 10 nas quatro coletas com maioria errada."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from avaliar_media_movel_10_arquivo import (
    DATASET, ORIGINAL_PREDICTIONS, SMOOTH_COLUMNS,
    loso_predictions, smooth_one_collection,
)
from validacao_por_vaso import vessel_id


OUT = Path(__file__).resolve().parent / "media_movel_10_coletas_destacadas"
HIGHLIGHTED = (
    "dia 12 - Soja Saudavel Vaso 2",
    "dia 19 - Soja Heterodera Vaso 9",
    "dia 19 - Soja Saudavel Vaso 2",
    "dia 19 - Soja Saudavel Vaso 1",
)


def summarize(predictions: pd.DataFrame, scenario: str,
              changed: tuple[str, ...]) -> tuple[dict, pd.DataFrame]:
    correct = predictions.Classe.eq(predictions.predicao)
    focus = predictions.Coleta.isin(HIGHLIGHTED)
    result = {
        "cenario": scenario,
        "n_coletas_modificadas": len(changed),
        "coletas_modificadas": " | ".join(changed) if changed else "nenhuma",
        "n_leituras_total": len(predictions),
        "n_acertos_total": int(correct.sum()),
        "acuracia_normal_total": float(correct.mean()),
        "n_leituras_quatro_destacadas": int(focus.sum()),
        "n_acertos_quatro_destacadas": int((correct & focus).sum()),
        "acuracia_nas_quatro_destacadas": float(correct.loc[focus].mean()),
    }
    by_collection = (
        predictions.assign(correta=correct.astype(int))
        .groupby(["Coleta", "Vaso_id", "Classe"], as_index=False)
        .agg(leituras=("correta", "size"), acertos=("correta", "sum"))
    )
    by_collection.insert(0, "cenario", scenario)
    by_collection["erros"] = by_collection.leituras - by_collection.acertos
    by_collection["acuracia_normal"] = by_collection.acertos / by_collection.leituras
    return result, by_collection


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATASET)
    df["Vaso_id"] = vessel_id(df)
    if not set(HIGHLIGHTED).issubset(set(df.Coleta)):
        raise ValueError("Alguma coleta destacada nao aparece no dataset")
    original = pd.read_csv(ORIGINAL_PREDICTIONS)
    if len(original) != len(df):
        raise ValueError("Predicoes originais nao correspondem a este dataset")

    rows, collection_rows, audits = [], [], []
    baseline, baseline_by_collection = summarize(original, "original", ())
    rows.append(baseline)
    collection_rows.append(baseline_by_collection)

    for number, collection in enumerate(HIGHLIGHTED, 1):
        modified, audit = smooth_one_collection(df, collection)
        scenario = f"individual_{number}"
        audit.insert(0, "cenario", scenario)
        audits.append(audit)
        predicted = loso_predictions(modified)
        row, by_collection = summarize(predicted, scenario, (collection,))
        rows.append(row)
        collection_rows.append(by_collection)
        print(scenario, collection, row["n_acertos_total"], flush=True)

    all_four = df.copy()
    for collection in HIGHLIGHTED:
        all_four, _ = smooth_one_collection(all_four, collection)
    predicted = loso_predictions(all_four)
    row, by_collection = summarize(predicted, "quatro_juntas", HIGHLIGHTED)
    rows.append(row)
    collection_rows.append(by_collection)
    predicted.sort_values(["Coleta", "Tempo"]).to_csv(
        OUT / "predicoes_loso_quatro_juntas.csv", index=False,
    )

    summary = pd.DataFrame(rows)
    summary["delta_acertos_vs_original"] = summary.n_acertos_total - baseline["n_acertos_total"]
    summary["delta_acuracia_pp_vs_original"] = (
        summary.acuracia_normal_total - baseline["acuracia_normal_total"]
    ) * 100
    summary.to_csv(OUT / "comparacao_cenarios.csv", index=False)
    pd.concat(collection_rows, ignore_index=True).to_csv(
        OUT / "acertos_por_coleta_e_cenario.csv", index=False,
    )
    pd.concat(audits, ignore_index=True).to_csv(
        OUT / "auditoria_valores_originais_e_medias.csv", index=False,
    )
    (OUT / "metodo.json").write_text(json.dumps({
        "dataset": str(DATASET),
        "coletas_destacadas": list(HIGHLIGHTED),
        "regra_media_movel": "Primeiras 9 leituras originais; da 10a em diante, media das 10 leituras originais consecutivas terminando na posicao atual",
        "colunas_suavizadas": SMOOTH_COLUMNS,
        "cenario_individual": "Cada uma das quatro coletas e modificada separadamente, mantendo as outras 35 originais",
        "cenario_conjunto": "As quatro coletas modificadas simultaneamente, mantendo as outras 32 originais",
        "validacao": "LOSO por vaso em todas as 36 coletas, com 10 vasos; mesma arquitetura e hiperparametros do modelo original",
        "comparacao": "Mesmas 63.492 leituras de teste em todos os cenarios; acuracia normal por leitura",
        "limite": "As coletas foram escolhidas porque ja apresentaram erros; o resultado e exploratorio e pode refletir ajuste a casos conhecidos",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
