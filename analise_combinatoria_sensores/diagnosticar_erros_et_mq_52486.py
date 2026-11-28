"""Localiza e caracteriza erros LOSO do ET usando apenas cinco MQ, em 52.486 leituras."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "loso_balanceado_52486_solo_binario"
OUT = ROOT / "diagnostico_erros_et_mq_52486"
MODELS = ("et", "rf", "tcn", "mlp", "knn", "ensemble")


def aligned_predictions(name: str, scenario: str, base: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(SRC / f"predicoes_{scenario}_{name}.csv")
    for key in ("linha_balanceada", "Coleta", "Vaso_id", "Classe", "Tempo"):
        assert pred[key].equals(base[key]), f"Previsoes desalinhadas: {name}/{scenario}/{key}"
    assert pred.fold.nunique() == 10
    assert pred.groupby("fold").Vaso_id.nunique().eq(1).all()
    return pred


def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = pd.read_csv(SRC / "base_balanceada.csv")
    assert len(base) == 52_486 and base.Classe.value_counts().to_dict() == {0: 26_243, 1: 26_243}
    mq = aligned_predictions("et", "mq", base)
    full = aligned_predictions("et", "mq_ambiente", base)
    cols = ["linha_balanceada", "Coleta", "Vaso_id", "Classe", "Dia", "Tempo",
            "Condicao_solo_binaria", "Temp_C", "Pres_kPa", "MQ2", "MQ3", "MQ8", "MQ135", "MQ138"]
    frame = base[cols].copy()
    frame["prob_classe_1_mq"] = mq.probabilidade_classe_1
    frame["prob_classe_1_mq_ambiente"] = full.probabilidade_classe_1
    frame["erro_mq"] = mq.Classe.ne(mq.predicao)
    frame["erro_mq_ambiente"] = full.Classe.ne(full.predicao)
    frame["prob_classe_errada_mq"] = np.where(
        frame.Classe.eq(0), frame.prob_classe_1_mq, 1 - frame.prob_classe_1_mq
    )
    frame["corrigido_com_ambiente"] = frame.erro_mq & ~frame.erro_mq_ambiente
    frame["novo_erro_com_ambiente"] = ~frame.erro_mq & frame.erro_mq_ambiente
    frame["erro_persistente"] = frame.erro_mq & frame.erro_mq_ambiente
    assert int(frame.erro_mq.sum()) == 2_932
    assert int(frame.erro_mq_ambiente.sum()) == 1_947
    frame.loc[frame.erro_mq].to_csv(OUT / "leituras_erradas_et_mq.csv", index=False)

    by_vase = (frame.groupby(["Vaso_id", "Classe"], as_index=False)
               .agg(n_leituras=("Classe", "size"), erros_mq=("erro_mq", "sum"),
                    erros_mq_ambiente=("erro_mq_ambiente", "sum"),
                    corrigidos_com_ambiente=("corrigido_com_ambiente", "sum"),
                    novos_erros_com_ambiente=("novo_erro_com_ambiente", "sum"),
                    persistentes=("erro_persistente", "sum")))
    by_vase["taxa_erro_mq"] = by_vase.erros_mq / by_vase.n_leituras
    by_vase.to_csv(OUT / "erros_por_vaso.csv", index=False)

    rows = []
    for (collection, vessel, label), group in frame.groupby(["Coleta", "Vaso_id", "Classe"]):
        group = group.sort_values("Tempo").copy()
        group["quartil_tempo"] = pd.qcut(np.arange(len(group)), 4, labels=False)
        rates = group.groupby("quartil_tempo").erro_mq.mean()
        runs = group.erro_mq.ne(group.erro_mq.shift()).cumsum()
        contiguous = (group.groupby(runs, sort=False)
                      .agg(erro=("erro_mq", "first"), n=("erro_mq", "size"),
                           inicio=("Tempo", "first"), fim=("Tempo", "last")))
        wrong_runs = contiguous.loc[contiguous.erro].sort_values("n", ascending=False)
        longest = wrong_runs.iloc[0] if len(wrong_runs) else None
        errors = group.loc[group.erro_mq]
        rows.append({
            "Coleta": collection, "Vaso_id": vessel, "Classe": label,
            "Dia": group.Dia.iloc[0], "n_leituras": len(group),
            "erros_mq": int(group.erro_mq.sum()),
            "taxa_erro_mq": float(group.erro_mq.mean()),
            "erros_mq_ambiente": int(group.erro_mq_ambiente.sum()),
            "corrigidos_com_ambiente": int(group.corrigido_com_ambiente.sum()),
            "novos_erros_com_ambiente": int(group.novo_erro_com_ambiente.sum()),
            "prob_media_classe_1_mq": float(group.prob_classe_1_mq.mean()),
            "prob_media_classe_1_mq_ambiente": float(group.prob_classe_1_mq_ambiente.mean()),
            "erros_prob_errada_ge_0_9": int(errors.prob_classe_errada_mq.ge(.9).sum()),
            **{f"taxa_erro_quartil_{q+1}": float(rates.loc[q]) for q in range(4)},
            "maior_bloco_erros_n": 0 if longest is None else int(longest.n),
            "maior_bloco_inicio_s": np.nan if longest is None else float(longest.inicio),
            "maior_bloco_fim_s": np.nan if longest is None else float(longest.fim),
        })
    by_collection = pd.DataFrame(rows).sort_values("erros_mq", ascending=False)
    assert by_collection.erros_mq.sum() == 2_932
    by_collection.to_csv(OUT / "erros_por_coleta.csv", index=False)

    wrong = frame.loc[frame.erro_mq].copy()
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
    labels = ["0,50 a <0,60", "0,60 a <0,70", "0,70 a <0,80", "0,80 a <0,90", "0,90 a 1,00"]
    wrong["faixa_probabilidade_errada"] = pd.cut(
        wrong.prob_classe_errada_mq, bins=bins, labels=labels, right=False
    )
    by_probability = (wrong.groupby("faixa_probabilidade_errada", observed=True)
                      .size().reindex(labels, fill_value=0).rename("n_erros").reset_index())
    by_probability["pct_erros"] = by_probability.n_erros / len(wrong) * 100
    by_probability.to_csv(OUT / "erros_por_probabilidade.csv", index=False)

    all_models = frame[["linha_balanceada", "Coleta", "Vaso_id", "Classe"]].copy()
    for scenario in ("mq", "mq_ambiente"):
        for name in MODELS:
            pred = aligned_predictions(name, scenario, base)
            all_models[f"{scenario}_{name}"] = pred.Classe.ne(pred.predicao)
    by_models = all_models.groupby(["Vaso_id", "Classe"])[
        [f"{scenario}_{name}" for scenario in ("mq", "mq_ambiente") for name in MODELS]
    ].sum().reset_index()
    by_models.to_csv(OUT / "erros_por_vaso_todos_modelos.csv", index=False)
    for scenario in ("mq", "mq_ambiente"):
        all_models[f"todos_seis_erraram_{scenario}"] = all_models[
            [f"{scenario}_{name}" for name in MODELS]
        ].all(axis=1)
    shared = all_models.groupby(["Coleta", "Vaso_id", "Classe"], as_index=False)[
        ["todos_seis_erraram_mq", "todos_seis_erraram_mq_ambiente"]
    ].sum()
    shared.to_csv(OUT / "erros_compartilhados_seis_modelos.csv", index=False)

    top = by_collection.head(5)
    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharey=True, constrained_layout=True)
    for ax, row in zip(axes, top.itertuples()):
        g = frame.loc[frame.Coleta.eq(row.Coleta)].sort_values("Tempo")
        ax.scatter(g.Tempo, g.prob_classe_1_mq, s=4, alpha=.23, color="#1260a1")
        ax.plot(g.Tempo, g.prob_classe_1_mq.rolling(41, center=True, min_periods=1).mean(),
                color="#1260a1", linewidth=1.8, label="MQ: média móvel de 41 leituras")
        ax.plot(g.Tempo, g.prob_classe_1_mq_ambiente.rolling(41, center=True, min_periods=1).mean(),
                color="#d07b19", linewidth=1.6, label="MQ + solo, temperatura, pressão")
        ax.axhline(.5, color="#aa2727", linewidth=1, linestyle="--")
        ax.set_ylim(-.03, 1.03)
        ax.set_ylabel("P(classe 1)")
        ax.set_title(f"{row.Coleta} | {row.erros_mq}/{row.n_leituras} erros com MQ", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Tempo na coleta (s)")
    fig.savefig(OUT / "probabilidades_ao_longo_das_cinco_coletas.png", dpi=180)
    plt.close(fig)
    print(f"Erros MQ: {len(wrong)}; coletas com erro: {by_collection.erros_mq.gt(0).sum()}")
    print(by_vase.to_string(index=False))
    print(by_probability.to_string(index=False))


if __name__ == "__main__":
    main()
