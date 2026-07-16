"""Ablacao e busca exaustiva das combinacoes dos seis sensores MQ.

Mantem o mesmo dataset, coluna ``Conjunto`` (split por coleta) e
hiperparametros do melhor ExtraTrees. As tres variaveis de ambiente continuam
no modelo em todos os cenarios; portanto, a unica mudanca entre os testes e
quais sensores de gas estao disponiveis.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[1]
DATASET = BASE / "dados" / "dataset_melhor_modelo_sensores_corrigidos.csv"
TARGET = "Classe"
SENSORES = ("MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138")
AMBIENTE = ("Soil_indice_0_1", "Temp_C", "Pres_kPa")
RANDOM_STATE = 42


def criar_modelo() -> ExtraTreesClassifier:
    """Replica os hiperparametros do rodar_extratrees_melhor_modelo.py."""
    return ExtraTreesClassifier(
        n_estimators=900,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=10,
        class_weight=None,
        bootstrap=False,
    )


def features_para(sensores: tuple[str, ...]) -> list[str]:
    return [f"{sensor}_corrigido_env" for sensor in sensores] + list(AMBIENTE)


def corrigir_mq_pelo_treino(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Refaz a compensacao ambiental usando somente o treino do cenario."""
    corrigido = df.copy()
    for sensor in SENSORES:
        compensador = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("huber", HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=400)),
            ]
        )
        compensador.fit(corrigido.loc[train_mask, AMBIENTE], corrigido.loc[train_mask, sensor])
        efeito_todos = compensador.predict(corrigido[list(AMBIENTE)])
        efeito_treino = compensador.predict(corrigido.loc[train_mask, AMBIENTE])
        corrigido[f"{sensor}_corrigido_env"] = corrigido[sensor] - (
            efeito_todos - float(efeito_treino.mean())
        )
    return corrigido


def localizar_coleta(rotulo: str) -> str:
    """Converte, por exemplo, C16 no nome de Coleta usado pelo dataset."""
    mapa = BASE / "auditoria_coletas_criticas" / "dados_auditoria" / "mapa_todas_coletas.csv"
    if not mapa.exists():
        raise FileNotFoundError(f"Mapa de coletas nao encontrado: {mapa}")
    encontrados = pd.read_csv(mapa)
    linha = encontrados.loc[encontrados["C"].eq(rotulo.upper())]
    if len(linha) != 1:
        raise ValueError(f"Rotulo de coleta invalido: {rotulo}. Use um C existente, como C16.")
    return str(linha.iloc[0]["Coleta"])


def avaliar(
    train_df: pd.DataFrame, test_df: pd.DataFrame, sensores: tuple[str, ...]
) -> dict[str, object]:
    features = features_para(sensores)
    modelo = criar_modelo()
    modelo.fit(train_df[features], train_df[TARGET])
    predicao = modelo.predict(test_df[features])

    return {
        "sensores_mantidos": " | ".join(sensores),
        "sensores_removidos": " | ".join(sensor for sensor in SENSORES if sensor not in sensores)
        or "nenhum",
        "qtd_sensores": len(sensores),
        "features": " | ".join(features),
        "accuracy": accuracy_score(test_df[TARGET], predicao),
        "balanced_accuracy": balanced_accuracy_score(test_df[TARGET], predicao),
        "f1_macro": f1_score(test_df[TARGET], predicao, average="macro"),
    }


def salvar_grafico(ablacao: pd.DataFrame, melhor_por_qtd: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ordem = ablacao.sort_values("balanced_accuracy", ascending=True)
    cores = ["#2d6cdf" if removido == "nenhum" else "#d9822b" for removido in ordem["sensor_removido"]]
    axes[0].barh(ordem["cenario"], ordem["balanced_accuracy"] * 100, color=cores)
    axes[0].set_xlabel("Balanced accuracy (%)")
    axes[0].set_title("Impacto de retirar um MQ por vez")
    axes[0].grid(axis="x", alpha=0.2)

    axes[1].plot(
        melhor_por_qtd["qtd_sensores"],
        melhor_por_qtd["balanced_accuracy"] * 100,
        marker="o",
        color="#2d6cdf",
    )
    axes[1].set_xticks(list(range(1, len(SENSORES) + 1)))
    axes[1].set_xlabel("Quantidade de sensores MQ")
    axes[1].set_ylabel("Melhor balanced accuracy (%)")
    axes[1].set_title("Melhor combinacao em cada tamanho")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output / "grafico_ablacao_e_melhores_combinacoes.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--somente-ablacao",
        action="store_true",
        help="Executa somente baseline e as seis retiradas individuais.",
    )
    parser.add_argument(
        "--tolerancia-pp",
        type=float,
        default=1.0,
        help="Queda maxima, em pontos percentuais, para recomendar a menor combinacao pratica.",
    )
    parser.add_argument(
        "--remover-coleta-treino",
        help="Remove uma coleta do treino e refaz a compensacao ambiental. Ex.: C16.",
    )
    parser.add_argument(
        "--pasta-saida",
        help="Nome da pasta de resultado dentro de 06_07_melhor_modelo.",
    )
    args = parser.parse_args()

    df = pd.read_csv(DATASET)
    required = [TARGET, "Conjunto", *features_para(SENSORES)]
    faltantes = [coluna for coluna in required if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas ausentes no dataset: {faltantes}")

    mask_treino = df["Conjunto"].eq("Treino")
    coleta_removida = None
    if args.remover_coleta_treino:
        coleta_removida = localizar_coleta(args.remover_coleta_treino)
        mask_treino &= ~df["Coleta"].eq(coleta_removida)
        # O resultado de 93% foi produzido desta forma: a compensacao dos MQ
        # tambem e aprendida apenas com o treino restante, sem C16.
        df = corrigir_mq_pelo_treino(df, mask_treino)

    train_df = df[mask_treino].copy()
    test_df = df[df["Conjunto"] == "Teste"].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("A coluna Conjunto precisa conter Treino e Teste.")

    nome_saida = args.pasta_saida or (
        f"ablacao_sensores_sem_{args.remover_coleta_treino.upper()}_treino"
        if args.remover_coleta_treino
        else "ablacao_sensores"
    )
    output = BASE / nome_saida
    output.mkdir(exist_ok=True)

    # Parte 1: baseline e retirada de cada MQ isoladamente.
    cenarios_ablacao = [("baseline_todos_os_MQ", SENSORES, "nenhum")]
    cenarios_ablacao += [
        (f"sem_{sensor}", tuple(item for item in SENSORES if item != sensor), sensor)
        for sensor in SENSORES
    ]
    resultados_ablacao = []
    for cenario, sensores, removido in cenarios_ablacao:
        resultado = avaliar(train_df, test_df, sensores)
        resultado.update({"cenario": cenario, "sensor_removido": removido})
        resultados_ablacao.append(resultado)

    ablacao = pd.DataFrame(resultados_ablacao)
    baseline = float(ablacao.loc[ablacao["cenario"] == "baseline_todos_os_MQ", "balanced_accuracy"].iloc[0])
    ablacao["delta_balanced_accuracy_pp_vs_baseline"] = (
        (ablacao["balanced_accuracy"] - baseline) * 100
    )
    ablacao.to_csv(output / "ablacao_um_sensor_por_vez.csv", index=False, encoding="utf-8-sig")

    # Parte 2: todas as 63 combinacoes nao vazias. Isto evita concluir a melhor
    # dupla/trio apenas pela importancia ou pela ablacao individual.
    if args.somente_ablacao:
        combinacoes = ablacao.copy()
    else:
        resultados = []
        for tamanho in range(1, len(SENSORES) + 1):
            for sensores in itertools.combinations(SENSORES, tamanho):
                resultado = avaliar(train_df, test_df, sensores)
                resultado["cenario"] = f"{tamanho}_MQ__{'_'.join(sensores)}"
                resultados.append(resultado)
        combinacoes = pd.DataFrame(resultados)

    combinacoes["delta_balanced_accuracy_pp_vs_baseline"] = (
        (combinacoes["balanced_accuracy"] - baseline) * 100
    )
    combinacoes = combinacoes.sort_values(
        ["balanced_accuracy", "f1_macro", "accuracy", "qtd_sensores"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    combinacoes.to_csv(output / "todas_combinacoes_sensores.csv", index=False, encoding="utf-8-sig")

    melhor_por_qtd = (
        combinacoes.sort_values(
            ["qtd_sensores", "balanced_accuracy", "f1_macro", "accuracy"],
            ascending=[True, False, False, False],
        )
        .groupby("qtd_sensores", as_index=False)
        .first()
        .sort_values("qtd_sensores")
    )
    melhor_por_qtd.to_csv(output / "melhor_combinacao_por_quantidade.csv", index=False, encoding="utf-8-sig")

    melhor_absoluta = combinacoes.iloc[0]
    elegiveis = combinacoes[
        combinacoes["delta_balanced_accuracy_pp_vs_baseline"] >= -args.tolerancia_pp
    ].sort_values(["qtd_sensores", "balanced_accuracy"], ascending=[True, False])
    menor_pratica = elegiveis.iloc[0] if not elegiveis.empty else melhor_absoluta

    recomendacao = {
        "metodo": "Mesmo ExtraTrees e mesmo split 70/30 por Coleta do modelo-base; as tres variaveis ambientais foram mantidas em todos os cenarios.",
        "baseline_todos_os_6_MQ_balanced_accuracy": baseline,
        "melhor_combinacao_absoluta": melhor_absoluta.to_dict(),
        "menor_combinacao_dentro_da_tolerancia": menor_pratica.to_dict(),
        "tolerancia_pontos_percentuais": args.tolerancia_pp,
        "linhas_treino": len(train_df),
        "linhas_teste": len(test_df),
        "coleta_removida_do_treino": coleta_removida,
    }
    with (output / "recomendacao_sensores.json").open("w", encoding="utf-8") as arquivo:
        json.dump(recomendacao, arquivo, indent=2, ensure_ascii=False)

    linhas = [
        "# Ablacao e combinacoes de sensores MQ",
        "",
        "Todos os testes mantem o mesmo ExtraTrees, a mesma divisao Treino/Teste por coleta e as variaveis Soil_indice_0_1, Temp_C e Pres_kPa.",
        "Assim, a comparacao mede somente o efeito de usar ou retirar MQs.",
        "",
        f"- Baseline (6 MQ): {baseline:.2%} de balanced accuracy.",
        f"- Melhor combinacao absoluta: {melhor_absoluta['sensores_mantidos']} ({melhor_absoluta['qtd_sensores']} MQ), {melhor_absoluta['balanced_accuracy']:.2%}.",
        f"- Menor combinacao com queda de no maximo {args.tolerancia_pp:.1f} p.p. frente ao baseline: {menor_pratica['sensores_mantidos']} ({menor_pratica['qtd_sensores']} MQ), {menor_pratica['balanced_accuracy']:.2%}.",
        "",
        "A recomendacao de reducao e operacional para este split. Antes de fechar hardware, confirme a combinacao em novas coletas/ensaios independentes.",
    ]
    if coleta_removida:
        linhas.insert(4, f"A coleta {args.remover_coleta_treino.upper()} ({coleta_removida}) foi removida somente do treino; a compensacao ambiental foi recalculada usando o treino restante.")
    (output / "README.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    salvar_grafico(ablacao, melhor_por_qtd, output)
    print(json.dumps(recomendacao, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
