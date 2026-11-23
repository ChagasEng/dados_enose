"""Calcula tabelas de metricas por leitura a partir das matrizes salvas."""

from pathlib import Path

import numpy as np
import pandas as pd

from gerar_matrizes import MODELS, OUTPUT, historical_matrices, loso_matrix


def metrics(matrix: np.ndarray) -> dict:
    matrix = matrix.astype(float)
    total = matrix.sum()
    true_positive = matrix.diagonal()
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    recall = np.divide(true_positive, support, out=np.zeros(2), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros(2), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(2), where=(precision + recall) > 0)
    return {
        "n_leituras": int(total),
        "acertos": int(true_positive.sum()),
        "acuracia_pct": float(100 * true_positive.sum() / total),
        "acuracia_balanceada_pct": float(100 * recall.mean()),
        "f1_macro_pct": float(100 * f1.mean()),
        "recall_pct": float(100 * recall[0]),
    }


def latex_table(data: pd.DataFrame, caption: str, label: str,
                scenario_column: bool = True) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \footnotesize",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \resizebox{\linewidth}{!}{%",
        r"    \begin{tabular}{" + ("llrrrr" if scenario_column else "lrrrr") + "}",
        r"        \hline",
        (r"        Cenário & Modelo & Acurácia (\%) & Acurácia bal. (\%) & Recall (\%) & F1 macro (\%) \\"
         if scenario_column else
         r"        Modelo & Acurácia (\%) & Acurácia bal. (\%) & Recall (\%) & F1 macro (\%) \\"),
        r"        \hline",
    ]
    for row in data.itertuples(index=False):
        values = [
            f"{row.acuracia_pct:.2f}".replace(".", ","),
            f"{row.acuracia_balanceada_pct:.2f}".replace(".", ","),
            f"{row.recall_pct:.2f}".replace(".", ","),
            f"{row.f1_macro_pct:.2f}".replace(".", ","),
        ]
        prefix = [row.cenario, row.modelo] if scenario_column else [row.modelo]
        lines.append("        " + " & ".join([*prefix, *values]) + r" \\")
        if scenario_column and row.modelo == "LDA":
            lines.append(r"        \hline")
    if not lines[-1].endswith(r"\hline"):
        lines.append(r"        \hline")
    lines.extend([r"    \end{tabular}", r"    }", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    rows = []
    for scenario, label in (("mq", "Cinco MQ"),
                            ("mq_ambiente", "Cinco MQ + ambiente")):
        for model in MODELS:
            if model[0] == "ANN + k-NN + SVM + LDA":
                continue  # O autor solicitou somente classificadores individuais.
            rows.append({"protocolo": "LOSO by pot", "cenario": label,
                         "modelo": "SVM linear" if model[0] == "SVM" else model[0],
                         "n_coletas": 31,
                         **metrics(loso_matrix(scenario, model))})
    historical = historical_matrices()
    for index, (title, matrix) in enumerate(historical):
        rows.append({"protocolo": "Historical split by collection",
                     "cenario": "Solo seco" if index < 2 else "Solo úmido",
                     "modelo": title.split(" · ")[0],
                     "n_coletas": 8 if index < 2 else 4,
                     **metrics(matrix)})
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "metricas_classificadores.csv", index=False)
    (OUTPUT / "tabela_metricas_loso.tex").write_text(latex_table(
        table.loc[table.protocolo.eq("LOSO by pot")],
        "Desempenho por leitura dos classificadores individuais nas 31 coletas, com validação LOSO por vaso.",
        "tab:metricas_classificadores_loso_filtrado",
    ), encoding="utf-8")
    (OUTPUT / "tabela_metricas_historico.tex").write_text(latex_table(
        table.loc[table.protocolo.eq("Historical split by collection")],
        "Desempenho por leitura no teste histórico por coleta, nos recortes de solo seco e úmido.",
        "tab:metricas_classificadores_historico",
    ), encoding="utf-8")
    print(table[["protocolo", "cenario", "modelo", "acuracia_pct",
                 "acuracia_balanceada_pct", "recall_pct", "f1_macro_pct"]]
          .to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
