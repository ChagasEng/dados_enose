"""IC binomial exato de Clopper-Pearson para decisões por vaso em LOSO."""

from pathlib import Path

import pandas as pd
from scipy.stats import beta


ROOT = Path(__file__).resolve().parent
BASELINES = ROOT / "baselines_fixos_base_completa"
SELECTED = ROOT / "avaliacao_parecer_base_completa" / "previsoes_por_vaso.csv"
OUT = ROOT / "avaliacao_parecer_base_completa" / "intervalos_exatos_por_vaso.csv"


def clopper_pearson(correct: int, total: int, alpha: float = .05) -> tuple[float, float]:
    low = 0.0 if correct == 0 else float(beta.ppf(alpha / 2, correct, total - correct + 1))
    high = 1.0 if correct == total else float(beta.ppf(1 - alpha / 2, correct + 1, total - correct))
    return low, high


def main() -> None:
    paths = {"cinco_mq_ambiente_selecionado": SELECTED}
    paths.update({
        p.stem.removeprefix("vasos_"): p
        for p in BASELINES.glob("vasos_*.csv")
    })
    rows = []
    for name, path in paths.items():
        frame = pd.read_csv(path)
        n = len(frame)
        k = int(frame.Classe.eq(frame.predicao).sum())
        if n != 10 or frame.Vaso_id.nunique() != 10:
            raise AssertionError(f"Esperadas dez decisões por vaso: {name}")
        lower, upper = clopper_pearson(k, n)
        rows.append({
            "modelo": name, "acertos": k, "vasos": n,
            "proporcao": k / n, "limite_inferior_95": lower,
            "limite_superior_95": upper,
            "metodo": "Clopper-Pearson; condicional ao modelo e a escolha das entradas",
        })
    result = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
