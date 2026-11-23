"""Recalcula ICs descritivos por vaso do ET com nove entradas fixas."""

from pathlib import Path

import pandas as pd

from avaliar_base_completa_parecer import bootstrap_por_vaso


BASE = Path(__file__).resolve().parent / "baselines_fixos_base_completa"
FILES = {
    "leitura": "predicoes_nove_entradas.csv",
    "coleta": "coletas_nove_entradas.csv",
    "vaso": "vasos_nove_entradas.csv",
}


def main() -> None:
    rows = []
    for unidade, filename in FILES.items():
        frame = pd.read_csv(BASE / filename)
        rows.append(bootstrap_por_vaso(frame, unidade))
    result = pd.DataFrame(rows)
    out = BASE / "intervalos_bootstrap_nove_entradas.csv"
    result.to_csv(out, index=False)
    print(result[["unidade", "limite_inferior_95", "limite_superior_95"]])


if __name__ == "__main__":
    main()
