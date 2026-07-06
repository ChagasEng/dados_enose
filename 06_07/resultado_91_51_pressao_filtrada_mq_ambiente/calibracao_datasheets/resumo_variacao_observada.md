# Variacao observada no dataset

Arquivo analisado: `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07\3_compensacao_umidade_temperatura\dados_base\antes_dia_20_pressao_filtrada_estrito_com_ambiente.csv`
Linhas: 63373

## Colunas ambientais

- `Soil`: min=1.0000, media=1321.1365, max=2971.0000, amplitude=2970.0000.
- `Temp.`: min=19.4000, media=36.8160, max=46.8000, amplitude=27.4000.
- `Pres.`: min=93.4700, media=93.6533, max=93.8800, amplitude=0.4100.

## Leitura

As colunas ambientais variam ao longo da base, entao faz sentido tratar `Temp.`, `Pres.` e a coluna `Soil` como possiveis fontes de variacao experimental. A interpretacao por datasheet, porem, depende de confirmar se `Soil` representa umidade relativa do ar ou outro sinal.
