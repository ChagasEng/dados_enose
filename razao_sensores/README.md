# Razao entre sensores

Esta pasta testa a hipotese de usar divisoes entre sensores como novas features.

Ideia:

Quando varios sensores sobem ou descem juntos por influencia do ambiente, da caixa ou da concentracao geral dos gases, a razao entre dois sensores pode reduzir essa influencia comum. Por exemplo:

`MQ7 / MQ135`

ou

`MQ8 / MQ138`

Assim, o modelo passa a olhar mais para a relacao entre sensores do que para o valor absoluto de cada leitura.

## Resultado

O teste comparou:

- sensores MQ originais;
- somente razoes entre sensores;
- sensores MQ originais + razoes;
- razoes envolvendo `MQ7`;
- `MQ7` + razoes envolvendo `MQ7`.

Melhor resultado nesta rodada:

- Feature set: `mq_originais`
- Modelo: `extra_trees`
- Accuracy: `0.870297`
- Balanced accuracy: `0.869949`
- F1 macro: `0.869179`

Ou seja, a ideia das razoes faz sentido como teste cientifico, mas nesta validacao nao superou os sensores originais.

## Arquivos principais

- `scripts/rodar_razao_sensores.py`: script do experimento.
- `resultados/relatorios/relatorio_razao_sensores.txt`: resumo em texto.
- `resultados/metricas/ranking_razao_sensores.csv`: ranking dos modelos/features.
- `graficos/ranking_razao_sensores.png`: grafico comparativo.
- `modelos/melhor_modelo_razao_sensores.joblib`: melhor modelo desta rodada.
