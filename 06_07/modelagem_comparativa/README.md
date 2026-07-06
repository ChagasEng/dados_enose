# Modelagem comparativa 06/07

Foram rodados quatro cenarios, cada um dentro da respectiva pasta 1, 2, 3 e 4:

| Cenario | Base | Features | ExtraTrees acc. | Rede neural acc. | Top ExtraTrees |
| --- | --- | --- | ---: | ---: | --- |
| 01 | Antes do corte | MQ + ambiente | 87.58% | 82.74% | MQ8 |
| 02 | Pressao filtrada | MQ-only | 89.96% | 79.12% | MQ8 |
| 03 | Pressao filtrada | MQ + ambiente | 91.51% | 89.94% | MQ8 |
| 04 | Polido final | MQ-only | 89.96% | 79.12% | MQ8 |

## Leitura rapida

O melhor resultado foi o cenario 03, com ExtraTrees usando MQ + `Soil`, `Temp.` e `Pres.`, chegando a 91.51% de acuracia. O melhor MQ-only ficou em 89.96%, usando a base ja filtrada por pressao.

A importancia nativa do ExtraTrees indicou `MQ8` como principal feature em todos os cenarios. No cenario com ambiente, `Pres.`, `Soil` e `Temp.` tambem aparecem com peso relevante, por isso esse resultado precisa ser interpretado junto com a investigacao fisica e ambiental.

## Arquivos principais

- `comparativo_extra_trees_rede_neural_importancia.csv`
- `comparativo_extra_trees_rede_neural.png`
- `resumo_melhores_modelos_e_sensores.csv`

Cada pasta tambem possui uma subpasta `modelagem` com:

- metricas em JSON;
- matrizes de confusao;
- modelos `.joblib`;
- importancia nativa do ExtraTrees;
- importancia por permutacao do ExtraTrees;
- importancia por permutacao da rede neural.
