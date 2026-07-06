# 3. Compensacao de umidade e temperatura

## Objetivo

Organizar as duas abordagens sugeridas para lidar com dependencia ambiental dos sensores MQ, principalmente temperatura e umidade.

## Colunas disponiveis

A base atual possui:

- `Temp.`: temperatura registrada na coleta.
- `Soil`: canal ambiental/solo, ainda precisa ser confirmado com Artur.
- `Pres.`: pressao registrada.

Nao existe coluna confirmada de umidade relativa do ar. Por isso, `Soil` nao deve ser tratado automaticamente como umidade de datasheet sem validacao.

## Abordagem 1: modelo aprende a compensar

Treinar um modelo recebendo MQ + ambiente:

`MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135`, `MQ138`, `Soil`, `Temp.`, `Pres.`

Vantagem: simples de comparar com MQ-only.

Risco: o modelo pode aprender atalho experimental de ambiente, e nao necessariamente o efeito biologico. Por isso a validacao precisa ser por coleta/grupo.

## Abordagem 2: correcao matematica

Aplicar uma correcao por sensor antes do modelo, usando fatores extraidos dos datasheets:

```text
sensor_corrigido = sensor_cru / fator_ambiente(temperatura, umidade)
```

Nesta pasta deixei um template para preencher os fatores de correcao. Sem esses fatores e sem confirmar o significado de `Soil`, a correcao matematica fica apenas como protocolo, nao como resultado final.

## Arquivos nesta pasta

- `documentos/protocolo_compensacao_temperatura_umidade.md`: roteiro das duas abordagens.
- `dados_base/antes_dia_20_pressao_filtrada_estrito_com_ambiente.csv`: base limpa com ambiente preservado.
- `templates/fatores_correcao_datasheet_template.csv`: tabela para preencher fatores por sensor.
- `scripts/aplicar_correcao_matematica_template.py`: script base para aplicar os fatores depois que forem definidos.
- `modelagem/`: ExtraTrees, rede neural e importancia usando MQ + ambiente.

## Modelagem com ambiente

Foi testada a abordagem em que o modelo recebe `Soil`, `Temp.` e `Pres.` junto com os sensores MQ:

- ExtraTrees accuracy: 91.51%.
- Rede neural accuracy: 89.94%.
- Feature mais importante no ExtraTrees: `MQ8`.
- Na leitura com ambiente, `Pres.`, `Soil` e `Temp.` tambem ficaram relevantes.
