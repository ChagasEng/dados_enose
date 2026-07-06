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
- `datasheets_calibracao/`: verificacao dos datasheets MQ, pendencias de `Soil`/`Temp.`/`Pres.`, estatisticas de variacao e modelo pratico de calibracao.

## Modelagem com ambiente

Foi testada a abordagem em que o modelo recebe `Soil`, `Temp.` e `Pres.` junto com os sensores MQ:

- ExtraTrees accuracy: 91.51%.
- Rede neural accuracy: 89.94%.
- Feature mais importante no ExtraTrees: `MQ8`.
- Na leitura com ambiente, `Pres.`, `Soil` e `Temp.` tambem ficaram relevantes.

## Verificacao por datasheet

Os datasheets Winsen dos MQ confirmam curvas de temperatura/umidade (`Rs/Rso`) e referencia de sensibilidade (`Rs/R0`). Isso sustenta a necessidade de compensacao ambiental antes de defender que a resposta e apenas biologica.

O hardware ambiental foi confirmado: `Temp.` e `Pres.` vieram do BMP280, e `Soil` veio do Capacitive Soil Moisture Sensor V2.0. Com isso, `Temp.` pode ser usada como temperatura ambiental, `Pres.` como pressao/estabilidade fisica, e `Soil` como umidade do solo. Importante: `Soil` nao e umidade relativa do ar, entao nao deve ser usado como `RH` para corrigir os MQ por datasheet.
