# Resultado 91,51% - pressao filtrada + MQ + ambiente

Este pacote separa o melhor resultado da rodada 06/07.

## Cenario

- Base: pressao filtrada em modo estrito.
- Features usadas: `MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135`, `MQ138`, `Soil`, `Temp.`, `Pres.`
- Split: 70/30 por grupos de `Coleta` dentro de cada classe.
- Linhas do dataset: 63.373.
- Linhas de teste: 22.115.
- Coletas totais: 36.

## Resultado principal

### ExtraTrees

- Accuracy: 91,51%.
- Balanced accuracy: 91,28%.
- F1 macro: 91,43%.

### Rede neural MLP

- Accuracy: 89,94%.
- Balanced accuracy: 90,12%.
- F1 macro: 89,94%.

## Importancia das variaveis

Na importancia nativa do ExtraTrees, as principais variaveis foram:

1. `MQ8`
2. `Pres.`
3. `Soil`
4. `MQ7`
5. `Temp.`

Na importancia por permutacao, `Pres.` ficou muito forte. Isso significa que o modelo com ambiente performou melhor, mas tambem reforca que a interpretacao precisa considerar efeito fisico/ambiental, nao apenas efeito biologico da planta.

## Conteudo da pasta

- `dataset_usado/`: CSV usado nesta rodada.
- `modelagem/metricas/`: metricas do ExtraTrees e da rede neural.
- `modelagem/matrizes/`: matrizes de confusao em CSV e PNG.
- `modelagem/importancias/`: importancia nativa e por permutacao.
- `modelagem/graficos/`: graficos de importancia.
- `modelagem/modelos/`: modelos `.joblib`.
- `modelagem/relatorios/`: resumo da modelagem.
- `comparativo_geral/`: comparativo com os outros cenarios.
- `calibracao_datasheets/`: verificacao dos datasheets MQ e roteiro de calibracao/correcao ambiental.
- `rodar_modelagem_extra_trees_rede_neural_importancia.py`: script para reproduzir a rodada.

## Leitura curta

Este foi o melhor resultado numerico da rodada, mas como `Pres.`, `Soil` e `Temp.` aparecem com peso relevante, ele deve ser apresentado como "modelo com compensacao/variaveis ambientais" e nao como prova isolada de que os MQ sozinhos explicam toda a classificacao.

## Observacao de calibracao

Os datasheets dos MQ mostram dependencia de temperatura/umidade, e tambem alertam sobre influencia de compostos de silicone. Portanto, o resultado de 91,51% deve ser discutido junto da pasta `calibracao_datasheets/`: ele e forte como desempenho, mas a interpretacao cientifica precisa separar sinal dos MQ de efeito ambiental/fisico da caixa.

Com a confirmacao do hardware, `Temp.` e `Pres.` vieram do BMP280 e `Soil` veio do Capacitive Soil Moisture Sensor V2.0. Isso fecha a origem das colunas, mas tambem mostra que ainda nao temos umidade relativa do ar: `Soil` e umidade do solo, nao `RH`. Para a correcao completa dos MQ por datasheet, sera necessario medir RH nos proximos ensaios.
