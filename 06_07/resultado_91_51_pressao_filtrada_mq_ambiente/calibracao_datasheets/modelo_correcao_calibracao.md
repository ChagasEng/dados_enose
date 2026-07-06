# Modelo pratico de correcao/calibracao

## Objetivo

Montar uma correcao que use o que o datasheet realmente informa: a leitura dos MQ muda por concentracao de gas, temperatura e umidade. A calibracao precisa transformar a leitura bruta em uma grandeza comparavel e depois remover, quando possivel, o fator ambiental.

## Fluxo recomendado

1. Manter o corte por pressao ja feito para remover trechos fisicamente instaveis.
2. Converter cada MQ para resistencia `Rs`, se tivermos `VRL`, `Vc` e `RL`.
3. Normalizar cada sensor por `R0`.
4. Usar `Temp.` do BMP280 e uma medida real de umidade relativa do ar para buscar o fator do datasheet.
5. Aplicar a correcao linha a linha.
6. Rodar novamente os classificadores com o mesmo split por `Coleta`.
7. Comparar acuracia, balanced accuracy, matriz de confusao e importancia de sensores.

## Formula base

Quando a coluna bruta representar a tensao sobre a resistencia de carga (`VRL`):

```text
Rs = RL * (Vc / VRL - 1)
sensor_normalizado = Rs / R0
sensor_corrigido = sensor_normalizado / fator_ambiente(Temp., RH)
```

Quando a coluna bruta for apenas contagem ADC, primeiro precisa converter:

```text
VRL = leitura_adc / adc_max * Vref
```

Depois aplica a mesma formula de `Rs`.

## O que mudou com a confirmacao do hardware

Agora sabemos que:

- `Temp.` e `Pres.` vem do BMP280.
- `Soil` vem do Capacitive Soil Moisture Sensor V2.0.

Isso permite tratar `Temp.` e `Pres.` com datasheet do BMP280. A pressao na faixa `93.x` fica coerente com kPa, pois o BMP280 trabalha em pressao barometrica absoluta e sua faixa de 300 a 1100 hPa equivale a 30 a 110 kPa.

## Por que ainda nao aplicar como resultado fechado

Ainda faltam tres informacoes criticas:

- `Soil` e umidade do solo, nao umidade relativa do ar.
- Nao temos uma coluna de RH do ar dentro/ao redor da caixa.
- Nao temos `RL`, `Vc`, `Vref`, ADC e `R0` confirmados para cada MQ.

Sem isso, qualquer correcao matematica completa de temperatura + umidade seria mais uma simulacao do que uma calibracao real. A abordagem valida por enquanto e: usar `Soil`, `Temp.` e `Pres.` como variaveis auxiliares no modelo, relatar que elas influenciam, aplicar somente a parte de temperatura quando os parametros eletricos dos MQ estiverem fechados, e preparar a coleta de RH para aplicar a correcao por datasheet.

## Entregavel para o professor

Texto curto:

"Os datasheets dos sensores MQ indicam que a resposta nao depende apenas do gas alvo, mas tambem de temperatura e umidade, usando curvas de referencia em torno de 20 C e 55% RH. No hardware usado, `Temp.` e `Pres.` vieram do BMP280, enquanto `Soil` veio de um sensor capacitivo de umidade do solo V2.0. Portanto, `Temp.` pode ser usada como eixo ambiental para compensacao, `Pres.` ajuda a detectar estabilidade fisica da caixa, e `Soil` deve ser tratado como condicao do vaso/solo, nao como umidade relativa do ar. Para a correcao completa por datasheet ainda e necessario medir RH do ar e fechar os parametros eletricos dos MQ (`RL`, `Vc`, `Vref`, ADC e `R0`)."
