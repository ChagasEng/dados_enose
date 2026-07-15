# LEIA PRIMEIRO - sensores corrigidos por datasheet

## Mensagem principal

Nesta pasta, o melhor modelo usa sensores MQ corrigidos com base na dependencia ambiental indicada nos datasheets e nas variaveis ambientais confirmadas por hardware.

## O que foi confirmado

- `Temp.` e `Pres.` vieram do BMP280.
- `Soil` veio do Capacitive Soil Moisture Sensor V2.0.
- Os datasheets dos MQ indicam que a resposta varia com temperatura e umidade.

## Como a correcao foi aplicada

Como nao temos umidade relativa do ar (`RH`) na base, nao foi possivel aplicar a calibracao fisica completa por datasheet:

```text
Rs/R0 corrigido por Temp. + RH
```

O que foi aplicado foi uma compensacao ambiental estatistica, usando os sensores confirmados:

```text
MQ_corrigido_env = MQ_cru - efeito_estimado_de(Soil_indice_0_1, Temp_C, Pres_kPa)
```

Essa correcao remove dos MQ a componente linear associada ao ambiente interno/condicao do vaso, estimada somente no treino.

## Como falar sem se comprometer errado

Use esta frase:

```text
Os sinais MQ foram corrigidos por uma compensacao ambiental orientada pelos datasheets, usando temperatura e pressao do BMP280 e umidade do solo do sensor capacitivo. Como ainda nao temos RH do ar, esta e uma correcao estatistica orientada por datasheet, nao a calibracao fisica final completa Rs/R0.
```

## Resultado

```text
ExtraTrees - MQ corrigido + ambiente
Accuracy: 93,20%
Balanced accuracy: 93,00%
F1 macro: 93,15%
```
