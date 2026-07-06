# Sensores ambientais confirmados

## BMP280

Usado para:

- `Temp.`
- `Pres.`

Leitura tecnica:

- O BMP280 e um sensor digital de pressao barometrica absoluta e temperatura.
- O datasheet Bosch informa faixa de pressao de 300 a 1100 hPa, equivalente a 30 a 110 kPa.
- Os valores do dataset em torno de `93.x` batem com kPa, ou seja, aproximadamente `930-939 hPa`.
- O BMP280 nao mede umidade relativa do ar.

## Capacitive Soil Moisture Sensor V2.0

Usado para:

- `Soil`

Leitura tecnica:

- Mede umidade do solo por capacitancia, nao por resistencia.
- A saida e analogica.
- A faixa comum do modulo e alimentacao 3.3-5 V e saida 0-3.0 VDC.
- Essa coluna pode representar condicao do vaso/solo, mas nao e `RH` do ar.

## Consequencia para calibracao dos MQ

A correcao por datasheet dos MQ precisa de temperatura e umidade relativa do ar. Agora temos temperatura pelo BMP280, mas ainda nao temos RH. Portanto:

- usar `Temp.` na compensacao por temperatura;
- usar `Pres.` para estabilidade fisica e possivel efeito da caixa;
- usar `Soil` como contexto do vaso/solo;
- nao usar `Soil` como substituto direto de umidade relativa do ar.
