# Sensores ambientais confirmados

O hardware foi confirmado:

- `Temp.`: BMP280.
- `Pres.`: BMP280.
- `Soil`: Capacitive Soil Moisture Sensor V2.0.

## Leitura correta

- `Temp.` deve ser tratada como temperatura medida pelo BMP280.
- `Pres.` deve ser tratada como pressao barometrica absoluta medida pelo BMP280. Os valores `93.x` provavelmente estao em kPa, equivalentes a aproximadamente `930-939 hPa`.
- `Soil` deve ser tratada como leitura analogica de umidade do solo, nao como umidade relativa do ar.

## Pendencias que continuam

1. Confirmar se `Pres.` foi salva em kPa ou hPa dividido por 10.
2. Confirmar a escala ADC usada em `Soil` e se houve calibracao seco/molhado.
3. Confirmar onde o BMP280 estava fisicamente instalado: dentro da caixa, fora da caixa ou na linha de ar.
4. Confirmar se existe medicao direta de alimentacao, corrente ou tensao da placa.
5. Medir umidade relativa do ar nos proximos ensaios; o BMP280 nao fornece RH.

## Decisao para modelagem

- `Temp.` e `Pres.` podem continuar no modelo como variaveis ambientais confirmadas.
- `Soil` pode continuar no modelo como condicao do vaso/solo.
- `Soil` nao deve ser usado como `RH` nos calculos de correcao dos MQ por datasheet.
