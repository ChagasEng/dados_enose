# Fontes verificadas

Fontes oficiais consultadas para embasar a parte de calibracao/correcao:

- MQ2: https://www.winsen-sensor.com/d/files/newpdf/mq-2-%28ver1_6%29---manual.pdf
- MQ3: https://www.winsen-sensor.com/d/files/manual/mq-3b.pdf
- MQ7: https://www.winsen-sensor.com/d/files/manual/mq-7b.pdf
- MQ8: https://www.winsen-sensor.com/d/files/mq-8-%28ver1_6%29---manual.pdf
- MQ135: https://www.winsen-sensor.com/d/files/PDF/Semiconductor%20Gas%20Sensor/MQ135%20%28Ver1.4%29%20-%20Manual.pdf
- MQ138: https://www.winsen-sensor.com/d/files/PDF/Semiconductor%20Gas%20Sensor/MQ138%20%28Ver1.4%29%20-%20Manual.pdf
- BMP280: https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmp280-ds001.pdf
- Capacitive Soil Moisture Sensor V2.0: https://rajguruelectronics.com/Product/5538/Capacitive%20Soil%20Moisture%20Sensor%20V2%281%29.0.pdf

## Pontos tecnicos em comum

- Os sensores MQ trabalham com alteracao de condutividade/resistencia do material sensivel.
- Os manuais apresentam curva de sensibilidade em `Rs/R0`.
- Os manuais apresentam curva de temperatura/umidade em `Rs/Rso`.
- A condicao padrao aparece em torno de `20 C +/- 2 C` e `55% RH +/- 5%`.
- Os manuais alertam para evitar compostos/vapores organicos de silicone, pois podem afetar a sensibilidade.

## Pontos tecnicos dos sensores ambientais

- O BMP280 mede pressao absoluta e temperatura. Ele nao mede umidade relativa.
- A faixa de pressao do BMP280 e 300 a 1100 hPa, equivalente a 30 a 110 kPa.
- O Capacitive Soil Moisture Sensor V2.0 mede umidade do solo por sensoriamento capacitivo e entrega saida analogica.
- O sensor de solo nao deve ser tratado como RH do ar para corrigir os MQ.
