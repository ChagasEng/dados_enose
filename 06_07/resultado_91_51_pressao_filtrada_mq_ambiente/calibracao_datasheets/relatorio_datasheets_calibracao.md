# Verificacao por datasheet para calibracao

## Conclusao curta

Os datasheets dos sensores MQ confirmam que a resposta desses sensores varia com temperatura e umidade. A propria forma tecnica sugerida pelos manuais e trabalhar com razao de resistencia, como `Rs/R0`, e comparar tudo contra uma condicao padrao de ensaio. Isso sustenta exatamente o pedido do professor: antes de interpretar a diferenca como efeito biologico da planta, precisamos compensar ou pelo menos controlar ambiente, pressao e possiveis falhas fisicas.

## Evidencia encontrada nos datasheets

Todos os MQ verificados usam a mesma logica geral: o material sensivel muda a condutividade conforme a concentracao de gas, e o manual apresenta curvas de sensibilidade. Alem disso, os manuais trazem uma curva tipica de temperatura/umidade usando `Rs/Rso`, ou seja, a resistencia medida em determinada temperatura/umidade comparada com uma resistencia de referencia em condicao padrao.

Resumo das referencias:

- `MQ2`: manual Winsen, gases inflamaveis/fumaca, padrao em 20 C +/- 2 C e 55% RH +/- 5%; curva `Rs/Rso` em propane.
- `MQ3`: foi localizado manual Winsen `MQ-3B`; alcool, padrao em 20 C +/- 2 C e 55% RH +/- 5%; confirmar se o componente fisico e MQ3 ou MQ-3B.
- `MQ7`: foi localizado manual Winsen `MQ-7B`; CO, padrao em 20 C +/- 2 C e 55% RH +/- 5%; esse modelo tambem depende do ciclo correto de aquecimento alto/baixo.
- `MQ8`: manual Winsen, hidrogenio, padrao em 20 C +/- 2 C e 55% RH +/- 5%; curva `Rs/Rso` em H2.
- `MQ135`: manual Winsen, gases de qualidade do ar/VOCs; padrao em 20 C +/- 2 C e 55% RH +/- 5%; curva `Rs/Rso` apresentada no manual.
- `MQ138`: manual Winsen, VOCs; padrao em 20 C +/- 2 C e 55% RH +/- 5%; curva `Rs/Rso` em tolueno.

## Silicone e vedacao

Outro ponto importante para a nossa caixa: os manuais dos MQ alertam para evitar exposicao a vapores/compostos organicos de silicone, porque isso pode reduzir ou alterar a sensibilidade. Entao a suspeita de influencia da vedacao de silicone e tecnicamente plausivel. Ainda nao da para cravar que foi a causa, mas e uma hipotese forte para testar com controle fisico: comparar caixa com vedacao atual, caixa sem silicone exposto e/ou sensor isolado em ambiente controlado.

## Sensores ambientais confirmados

A base atual tem as colunas:

```text
Soil, Temp., Pres., MQ2, MQ3, MQ7, MQ8, MQ135, MQ138
```

Com a confirmacao do hardware:

- `Temp.` e `Pres.` vieram do `BMP280`.
- `Soil` veio do `Capacitive Soil Moisture Sensor V2.0`.

Isso melhora bastante a rastreabilidade. O BMP280 e um sensor Bosch de pressao absoluta e temperatura. A faixa de pressao do datasheet e 300 a 1100 hPa, ou 30 a 110 kPa; portanto os valores do dataset em torno de `93.x` fazem sentido se estiverem salvos em kPa.

O ponto critico e o `Soil`: o sensor capacitivo V2.0 mede umidade do solo por saida analogica. Ele nao mede umidade relativa do ar. Portanto, ele pode entrar como variavel do vaso/solo no modelo, mas nao deve ser usado diretamente como `RH` para corrigir os MQ pelas curvas de temperatura/umidade dos datasheets.

## Limitacao que continua

Ainda falta uma medicao de umidade relativa do ar dentro/ao redor da camara. Para aplicar a correcao matematica completa dos MQ por datasheet, precisamos de `Temp.` e `RH`. O BMP280 entrega temperatura e pressao, mas nao entrega umidade relativa. Se o sensor fosse BME280, haveria RH; como e BMP280, nao ha.

## Como a correcao deve ser feita agora

O caminho correto e:

1. Manter `Temp.` e `Pres.` como variaveis ambientais confirmadas pelo BMP280.
2. Manter `Soil` como variavel de umidade do solo/condicao do vaso, nao como umidade do ar.
3. Confirmar no hardware de cada MQ os valores de `Vc`, `VH`, `RL`, resolucao ADC e referencia eletrica.
4. Obter ou medir `R0` de cada sensor em condicao de referencia.
5. Converter a leitura bruta para `Rs`, quando a leitura representar `VRL`:

```text
Rs = RL * (Vc / VRL - 1)
```

6. Normalizar:

```text
razao = Rs / R0
```

7. Extrair dos datasheets os fatores `Rs/Rso` por temperatura e umidade.
8. Para a parte de temperatura, usar `Temp.` do BMP280.
9. Para a parte de umidade relativa, adicionar/recuperar uma medida de RH real; nao usar `Soil` como substituto direto.
10. Interpolar o fator ambiente para cada linha da coleta:

```text
sensor_corrigido = sensor_normalizado / fator_ambiente(Temp., RH)
```

11. Rodar novamente ExtraTrees/rede neural e comparar:

```text
MQ cru
MQ com corte por pressao
MQ + ambiente
MQ corrigido por datasheet
```

## Relacao com o resultado de 91,51%

O resultado de 91,51% com ExtraTrees usando `MQ + Soil + Temp. + Pres.` reforca que as variaveis ambientais ajudam o classificador. Isso e bom para desempenho, mas tambem mostra que a classificacao pode estar usando informacao fisica/ambiental da caixa. Para defender cientificamente, precisamos separar o que e sinal biologico dos MQ do que e variacao de temperatura, umidade, pressao, alimentacao ou vedacao.

## Pendencias objetivas

- Confirmar se `MQ3` e `MQ7` fisicos sao as versoes Winsen `MQ-3B` e `MQ-7B` ou outra variante.
- Confirmar se a coluna `Pres.` foi salva em kPa; pelo valor `93.x`, ela parece estar em kPa.
- Confirmar a conversao ADC usada na coluna `Soil` para transformar leitura analogica em porcentagem/calibracao seco-molhado.
- Medir umidade relativa do ar nos proximos ensaios, porque o BMP280 nao mede RH.
- Confirmar se houve medicao direta de alimentacao/corrente/tensao durante as coletas.
- Confirmar `RL`, `Vc`, `VH`, ADC e `R0` de cada MQ.
